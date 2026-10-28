"""Hardware-response layer: drive frames in, measured currents out.

Implementation note: this module intentionally does not import the image
processor or the readout classifier.  A real acquisition service can replace
:class:`HttpHardwareBackend` without touching either end of the pipeline.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Protocol
from urllib.request import Request, urlopen

import numpy as np

from .contracts import (
    DRIVE_CHANNELS,
    RESPONSE_CHANNELS,
    RESPONSE_POINTS,
    RESPONSE_ROWS,
    DriveSignals,
    HardwareResponse,
    LayerUnavailable,
)

_METADATA = (
    "subset_sample_index",
    "label_index",
    "class_name",
    "source_sample_index",
    "sample_id",
)
_RESPONSE_RE = re.compile(r"(.+)_row(\d{2})_(?:step(\d{2})|tail)_A")
_INPUT_COLUMNS = (
    "subset_sample_index",
    "source_sample_index",
    "sample_id",
    "label_index",
    "class_name",
    "step",
    "patch_row",
    "patch_col",
    *RESPONSE_CHANNELS,
)
_MIN_CURRENT = -1e-3
_MAX_CURRENT = 1e-3


class HardwareBackend(Protocol):
    name: str
    measured: bool

    def acquire(self, drive: DriveSignals) -> HardwareResponse: ...


def response_from_nested(value: Any) -> np.ndarray:
    """Validate an HTTP/JSON response shaped [9,12,65], in amperes."""
    if isinstance(value, dict):
        value = value.get("currents_amperes", value.get("response"))
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (len(RESPONSE_CHANNELS), RESPONSE_ROWS, RESPONSE_POINTS):
        raise ValueError(
            "hardware response must be [9,12,65] in amperes, "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("hardware response contains non-finite current values")
    if array.min() < _MIN_CURRENT or array.max() > _MAX_CURRENT:
        raise ValueError("hardware response is outside the plausible ±1 mA range")
    return array


def _read_response_csv(path: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = set(_METADATA) - set(header)
        if missing:
            raise ValueError(f"response CSV is missing metadata columns: {sorted(missing)}")
        coordinates: dict[tuple[int, int, int], str] = {}
        for column in header:
            if column in _METADATA:
                continue
            match = _RESPONSE_RE.fullmatch(column)
            if not match or match.group(1) not in RESPONSE_CHANNELS:
                raise ValueError(f"unsupported response column: {column}")
            channel = RESPONSE_CHANNELS.index(match.group(1))
            row = int(match.group(2)) - 1
            point = int(match.group(3)) - 1 if match.group(3) else 64
            if not 0 <= row < RESPONSE_ROWS or not 0 <= point < RESPONSE_POINTS:
                raise ValueError(f"response coordinate out of range: {column}")
            coordinates[(channel, row, point)] = column
        if len(coordinates) != len(RESPONSE_CHANNELS) * RESPONSE_ROWS * RESPONSE_POINTS:
            raise ValueError("response CSV must contain all 7020 current columns")
        rows = list(reader)
    if not rows:
        raise ValueError("response CSV has no sample rows")
    rows.sort(key=lambda row: int(row["subset_sample_index"]))
    order = [
        coordinates[(channel, row, point)]
        for channel in range(len(RESPONSE_CHANNELS))
        for row in range(RESPONSE_ROWS)
        for point in range(RESPONSE_POINTS)
    ]
    currents = np.asarray(
        [[float(row[name]) for name in order] for row in rows], dtype=np.float64
    ).reshape(-1, len(RESPONSE_CHANNELS), RESPONSE_ROWS, RESPONSE_POINTS)
    if not np.isfinite(currents).all():
        raise ValueError("response CSV contains non-finite currents")
    metadata = [{key: row[key] for key in _METADATA} for row in rows]
    if len({row["sample_id"] for row in metadata}) != len(metadata):
        raise ValueError("response CSV contains duplicate sample_id values")
    return currents, metadata


def _read_input_csv(path: Path) -> dict[int, DriveSignals]:
    """Read the 64-row-per-sample optical input handoff CSV."""
    required = set(_INPUT_COLUMNS)
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = required - set(header)
        if missing:
            raise ValueError(f"input CSV is missing columns: {sorted(missing)}")
        for row in reader:
            grouped.setdefault(int(row["subset_sample_index"]), []).append(row)
    drives: dict[int, DriveSignals] = {}
    for sample_index, rows in grouped.items():
        if len(rows) != 64:
            raise ValueError(f"sample {sample_index} has {len(rows)} frames, expected 64")
        rows.sort(key=lambda row: int(row["step"]))
        first = rows[0]
        sample_id = first["sample_id"]
        if any(
            row["sample_id"] != sample_id
            or row["label_index"] != first["label_index"]
            or row["class_name"] != first["class_name"]
            for row in rows
        ):
            raise ValueError(f"sample {sample_index} has inconsistent metadata")
        if [int(row["step"]) for row in rows] != list(range(1, 65)):
            raise ValueError(f"sample {sample_index} is missing a scan step")
        frames = np.asarray(
            [[float(row[channel]) for channel in RESPONSE_CHANNELS] for row in rows],
            dtype=np.float32,
        )
        tonic = frames[:, :3].copy()
        texture = np.zeros_like(tonic)
        drives[sample_index] = DriveSignals(
            frames=frames,
            sample_id=sample_id,
            source="recorded-processing-output",
            patches_rgb=tonic,
            texture=texture,
            config={
                "encoding": "graded_q000",
                "scan": "row-alternating-snake",
                "protocol_version": "1.0",
                "recorded": True,
            },
        )
    if not drives:
        raise ValueError("input CSV has no complete samples")
    return drives


class RecordedHardwareDataset:
    """Pair local optical handoff rows with the measured current CSV."""

    def __init__(self, responses_path: Path | str, inputs_path: Path | str):
        self.responses_path = Path(responses_path)
        self.inputs_path = Path(inputs_path)
        self.currents, self.metadata = _read_response_csv(self.responses_path)
        self.drives = _read_input_csv(self.inputs_path)
        if len(self.currents) != len(self.drives):
            raise ValueError("response and input files contain different sample counts")
        self._response_by_id = {
            row["sample_id"]: i for i, row in enumerate(self.metadata)
        }
        if len(self._response_by_id) != len(self.currents):
            raise ValueError("duplicate sample identifiers in response file")
        missing_inputs = set(self._response_by_id) - {
            drive.sample_id for drive in self.drives.values()
        }
        if missing_inputs:
            raise ValueError(
                f"input file is missing {len(missing_inputs)} measured sample(s)"
            )

    def sample_summaries(self) -> list[dict[str, Any]]:
        result = []
        for index, row in enumerate(self.metadata):
            result.append(
                {
                    "index": index,
                    "sample_id": row["sample_id"],
                    "source_sample_index": int(row["source_sample_index"]),
                    "label_index": int(row["label_index"]),
                    "class_name": row["class_name"],
                }
            )
        return result

    def sample_id_for_index(self, sample_index: int) -> str:
        if not isinstance(sample_index, int) or not 0 <= sample_index < len(self.metadata):
            raise ValueError("sample_index is out of range")
        return self.metadata[sample_index]["sample_id"]

    def drive(self, sample_index: int) -> DriveSignals:
        sample_id = self.sample_id_for_index(sample_index)
        external_index = int(self.metadata[sample_index]["subset_sample_index"])
        if external_index in self.drives:
            drive = self.drives[external_index]
            if drive.sample_id == sample_id:
                return drive
        for drive in self.drives.values():
            if drive.sample_id == sample_id:
                return drive
        raise LayerUnavailable(f"no recorded drive found for {sample_id}")

    def response(self, sample_index: int) -> HardwareResponse:
        sample_id = self.sample_id_for_index(sample_index)
        row = self.metadata[sample_index]
        return HardwareResponse(
            currents=self.currents[sample_index],
            sample_id=sample_id,
            backend="recorded-measured-currents",
            measured=True,
            metadata={
                "subset_sample_index": int(row["subset_sample_index"]),
                "source_sample_index": int(row["source_sample_index"]),
                "label_index": int(row["label_index"]),
                "class_name": row["class_name"],
                "source_file": self.responses_path.name,
            },
        )


class RecordedHardwareBackend:
    """Adapter that replays a trusted, locally recorded hardware response."""

    name = "recorded-measured-currents"
    measured = True

    def __init__(self, dataset: RecordedHardwareDataset):
        self.dataset = dataset

    def acquire(self, drive: DriveSignals) -> HardwareResponse:
        for sample_index, row in enumerate(self.dataset.metadata):
            if row["sample_id"] != drive.sample_id:
                continue
            return self.dataset.response(sample_index)
        raise LayerUnavailable(
            "recorded replay only accepts sample IDs present in the measured "
            "response dataset; use an HTTP hardware backend for new images"
        )


class HttpHardwareBackend:
    """Talk to the team's acquisition service using a small JSON protocol.

    Request body::

        {
          "protocol": "plantvillage-rc/hardware-response/v1",
          "sample_id": "...",
          "frames": [[9 floats], ... 64 rows],
          "channels": [...]
        }

    Response body::

        {"currents_amperes": [[[65 floats], ... 12 rows], ... 9 channels]}
    """

    name = "http-hardware"
    measured = True

    def __init__(self, url: str, *, timeout: float = 60.0):
        if not url.startswith(("http://", "https://")):
            raise ValueError("hardware URL must start with http:// or https://")
        self.url = url
        self.timeout = float(timeout)

    def acquire(self, drive: DriveSignals) -> HardwareResponse:
        payload = json.dumps(
            {
                "protocol": "plantvillage-rc/hardware-response/v1",
                "sample_id": drive.sample_id,
                "channels": list(DRIVE_CHANNELS),
                "frames": drive.frames.round(8).tolist(),
                "expected_response_shape": [
                    len(RESPONSE_CHANNELS),
                    RESPONSE_ROWS,
                    RESPONSE_POINTS,
                ],
                "units": "amperes",
            }
        ).encode("utf-8")
        request = Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as error:  # urllib raises several platform-specific types
            raise LayerUnavailable(f"hardware acquisition failed: {error}") from error
        return HardwareResponse(
            currents=response_from_nested(body),
            sample_id=drive.sample_id,
            backend=self.name,
            measured=True,
            metadata={"endpoint": self.url},
        )


def response_grid(currents: np.ndarray) -> list[list[float]]:
    """Compact visual summary: 9 channel means over 64 regular points."""
    values = np.asarray(currents, dtype=np.float64)
    return (values[:, :, :64].mean(axis=1) * 1e6).round(6).tolist()
