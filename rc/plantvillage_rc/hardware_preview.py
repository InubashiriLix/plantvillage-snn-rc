"""Read and validate the small, frame-oriented hardware handoff CSV."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .hardware_export import INPUT9_COLUMNS, METADATA_COLUMNS
from .preprocessing import grid_geometry


def read_preview(csv_path: Path | str, mapping_path: Path | str):
    """Return RGB [N,192], labels [N], signals [N,64,9], in file sample order.

    RGB flattening is snake-frame-major, interleaved R,G,B, not channel-first.
    Reject corrupt frame order, changing sample labels and invalid signals.
    """
    with Path(mapping_path).open(newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle))
    mapping = {int(row["label_index"]): row["class_name"] for row in entries}
    if len(mapping) != len(entries) or sorted(mapping) != list(range(len(mapping))):
        raise ValueError("class mapping must have unique, contiguous labels from zero")
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(METADATA_COLUMNS + INPUT9_COLUMNS):
            raise ValueError("unexpected hardware preview CSV columns")
        rows = list(reader)
    if not rows or len(rows) % 64:
        raise ValueError("expected complete samples of 64 frames")
    signals, labels, seen = [], [], set()
    _, _, order = grid_geometry((8, 8))
    for start in range(0, len(rows), 64):
        block = rows[start:start + 64]
        first = block[0]
        identity = (first["sample_index"], first["sample_id"])
        label = int(first["label_index"])
        if identity in seen or mapping.get(label) != first["class_name"]:
            raise ValueError("duplicate sample or incorrect class mapping")
        seen.add(identity)
        for step, (row, position) in enumerate(zip(block, order)):
            if ((row["sample_index"], row["sample_id"]) != identity
                or int(row["label_index"]) != label
                or row["class_name"] != mapping[label]
                or int(row["step"]) != step + 1
                or int(row["frame_index"]) != step
                or (int(row["patch_row"]), int(row["patch_col"])) != position):
                raise ValueError("inconsistent sample, frame order or snake coordinates")
        values = np.asarray([[float(row[key]) for key in INPUT9_COLUMNS] for row in block], dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("signals must be finite normalized levels in [0,1]")
        if values[0, 3:].any() or ((values[:, 3:6] > 0) & (values[:, 6:9] > 0)).any():
            raise ValueError("invalid first-frame or simultaneous ON/OFF events")
        signals.append(values)
        labels.append(label)
    signals = np.stack(signals)
    return signals[:, :, :3].reshape(-1, 192), np.asarray(labels, dtype=np.int64), signals
