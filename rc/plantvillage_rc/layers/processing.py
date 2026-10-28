"""Processing layer: image bytes in, normalized 64 x 9 drive frames out.

This module knows nothing about hardware acquisition or classification.  Its
output is the stable hand-off contract :class:`DriveSignals`.
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..preprocessing import (
    batch_relative_difference,
    extract_texture,
    extract_tonic,
    make_graded_rc_inputs,
    normalize_texture,
)
from .contracts import DriveSignals

_IMAGE_SIZE = 64
_PATCH_GRID = (8, 8)


@dataclass(frozen=True)
class ProcessingConfig:
    """Deterministic image-to-drive parameters.

    The graded event defaults are the training-set q000 encoding shipped with
    the ten-class hardware model.  They are calibration constants, not values
    fitted at inference time.
    """

    image_size: int = _IMAGE_SIZE
    patch_grid: tuple[int, int] = _PATCH_GRID
    graded_thresholds: tuple[float, float, float] = (0.0, 0.0, 0.0)
    on_scales: tuple[float, float, float] = (
        0.4900963604450226,
        0.4470106065273285,
        0.6372453570365906,
    )
    off_scales: tuple[float, float, float] = (
        0.4950235188007355,
        0.4525490701198578,
        0.6430960893630981,
    )
    texture_scales: tuple[float, float, float] = (
        0.226221874,
        0.765139222,
        2.975214005,
    )

    def __post_init__(self) -> None:
        if self.image_size != 64:
            raise ValueError("the ten-class hardware protocol expects a 64x64 image")
        if self.patch_grid != (8, 8):
            raise ValueError("the measured-current model expects an 8x8 patch scan")
        for name in ("graded_thresholds", "on_scales", "off_scales", "texture_scales"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain three finite values")
        if np.any(np.asarray(self.on_scales) <= 0) or np.any(np.asarray(self.off_scales) <= 0):
            raise ValueError("event scales must be positive")
        if np.any(np.asarray(self.texture_scales) <= 0):
            raise ValueError("texture scales must be positive")
        if np.any(np.asarray(self.graded_thresholds) < 0):
            raise ValueError("event thresholds must be non-negative")

    @classmethod
    def from_encoding_file(
        cls,
        path: Path | str,
        *,
        encoding: str = "graded_q000",
        texture_scales: tuple[float, float, float] | None = None,
    ) -> "ProcessingConfig":
        """Load a frozen encoding from ``results/rc10/encoding_parameters.json``."""
        record = json.loads(Path(path).read_text(encoding="utf-8"))[encoding]
        scales = np.asarray(record["scales"], dtype=np.float64)
        if scales.shape != (2, 3):
            raise ValueError("encoding scales must have shape [ON/OFF, RGB]")
        values: dict[str, Any] = {
            "graded_thresholds": tuple(float(v) for v in record["thresholds"]),
            "on_scales": tuple(float(v) for v in scales[0]),
            "off_scales": tuple(float(v) for v in scales[1]),
        }
        if texture_scales is not None:
            values["texture_scales"] = texture_scales
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeafImageProcessor:
    """Turn one RGB leaf image into the exact 9-channel hardware drive."""

    layer_name = "processing"

    def __init__(self, config: ProcessingConfig | None = None):
        self.config = config or ProcessingConfig()

    def _decode(self, image: Image.Image | np.ndarray | bytes | bytearray | Path | str) -> np.ndarray:
        if isinstance(image, (bytes, bytearray)):
            try:
                with Image.open(io.BytesIO(bytes(image))) as decoded:
                    decoded = decoded.convert("RGB").resize(
                        (self.config.image_size, self.config.image_size),
                        Image.Resampling.BILINEAR,
                    )
                    return np.asarray(decoded, dtype=np.float32) / 255.0
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"unable to decode uploaded image: {error}") from error
        if isinstance(image, (Path, str)):
            try:
                with Image.open(image) as decoded:
                    decoded = decoded.convert("RGB").resize(
                        (self.config.image_size, self.config.image_size),
                        Image.Resampling.BILINEAR,
                    )
                    return np.asarray(decoded, dtype=np.float32) / 255.0
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"unable to decode image {image}: {error}") from error
        if isinstance(image, Image.Image):
            decoded = image.convert("RGB").resize(
                (self.config.image_size, self.config.image_size),
                Image.Resampling.BILINEAR,
            )
            return np.asarray(decoded, dtype=np.float32) / 255.0
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("image array must have shape [H,W,3] or [H,W,4]")
        array = array[:, :, :3]
        if array.max(initial=0.0) > 1.0:
            array = array / 255.0
        with Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8)) as decoded:
            decoded = decoded.resize(
                (self.config.image_size, self.config.image_size),
                Image.Resampling.BILINEAR,
            )
            return np.asarray(decoded, dtype=np.float32) / 255.0

    def process(
        self,
        image: Image.Image | np.ndarray | bytes | bytearray | Path | str,
        *,
        sample_id: str,
    ) -> DriveSignals:
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must not be empty")
        rgb = self._decode(image)
        tonic = extract_tonic(rgb, self.config.patch_grid)
        difference = batch_relative_difference(tonic[None, ...])[0]
        frames = make_graded_rc_inputs(
            tonic[None, ...],
            difference[None, ...],
            np.asarray(self.config.graded_thresholds, dtype=np.float32),
            np.asarray(
                [self.config.on_scales, self.config.off_scales], dtype=np.float32
            ),
        )[0]
        texture = extract_texture(rgb, self.config.patch_grid)
        texture = normalize_texture(
            texture[None, ...],
            np.asarray(self.config.texture_scales, dtype=np.float32),
        )[0]
        return DriveSignals(
            frames=frames,
            sample_id=sample_id,
            source="uploaded-image",
            patches_rgb=tonic,
            texture=texture,
            config={
                **self.config.to_dict(),
                "encoding": "graded_q000",
                "scan": "row-alternating-snake",
                "protocol_version": "1.0",
            },
        )
