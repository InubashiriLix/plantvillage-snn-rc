"""Small, explicit contracts between the three runtime layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DRIVE_CHANNELS = (
    "R",
    "G",
    "B",
    "ON_R",
    "ON_G",
    "ON_B",
    "OFF_R",
    "OFF_G",
    "OFF_B",
)
RESPONSE_CHANNELS = (
    "R",
    "G",
    "B",
    "ON-red",
    "ON-green",
    "ON-blue",
    "OFF-red",
    "OFF-green",
    "OFF-blue",
)
DRIVE_STEPS = 64
RESPONSE_ROWS = 12
RESPONSE_POINTS = 65


class LayerUnavailable(RuntimeError):
    """Raised when an injected layer cannot serve the requested mode."""


def _finite_float32(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class DriveSignals:
    """The processing-layer output and the only input accepted by hardware."""

    frames: np.ndarray
    sample_id: str
    source: str
    patches_rgb: np.ndarray | None = None
    texture: np.ndarray | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = _finite_float32(self.frames, (DRIVE_STEPS, len(DRIVE_CHANNELS)), "frames")
        if np.any((frames < 0.0) | (frames > 1.0)):
            raise ValueError("frames must be normalized levels in [0, 1]")
        if frames[0, 3:].any():
            raise ValueError("first frame must not contain ON/OFF events")
        if np.any((frames[:, 3:6] > 0) & (frames[:, 6:9] > 0)):
            raise ValueError("ON and OFF events must be mutually exclusive")
        object.__setattr__(self, "frames", frames)
        if self.patches_rgb is not None:
            object.__setattr__(
                self,
                "patches_rgb",
                _finite_float32(self.patches_rgb, (DRIVE_STEPS, 3), "patches_rgb"),
            )
        if self.texture is not None:
            object.__setattr__(
                self,
                "texture",
                _finite_float32(self.texture, (DRIVE_STEPS, 3), "texture"),
            )
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")

    def to_dict(self, *, include_frames: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "sample_id": self.sample_id,
            "source": self.source,
            "shape": [DRIVE_STEPS, len(DRIVE_CHANNELS)],
            "channels": list(DRIVE_CHANNELS),
            "config": self.config,
        }
        if include_frames:
            value["frames"] = self.frames.round(7).tolist()
        if self.patches_rgb is not None:
            value["patches_rgb"] = self.patches_rgb.round(6).tolist()
        if self.texture is not None:
            value["texture"] = self.texture.round(6).tolist()
        return value


@dataclass(frozen=True)
class HardwareResponse:
    """Current measurements in amperes, shaped [channel, row, response point]."""

    currents: np.ndarray
    sample_id: str
    backend: str
    measured: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        currents = np.asarray(self.currents, dtype=np.float64)
        expected = (len(RESPONSE_CHANNELS), RESPONSE_ROWS, RESPONSE_POINTS)
        if currents.shape != expected:
            raise ValueError(f"currents must have shape {expected}, got {currents.shape}")
        if not np.isfinite(currents).all():
            raise ValueError("currents contains non-finite values")
        object.__setattr__(self, "currents", currents)
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")

    def to_dict(self, *, include_currents: bool = True) -> dict[str, Any]:
        frequency = self.currents.mean(axis=(1, 2)) * 1e6
        value: dict[str, Any] = {
            "sample_id": self.sample_id,
            "backend": self.backend,
            "measured": self.measured,
            "shape": list(self.currents.shape),
            "channels": list(RESPONSE_CHANNELS),
            "mean_current_uA": frequency.round(6).tolist(),
            "metadata": self.metadata,
        }
        if include_currents:
            value["currents_amperes"] = self.currents.round(12).tolist()
        return value


@dataclass(frozen=True)
class DecisionResult:
    """A classifier result that contains no reference to its input modality."""

    predicted_index: int
    predicted_class: str
    probabilities: np.ndarray
    classes: tuple[str, ...]
    model: str = "readout"
    top_k: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if probabilities.ndim != 1 or len(probabilities) != len(self.classes):
            raise ValueError("probabilities must match the class list")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
            raise ValueError("probabilities must sum to one")
        object.__setattr__(self, "probabilities", probabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_index": int(self.predicted_index),
            "predicted_class": self.predicted_class,
            "probability": float(self.probabilities[self.predicted_index]),
            "probabilities": self.probabilities.round(8).tolist(),
            "classes": list(self.classes),
            "top_k": list(self.top_k),
            "model": self.model,
        }


@dataclass(frozen=True)
class PipelineResult:
    drive: DriveSignals
    response: HardwareResponse
    decision: DecisionResult
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self, *, include_currents: bool = False) -> dict[str, Any]:
        return {
            "drive": self.drive.to_dict(),
            "response": self.response.to_dict(include_currents=include_currents),
            "decision": self.decision.to_dict(),
            "timings_ms": self.timings_ms,
        }
