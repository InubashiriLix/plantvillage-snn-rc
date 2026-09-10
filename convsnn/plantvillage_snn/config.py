from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import json


@dataclass
class ExperimentConfig:
    data_root: str = "data"
    output_dir: str = "artifacts"
    model: str = "conv_snn"
    stage: str = "all38"
    phase: str = "ideal"
    seed: int = 42
    split_seed: int = 42
    selection_fraction: float = 0.15
    split_manifest_path: str | None = None
    image_size: int = 64
    batch_size: int = 64
    num_workers: int = 0
    time_steps: int = 8
    beta: float = 0.9
    threshold: float = 1.0
    channels: list[int] | None = None
    layer_betas: list[float] | None = None
    layer_thresholds: list[float] | None = None
    input_encoding: str = "direct_current"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    patience: int = 5
    augmentation: str = "basic"
    class_balance: str = "class_weights"
    top10_path: str = "artifacts/top10_classes.json"
    checkpoint: str | None = None
    teacher_checkpoint: str | None = None
    membrane_loss_weight: float = 0.25
    distill_weight: float = 0.5
    distill_temperature: float = 2.0
    firing_regularization_weight: float = 1e-3
    firing_rate_min: float = 0.01
    firing_rate_max: float = 0.35
    evaluate_final: bool = False
    device_csv: str = str(Path(__file__).resolve().parents[1] / "cnnRef/source/data.csv")
    device_v_bias: float = -5.0
    device_g_bins: int = 200
    device_max_pulses: int = 200
    max_pulses_per_update: int = 8
    hw_scale_margin: float = 1.5
    accum_decay: float = 0.9
    accum_decay_interval: int = 10
    signed_mapping: str = "differential_pair"
    device_preprocess: str = "raw"
    max_train_samples: int | None = None
    max_selection_samples: int | None = None
    max_final_samples: int | None = None
    device: str = "auto"
    recording_enabled: bool = True
    record_internal_summaries: bool = True
    record_sample_predictions: bool = True
    checkpoint_interval: int = 10
    dataset_hash_mode: str = "sha256"
    recording_dir: str | None = None

    def validate(self) -> None:
        if self.model not in {"conv_snn", "compact_snn"}:
            raise ValueError("model must be conv_snn or compact_snn")
        if self.stage not in {"all38", "top10"}:
            raise ValueError("stage must be all38 or top10")
        if self.phase not in {"ideal", "hw_finetune"}:
            raise ValueError("phase must be ideal or hw_finetune")
        if self.phase == "hw_finetune" and not self.checkpoint:
            raise ValueError("hw_finetune requires an ideal --checkpoint")
        if self.phase == "ideal" and self.checkpoint:
            raise ValueError("ideal training always builds a fresh model; omit --checkpoint")
        if self.stage == "top10" and not Path(self.top10_path).is_file():
            raise FileNotFoundError(
                f"top10 stage is defined solely by missing file: {self.top10_path}"
            )
        if not 0 < self.selection_fraction < 1:
            raise ValueError("selection_fraction must be between 0 and 1")
        if self.time_steps < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("time_steps, epochs, and batch_size must be positive")
        if self.input_encoding != "direct_current":
            raise ValueError("only direct_current input encoding is currently calibrated")
        if self.channels is not None:
            if self.model != "compact_snn" or len(self.channels) != 3 or any(c < 1 for c in self.channels):
                raise ValueError("compact_snn channels must contain three positive widths")
        expected_layers = 4 if self.model == "compact_snn" else 5
        for name, values in (
            ("layer_betas", self.layer_betas),
            ("layer_thresholds", self.layer_thresholds),
        ):
            if values is not None and len(values) != expected_layers:
                raise ValueError(f"{name} must contain {expected_layers} values for {self.model}")
        if self.signed_mapping != "differential_pair":
            raise ValueError("hardware-aware training requires differential_pair signed mapping")
        if self.device_preprocess not in {"raw", "robust"}:
            raise ValueError("device_preprocess must be raw or robust")
        if self.distill_weight < 0 or self.membrane_loss_weight < 0:
            raise ValueError("loss weights cannot be negative")
        if self.distill_temperature <= 0:
            raise ValueError("distill_temperature must be positive")
        if not 0 <= self.firing_rate_min <= self.firing_rate_max <= 1:
            raise ValueError("firing-rate bounds must satisfy 0 <= min <= max <= 1")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if self.dataset_hash_mode not in {"sha256", "none"}:
            raise ValueError("dataset_hash_mode must be sha256 or none")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return cls(**raw)

    def with_overrides(self, overrides: dict[str, Any]) -> "ExperimentConfig":
        values = self.to_dict()
        # Callers already omit unspecified CLI values. Explicit None is
        # meaningful here: formal runs use it to clear search sample limits.
        values.update(overrides)
        return ExperimentConfig(**values)
