from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, recall_score
import torch

from .reservoir import generate_mask, reservoir_states, time_constant_spectrum


CHECKPOINTS = (8, 16, 24, 32, 40, 48, 56, 64)


class TorchRidgeClassifier:
    """Balanced multiclass ridge readout trained with CUDA normal equations.

    Learned parameters are copied back to NumPy, so prediction and joblib export do
    not require a GPU.  The target coding (-1/+1) matches RidgeClassifier.
    """

    def __init__(self, alpha: float, device: str = "cuda") -> None:
        self.alpha = float(alpha)
        self.device = device

    def fit(self, features: np.ndarray, labels: np.ndarray):
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; run with --device cpu "
                "only when a CPU fallback is intentional"
            )
        x = torch.as_tensor(
            np.asarray(features), dtype=torch.float32, device=device
        )
        y = torch.as_tensor(labels, dtype=torch.long, device=device)
        self.classes_ = np.unique(labels)
        class_count = len(self.classes_)
        if not np.array_equal(self.classes_, np.arange(class_count)):
            raise ValueError("v4 labels must be contiguous integers starting at zero")

        mean = x.mean(dim=0)
        scale = x.std(dim=0, correction=0).clamp_min_(1e-6)
        x = (x - mean) / scale
        counts = torch.bincount(y, minlength=class_count).float()
        sample_weight = y.numel() / (class_count * counts[y])
        targets = torch.full(
            (y.numel(), class_count), -1.0, dtype=torch.float32, device=device
        )
        targets.scatter_(1, y[:, None], 1.0)

        # Add an unregularized bias column and form the weighted normal equations.
        design = torch.cat(
            [x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=device)], dim=1
        )
        root_weight = sample_weight.sqrt_()[:, None]
        weighted_design = design * root_weight
        gram = weighted_design.T @ weighted_design
        rhs = weighted_design.T @ (targets * root_weight)
        regularizer = torch.eye(gram.shape[0], dtype=gram.dtype, device=device)
        regularizer[-1, -1] = 0.0
        solution = torch.linalg.solve(gram + self.alpha * regularizer, rhs)

        self.mean_ = mean.cpu().numpy()
        self.scale_ = scale.cpu().numpy()
        self.coef_ = solution[:-1].T.cpu().numpy()
        self.intercept_ = solution[-1].cpu().numpy()
        del x, y, design, weighted_design, gram, rhs, targets, solution
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features) - self.mean_) / self.scale_
        scores = x @ self.coef_.T + self.intercept_
        return self.classes_[np.argmax(scores, axis=1)]


def _model(ridge_alpha: float, device: str):
    return TorchRidgeClassifier(ridge_alpha, device=device)


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "predicted_class_count": int(len(np.unique(predictions))),
    }


def _input_path(input_dir: Path, split: str, channels: int, tag: str) -> Path:
    middle = "_inputs" if channels == 9 else "_inputs12"
    return input_dir / f"{split}{middle}_{tag}.npy"


def _states(
    inputs: np.ndarray,
    mask: np.ndarray,
    alpha: np.ndarray,
    gamma: float,
    checkpoints: bool,
) -> np.ndarray:
    return reservoir_states(
        inputs,
        mask,
        alpha=alpha,
        gamma=gamma,
        checkpoint_steps=CHECKPOINTS if checkpoints else None,
    )


def run_v4_experiments(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    quantile_tags: Iterable[str] = ("q060", "q070", "q080"),
    tau_max_values: Iterable[float] = (64.0, 128.0),
    gamma_values: Iterable[float] = (0.5, 1.0, 2.0),
    ridge_alphas: Iterable[float] = (1e-4, 1e-2, 1.0),
    target_macro_recall: float = 0.80,
    mask_seed: int = 42,
    device: str = "cuda",
) -> dict[str, object]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((input_dir / "preprocess_config.json").read_text())
    if config.get("patch_grid") != [8, 8] or config.get("time_steps") != 64:
        raise ValueError("v4 requires preprocessing with --patch-grid 8 8")
    classes = config["classes"]
    y_train = np.load(input_dir / "train_labels.npy")
    y_val = np.load(input_dir / "val_labels.npy")
    full_mask = generate_mask(mask_seed, channels=12)
    np.save(output_dir / "mask_12x12.npy", full_mask)

    modes = (
        ("rgb_event_final", 9, False),
        ("rgb_event_checkpoints", 9, True),
        ("all_columns_final", 12, False),
        ("all_columns_checkpoints", 12, True),
    )
    trials: list[dict[str, object]] = []
    best: dict[str, dict[str, object]] = {}
    for channels in (9, 12):
        mask = full_mask[:, :channels]
        for tag in quantile_tags:
            train_inputs = np.load(
                _input_path(input_dir, "train", channels, tag), mmap_mode="r"
            )
            val_inputs = np.load(
                _input_path(input_dir, "val", channels, tag), mmap_mode="r"
            )
            for tau_max in tau_max_values:
                alpha = time_constant_spectrum(
                    1.0, float(tau_max), channels=channels
                )
                for gamma in gamma_values:
                    for mode, mode_channels, checkpoints in modes:
                        if mode_channels != channels:
                            continue
                        x_train = _states(
                            train_inputs, mask, alpha, float(gamma), checkpoints
                        )
                        x_val = _states(
                            val_inputs, mask, alpha, float(gamma), checkpoints
                        )
                        for ridge_alpha in ridge_alphas:
                            classifier = _model(float(ridge_alpha), device)
                            classifier.fit(x_train, y_train)
                            predictions = classifier.predict(x_val)
                            row: dict[str, object] = {
                                "mode": mode,
                                "channels": channels,
                                "quantile_tag": tag,
                                "tau_max": float(tau_max),
                                "gamma": float(gamma),
                                "ridge_alpha": float(ridge_alpha),
                                "optical_frames": 64,
                                "read_count": 8 if checkpoints else 1,
                                "feature_dimension": int(x_train.shape[1]),
                                **{
                                    f"val_{key}": value
                                    for key, value in _metrics(y_val, predictions).items()
                                },
                            }
                            trials.append(row)
                            key = (
                                float(row["val_macro_recall"]),
                                float(row["val_accuracy"]),
                            )
                            previous = best.get(mode)
                            if previous is None or key > (
                                float(previous["val_macro_recall"]),
                                float(previous["val_accuracy"]),
                            ):
                                best[mode] = row

    with (output_dir / "validation_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)

    primary = best["all_columns_checkpoints"]
    primary_channels = int(primary["channels"])
    primary_mask = full_mask[:, :primary_channels]
    primary_alpha = time_constant_spectrum(
        1.0, float(primary["tau_max"]), channels=primary_channels
    )
    primary_inputs = {
        split: np.load(
            _input_path(
                input_dir, split, primary_channels, str(primary["quantile_tag"])
            ),
            mmap_mode="r",
        )
        for split in ("train", "val")
    }
    primary_states = {
        split: _states(
            inputs,
            primary_mask,
            primary_alpha,
            float(primary["gamma"]),
            True,
        )
        for split, inputs in primary_inputs.items()
    }
    states_dir = output_dir / "states"
    states_dir.mkdir(exist_ok=True)
    for split, values in primary_states.items():
        np.save(states_dir / f"{split}_checkpoint_states.npy", values)
    np.save(output_dir / "alpha_nodes.npy", primary_alpha)

    primary_model = _model(float(primary["ridge_alpha"]), device)
    primary_model.fit(primary_states["train"], y_train)
    val_predictions = primary_model.predict(primary_states["val"])
    with (output_dir / "validation_classification_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            classification_report(
                y_val,
                val_predictions,
                labels=np.arange(len(classes)),
                target_names=classes,
                output_dict=True,
                zero_division=0,
            ),
            handle,
            indent=2,
        )

    reached_target = float(primary["val_macro_recall"]) >= target_macro_recall
    result: dict[str, object] = {
        "baseline_v3_validation_macro_recall": 0.6209266415318196,
        "best_by_mode": best,
        "primary": primary,
        "macro_recall_gain_over_v3": float(primary["val_macro_recall"])
        - 0.6209266415318196,
        "target_macro_recall": target_macro_recall,
        "reached_target": reached_target,
        "test_evaluated": False,
        "training_device": device,
        "latency": {
            "direct_pixel_frames": 4096,
            "v4_optical_frames": 64,
            "optical_frame_speedup": 64.0,
            "checkpoint_reads": 8,
        },
    }
    if reached_target:
        y_test = np.load(input_dir / "test_labels.npy")
        test_inputs = np.load(
            _input_path(
                input_dir, "test", primary_channels, str(primary["quantile_tag"])
            ),
            mmap_mode="r",
        )
        test_states = _states(
            test_inputs,
            primary_mask,
            primary_alpha,
            float(primary["gamma"]),
            True,
        )
        final_model = _model(float(primary["ridge_alpha"]), device)
        final_model.fit(
            np.concatenate([primary_states["train"], primary_states["val"]]),
            np.concatenate([y_train, y_val]),
        )
        result["test"] = _metrics(y_test, final_model.predict(test_states))
        result["test_evaluated"] = True
        np.save(states_dir / "test_checkpoint_states.npy", test_states)
        joblib.dump(final_model, output_dir / "readout.joblib")
    else:
        joblib.dump(primary_model, output_dir / "validation_readout.joblib")

    with (output_dir / "v4_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
