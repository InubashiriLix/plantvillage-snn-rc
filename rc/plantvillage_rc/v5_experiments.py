from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import train_test_split

from .preprocessing import (
    EPSILON,
    batch_relative_difference,
    compute_graded_event_scales,
    compute_thresholds,
    flip_scanned_features,
    make_graded_rc_inputs,
    make_rc_inputs,
    normalize_texture,
)
from .reservoir import (
    generate_mask,
    reservoir_states,
    reservoir_states_to_memmap,
    time_constant_spectrum,
)


SCREEN_CHECKPOINTS = tuple(range(4, 65, 4))
FULL_CHECKPOINTS = tuple(range(1, 65))


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "predicted_class_count": int(len(np.unique(predictions))),
    }


@dataclass
class LinearSoftmaxReadout:
    mean_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray
    intercept_: np.ndarray
    classes_: np.ndarray

    def predict(self, features: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        predictions = np.empty(len(features), dtype=self.classes_.dtype)
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            x = (np.asarray(features[start:stop]) - self.mean_) / self.scale_
            scores = x @ self.coef_.T + self.intercept_
            predictions[start:stop] = self.classes_[np.argmax(scores, axis=1)]
        return predictions


def _feature_statistics(features: np.ndarray, batch_size: int = 2048):
    total = np.zeros(features.shape[1], dtype=np.float64)
    squared = np.zeros_like(total)
    count = 0
    for start in range(0, len(features), batch_size):
        batch = np.asarray(features[start : start + batch_size], dtype=np.float64)
        total += batch.sum(axis=0)
        squared += np.square(batch).sum(axis=0)
        count += len(batch)
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 0.0)
    scale = np.maximum(np.sqrt(variance), 1e-6)
    return mean.astype(np.float32), scale.astype(np.float32)


def _predict_torch(
    layer: torch.nn.Linear,
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = []
    layer.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                np.asarray(features[start : start + batch_size]),
                dtype=torch.float32,
                device=device,
            )
            batch = (batch - torch.as_tensor(mean, device=device)) / torch.as_tensor(
                scale, device=device
            )
            output.append(layer(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(output)


def train_linear_softmax(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    weight_decay: float,
    device: str = "cuda",
    batch_size: int = 512,
    max_epochs: int = 80,
    patience: int = 10,
    validation_indices: np.ndarray | None = None,
    fit_indices: np.ndarray | None = None,
    fixed_epochs: int | None = None,
    seed: int = 42,
) -> tuple[LinearSoftmaxReadout, dict[str, float | int]]:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; run with --device cpu for an intentional CPU fallback")
    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    labels = np.asarray(labels, dtype=np.int64)
    all_indices = np.arange(len(labels))
    fit_indices = all_indices if fit_indices is None else np.asarray(fit_indices)
    validation_indices = (
        None if validation_indices is None else np.asarray(validation_indices)
    )
    mean, scale = _feature_statistics(features)
    layer = torch.nn.Linear(features.shape[1], class_count).to(torch_device)
    optimizer = torch.optim.AdamW(
        layer.parameters(), lr=3e-3, weight_decay=float(weight_decay)
    )
    counts = np.bincount(labels[fit_indices], minlength=class_count)
    class_weights = len(fit_indices) / (
        class_count * np.maximum(counts, 1)
    )
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=torch_device)
    )
    rng = np.random.default_rng(seed)
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    best_metric = -np.inf
    best_epoch = epochs
    best_state = None
    stale = 0
    mean_tensor = torch.as_tensor(mean, device=torch_device)
    scale_tensor = torch.as_tensor(scale, device=torch_device)
    for epoch in range(1, epochs + 1):
        layer.train()
        shuffled = rng.permutation(fit_indices)
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            x = torch.as_tensor(
                np.asarray(features[indices]), dtype=torch.float32, device=torch_device
            )
            y = torch.as_tensor(labels[indices], dtype=torch.long, device=torch_device)
            x = (x - mean_tensor) / scale_tensor
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(layer(x), y)
            loss.backward()
            optimizer.step()
        if validation_indices is not None:
            prediction = _predict_torch(
                layer,
                features[validation_indices],
                mean,
                scale,
                torch_device,
                batch_size,
            )
            metric = recall_score(
                labels[validation_indices], prediction, average="macro", zero_division=0
            )
            if metric > best_metric + 1e-5:
                best_metric = float(metric)
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in layer.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        layer.load_state_dict(best_state)
    model = LinearSoftmaxReadout(
        mean_=mean,
        scale_=scale,
        coef_=layer.weight.detach().cpu().numpy().copy(),
        intercept_=layer.bias.detach().cpu().numpy().copy(),
        classes_=np.arange(class_count),
    )
    del layer, optimizer, mean_tensor, scale_tensor
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return model, {
        "best_epoch": int(best_epoch),
        "internal_macro_recall": float(best_metric) if best_state is not None else -1.0,
    }


def _load_bases(input_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for split in ("train", "val", "test"):
        with np.load(input_dir / f"{split}_base.npz") as archive:
            output[split] = {key: archive[key] for key in archive.files}
    return output


def _encoding_parameters(
    train_diff: np.ndarray, encodings: Iterable[str]
) -> dict[str, dict[str, object]]:
    parameters = {}
    for encoding in encodings:
        if encoding.startswith("binary_"):
            quantile = int(encoding.rsplit("q", 1)[1]) / 100.0
            threshold = compute_thresholds(train_diff, quantile)
            parameters[encoding] = {"kind": "binary", "thresholds": threshold}
        elif encoding.startswith("graded_"):
            quantile = int(encoding.rsplit("q", 1)[1]) / 100.0
            threshold = (
                np.zeros(3, dtype=np.float32)
                if quantile == 0.0
                else compute_thresholds(train_diff, quantile)
            )
            scales = compute_graded_event_scales(train_diff, threshold)
            parameters[encoding] = {
                "kind": "graded",
                "thresholds": threshold,
                "scales": scales,
            }
        else:
            raise ValueError(f"Unknown event encoding {encoding}")
    return parameters


def _make_inputs(
    base: dict[str, np.ndarray],
    parameters: dict[str, object],
    texture_scales: np.ndarray,
    *,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    tonic_parts = [base["tonic"]]
    texture = normalize_texture(base["texture"], texture_scales)
    texture_parts = [texture]
    if augment:
        for direction in ("horizontal", "vertical"):
            tonic_parts.append(flip_scanned_features(base["tonic"], (8, 8), direction))
            texture_parts.append(flip_scanned_features(texture, (8, 8), direction))
    inputs = []
    for tonic, texture_part in zip(tonic_parts, texture_parts):
        difference = batch_relative_difference(tonic)
        if parameters["kind"] == "graded":
            rgb = make_graded_rc_inputs(
                tonic, difference, parameters["thresholds"], parameters["scales"]
            )
        else:
            rgb = make_rc_inputs(tonic, difference, parameters["thresholds"])
        inputs.append(np.concatenate([rgb, texture_part], axis=2).astype(np.float32))
    labels = np.tile(base["labels"], len(inputs))
    return np.concatenate(inputs), labels


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_v5_experiments(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    encodings: Iterable[str] = (
        "binary_q080",
        "graded_q000",
        "graded_q050",
        "graded_q060",
        "graded_q070",
    ),
    tau_max_values: Iterable[float] = (64.0, 128.0, 256.0),
    gamma_values: Iterable[float] = (1.5, 2.0, 3.0),
    weight_decays: Iterable[float] = (1e-5, 1e-4, 1e-3),
    top_k: int = 3,
    max_epochs: int = 80,
    patience: int = 10,
    target_macro_recall: float = 0.80,
    device: str = "cuda",
    mask_seed: int = 42,
    enable_augmentation: bool = True,
) -> dict[str, object]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = output_dir / "states"
    states_dir.mkdir(exist_ok=True)
    config = json.loads((input_dir / "preprocess_config.json").read_text())
    if config.get("patch_grid") != [8, 8] or config.get("time_steps") != 64:
        raise ValueError("v5 requires 8x8 v4 preprocessing")
    classes = config["classes"]
    class_count = len(classes)
    bases = _load_bases(input_dir)
    texture_scales = np.asarray(config["texture_scales"], dtype=np.float32)
    encodings = tuple(encodings)
    # Always persist the two hardware interchange encodings, even for --quick.
    # This does not expand the model-selection search below.
    export_encodings = tuple(dict.fromkeys((*encodings, "graded_q000", "binary_q080")))
    parameters = _encoding_parameters(bases["train"]["diff"], export_encodings)
    mask = generate_mask(mask_seed, channels=12)
    np.save(output_dir / "mask_12x12.npy", mask)
    serializable_parameters = {
        key: {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in values.items()
        }
        for key, values in parameters.items()
    }
    (output_dir / "encoding_parameters.json").write_text(
        json.dumps(serializable_parameters, indent=2)
    )

    screen_rows = []
    for encoding in encodings:
        train_inputs, y_train = _make_inputs(
            bases["train"], parameters[encoding], texture_scales
        )
        val_inputs, y_val = _make_inputs(
            bases["val"], parameters[encoding], texture_scales
        )
        for tau_max in tau_max_values:
            alpha = time_constant_spectrum(1.0, float(tau_max), channels=12)
            for gamma in gamma_values:
                x_train = reservoir_states(
                    train_inputs,
                    mask,
                    alpha=alpha,
                    gamma=float(gamma),
                    checkpoint_steps=SCREEN_CHECKPOINTS,
                )
                x_val = reservoir_states(
                    val_inputs,
                    mask,
                    alpha=alpha,
                    gamma=float(gamma),
                    checkpoint_steps=SCREEN_CHECKPOINTS,
                )
                from .v4_experiments import TorchRidgeClassifier

                readout = TorchRidgeClassifier(1.0, device=device).fit(x_train, y_train)
                row = {
                    "encoding": encoding,
                    "tau_max": float(tau_max),
                    "gamma": float(gamma),
                    "read_count": len(SCREEN_CHECKPOINTS),
                    **{
                        f"val_{key}": value
                        for key, value in _metrics(y_val, readout.predict(x_val)).items()
                    },
                }
                screen_rows.append(row)
                del x_train, x_val, readout
        del train_inputs, val_inputs
    screen_rows.sort(
        key=lambda row: (row["val_macro_recall"], row["val_accuracy"]), reverse=True
    )
    _write_csv(output_dir / "screening_trials.csv", screen_rows)
    finalists = screen_rows[:top_k]

    y_train = bases["train"]["labels"]
    y_val = bases["val"]["labels"]
    fit_indices, early_indices = train_test_split(
        np.arange(len(y_train)),
        test_size=0.1,
        random_state=42,
        stratify=y_train,
    )
    dense_rows = []
    dense_models: dict[int, LinearSoftmaxReadout] = {}
    dense_paths: dict[int, tuple[Path, Path]] = {}
    for finalist_index, finalist in enumerate(finalists):
        encoding = str(finalist["encoding"])
        train_inputs, _ = _make_inputs(
            bases["train"], parameters[encoding], texture_scales
        )
        val_inputs, _ = _make_inputs(bases["val"], parameters[encoding], texture_scales)
        alpha = time_constant_spectrum(
            1.0, float(finalist["tau_max"]), channels=12
        )
        train_path = states_dir / f"candidate_{finalist_index}_train.npy"
        val_path = states_dir / f"candidate_{finalist_index}_val.npy"
        x_train = reservoir_states_to_memmap(
            train_inputs,
            train_path,
            mask,
            checkpoint_steps=FULL_CHECKPOINTS,
            alpha=alpha,
            gamma=float(finalist["gamma"]),
        )
        x_val = reservoir_states_to_memmap(
            val_inputs,
            val_path,
            mask,
            checkpoint_steps=FULL_CHECKPOINTS,
            alpha=alpha,
            gamma=float(finalist["gamma"]),
        )
        dense_paths[finalist_index] = (train_path, val_path)
        internal_trials = []
        for weight_decay in weight_decays:
            _, history = train_linear_softmax(
                x_train,
                y_train,
                class_count=class_count,
                weight_decay=float(weight_decay),
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                fit_indices=fit_indices,
                validation_indices=early_indices,
            )
            internal_trials.append((history["internal_macro_recall"], weight_decay, history))
        _, best_decay, history = max(internal_trials, key=lambda item: item[0])
        model, _ = train_linear_softmax(
            x_train,
            y_train,
            class_count=class_count,
            weight_decay=float(best_decay),
            device=device,
            fixed_epochs=int(history["best_epoch"]),
        )
        metrics = _metrics(y_val, model.predict(x_val))
        dense_rows.append(
            {
                "candidate": finalist_index,
                "encoding": encoding,
                "tau_max": finalist["tau_max"],
                "gamma": finalist["gamma"],
                "weight_decay": float(best_decay),
                "epochs": history["best_epoch"],
                "internal_macro_recall": history["internal_macro_recall"],
                **{f"val_{key}": value for key, value in metrics.items()},
            }
        )
        dense_models[finalist_index] = model
        del train_inputs, val_inputs, x_train, x_val
    dense_rows.sort(
        key=lambda row: (row["val_macro_recall"], row["val_accuracy"]), reverse=True
    )
    _write_csv(output_dir / "dense_trials.csv", dense_rows)
    primary = dense_rows[0]
    best_index = int(primary["candidate"])
    primary_model = dense_models[best_index]
    augmentation_result = None

    if enable_augmentation:
        encoding = str(primary["encoding"])
        augmented_inputs, augmented_labels = _make_inputs(
            bases["train"], parameters[encoding], texture_scales, augment=True
        )
        alpha = time_constant_spectrum(1.0, float(primary["tau_max"]), channels=12)
        augmented_path = states_dir / "augmented_train.npy"
        augmented_states = reservoir_states_to_memmap(
            augmented_inputs,
            augmented_path,
            mask,
            checkpoint_steps=FULL_CHECKPOINTS,
            alpha=alpha,
            gamma=float(primary["gamma"]),
        )
        augmented_model, _ = train_linear_softmax(
            augmented_states,
            augmented_labels,
            class_count=class_count,
            weight_decay=float(primary["weight_decay"]),
            device=device,
            fixed_epochs=int(primary["epochs"]),
        )
        validation_states = np.load(dense_paths[best_index][1], mmap_mode="r")
        augmented_metrics = _metrics(y_val, augmented_model.predict(validation_states))
        augmentation_result = {
            "enabled": True,
            **{f"val_{key}": value for key, value in augmented_metrics.items()},
        }
        if augmented_metrics["macro_recall"] > primary["val_macro_recall"]:
            primary_model = augmented_model
            primary = dict(primary)
            primary.update(
                {f"val_{key}": value for key, value in augmented_metrics.items()}
            )
            primary["training_augmentation"] = "horizontal_and_vertical_flips"
        else:
            primary["training_augmentation"] = "none"
    else:
        primary["training_augmentation"] = "none"

    validation_states = np.load(dense_paths[best_index][1], mmap_mode="r")
    val_predictions = primary_model.predict(validation_states)
    (output_dir / "validation_classification_report.json").write_text(
        json.dumps(
            classification_report(
                y_val,
                val_predictions,
                labels=np.arange(class_count),
                target_names=classes,
                output_dict=True,
                zero_division=0,
            ),
            indent=2,
        )
    )
    reached_target = float(primary["val_macro_recall"]) >= target_macro_recall
    result: dict[str, object] = {
        "primary": primary,
        "augmentation_trial": augmentation_result,
        "target_macro_recall": target_macro_recall,
        "reached_target": reached_target,
        "test_evaluated": False,
        "training_device": device,
        "latency": {
            "optical_frames": 64,
            "electrical_reads": 64,
            "feature_dimension": 9216,
        },
    }
    if class_count == 38:
        result["baseline_v4_validation_macro_recall"] = 0.7216786592450916
        result["macro_recall_gain_over_v4"] = float(primary["val_macro_recall"]) - 0.7216786592450916
    else:
        result["subset_note"] = (
            "Metrics are for the configured class subset and are not directly "
            "comparable with the 38-class v4 baseline."
        )
    if reached_target:
        test_inputs, y_test = _make_inputs(
            bases["test"], parameters[str(primary["encoding"])], texture_scales
        )
        alpha = time_constant_spectrum(1.0, float(primary["tau_max"]), channels=12)
        test_states = reservoir_states_to_memmap(
            test_inputs,
            states_dir / "test.npy",
            mask,
            checkpoint_steps=FULL_CHECKPOINTS,
            alpha=alpha,
            gamma=float(primary["gamma"]),
        )
        result["test"] = _metrics(y_test, primary_model.predict(test_states))
        result["test_evaluated"] = True
    joblib.dump(primary_model, output_dir / "readout.joblib")
    (output_dir / "v5_results.json").write_text(json.dumps(result, indent=2))
    return result
