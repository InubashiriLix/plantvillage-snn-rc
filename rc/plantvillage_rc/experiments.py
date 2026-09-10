from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plantvillage-rc-matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .reservoir import (
    DeviceProfile,
    Nonlinearity,
    Variant,
    generate_mask,
    reservoir_states,
    row_spectrum,
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    variant: Variant = "full"
    quantile_tag: str = "q070"
    nonlinearity: Nonlinearity = "saturating"
    device_profile: DeviceProfile = "row-spectrum"
    class_weight: str | None = "balanced"
    uniform_alphas: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)
    alpha_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.9),
        (0.05, 0.98),
        (0.2, 0.98),
    )
    gammas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)


DEFAULT_EXPERIMENTS = (
    ExperimentSpec(
        "full_uniform_unbalanced",
        quantile_tag="q060",
        device_profile="uniform",
        class_weight=None,
    ),
    ExperimentSpec(
        "full_uniform_balanced",
        quantile_tag="q060",
        device_profile="uniform",
    ),
    ExperimentSpec(
        "full_heterogeneous_unbalanced",
        quantile_tag="q060",
        class_weight=None,
    ),
    ExperimentSpec("full_heterogeneous_q060", quantile_tag="q060"),
    ExperimentSpec("full_heterogeneous_q070", quantile_tag="q070"),
    ExperimentSpec("full_heterogeneous_q080", quantile_tag="q080"),
)


def _load_inputs(input_dir: Path, split: str, tag: str) -> np.ndarray:
    return np.load(input_dir / f"{split}_inputs_{tag}.npy", mmap_mode="r")


def _load_labels(input_dir: Path, split: str) -> np.ndarray:
    return np.load(input_dir / f"{split}_labels.npy")


def _score(y_true: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    return (
        float(accuracy_score(y_true, predictions)),
        float(recall_score(y_true, predictions, average="macro", zero_division=0)),
    )


def _model(ridge_alpha: float, class_weight: str | None):
    return make_pipeline(
        StandardScaler(),
        RidgeClassifier(
            alpha=ridge_alpha,
            class_weight=class_weight,
            solver="svd",
        ),
    )


def _dynamics_candidates(spec: ExperimentSpec):
    if spec.device_profile == "uniform":
        for alpha in spec.uniform_alphas:
            for gamma in spec.gammas:
                yield {
                    "device_profile": "uniform",
                    "reservoir_alpha": float(alpha),
                    "alpha_min": None,
                    "alpha_max": None,
                    "gamma": float(gamma),
                }
        return
    for alpha_min, alpha_max in spec.alpha_ranges:
        for gamma in spec.gammas:
            yield {
                "device_profile": "row-spectrum",
                "reservoir_alpha": None,
                "alpha_min": float(alpha_min),
                "alpha_max": float(alpha_max),
                "gamma": float(gamma),
            }


def _alpha_parameter(trial: dict[str, object]) -> float | np.ndarray:
    if trial["device_profile"] == "uniform":
        return float(trial["reservoir_alpha"])
    return row_spectrum(float(trial["alpha_min"]), float(trial["alpha_max"]))


def _selection_key(trial: dict[str, object]) -> tuple[float, float, int]:
    return (
        float(trial["val_macro_recall"]),
        float(trial["val_accuracy"]),
        int(trial["val_predicted_class_count"]),
    )


def _plot_confusion(
    matrix: np.ndarray,
    class_names: list[str],
    output: Path,
    *,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    axis.tick_params(axis="y", labelsize=6)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run_experiment(
    spec: ExperimentSpec,
    input_dir: Path,
    output_dir: Path,
    *,
    ridge_alphas: Iterable[float],
    mask: np.ndarray,
) -> dict[str, object]:
    train_inputs = _load_inputs(input_dir, "train", spec.quantile_tag)
    val_inputs = _load_inputs(input_dir, "val", spec.quantile_tag)
    test_inputs = _load_inputs(input_dir, "test", spec.quantile_tag)
    y_train = _load_labels(input_dir, "train")
    y_val = _load_labels(input_dir, "val")
    y_test = _load_labels(input_dir, "test")
    with (input_dir / "preprocess_config.json").open(encoding="utf-8") as handle:
        preprocess_config = json.load(handle)
    class_names = preprocess_config["classes"]
    test_base = np.load(input_dir / "test_base.npz")
    test_sample_ids = test_base["sample_ids"]

    trials: list[dict[str, object]] = []
    for dynamics in _dynamics_candidates(spec):
        alpha_parameter = _alpha_parameter(dynamics)
        x_train = reservoir_states(
            train_inputs,
            mask,
            alpha=alpha_parameter,
            gamma=float(dynamics["gamma"]),
            variant=spec.variant,
            nonlinearity=spec.nonlinearity,
        )
        x_val = reservoir_states(
            val_inputs,
            mask,
            alpha=alpha_parameter,
            gamma=float(dynamics["gamma"]),
            variant=spec.variant,
            nonlinearity=spec.nonlinearity,
        )
        for ridge_alpha in ridge_alphas:
            classifier = _model(float(ridge_alpha), spec.class_weight)
            classifier.fit(x_train, y_train)
            val_predictions = classifier.predict(x_val)
            val_accuracy, val_macro_recall = _score(y_val, val_predictions)
            trials.append(
                {
                    **dynamics,
                    "ridge_alpha": float(ridge_alpha),
                    "class_weight": spec.class_weight,
                    "val_accuracy": val_accuracy,
                    "val_macro_recall": val_macro_recall,
                    "val_predicted_class_count": int(len(np.unique(val_predictions))),
                }
            )
    if not trials:
        raise ValueError(f"Experiment {spec.name} has no dynamics candidates")
    selected = max(trials, key=_selection_key)
    selected_alpha = _alpha_parameter(selected)

    x_train = reservoir_states(
        train_inputs,
        mask,
        alpha=selected_alpha,
        gamma=float(selected["gamma"]),
        variant=spec.variant,
        nonlinearity=spec.nonlinearity,
    )
    x_val = reservoir_states(
        val_inputs,
        mask,
        alpha=selected_alpha,
        gamma=float(selected["gamma"]),
        variant=spec.variant,
        nonlinearity=spec.nonlinearity,
    )
    x_test = reservoir_states(
        test_inputs,
        mask,
        alpha=selected_alpha,
        gamma=float(selected["gamma"]),
        variant=spec.variant,
        nonlinearity=spec.nonlinearity,
    )
    classifier = _model(float(selected["ridge_alpha"]), spec.class_weight)
    classifier.fit(np.concatenate([x_train, x_val]), np.concatenate([y_train, y_val]))
    predictions = classifier.predict(x_test)
    test_accuracy, test_macro_recall = _score(y_test, predictions)

    experiment_dir = output_dir / spec.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_test, predictions, labels=np.arange(len(class_names)))
    np.save(experiment_dir / "confusion_matrix.npy", matrix)
    _plot_confusion(matrix, class_names, experiment_dir / "confusion_matrix.png", title="Counts")
    balanced_matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        normalize="true",
    )
    np.save(experiment_dir / "confusion_matrix_balanced.npy", balanced_matrix)
    _plot_confusion(
        balanced_matrix,
        class_names,
        experiment_dir / "confusion_matrix_balanced.png",
        title="Row-normalized recall",
    )
    with (experiment_dir / "classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            classification_report(
                y_test,
                predictions,
                labels=np.arange(len(class_names)),
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            ),
            handle,
            indent=2,
        )
    with (experiment_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "true_index", "true_class", "predicted_index", "predicted_class"))
        for sample_id, truth, prediction in zip(test_sample_ids, y_test, predictions):
            writer.writerow(
                (
                    str(sample_id),
                    int(truth),
                    class_names[int(truth)],
                    int(prediction),
                    class_names[int(prediction)],
                )
            )
    joblib.dump(classifier, experiment_dir / "readout.joblib")
    alpha_nodes = np.asarray(selected_alpha, dtype=np.float32)
    if alpha_nodes.ndim == 0:
        alpha_nodes = np.full((12, 9), float(alpha_nodes), dtype=np.float32)
    np.save(experiment_dir / "alpha_nodes.npy", alpha_nodes)
    np.save(
        experiment_dir / "gamma_nodes.npy",
        np.full((12, 9), float(selected["gamma"]), dtype=np.float32),
    )
    with (experiment_dir / "selected_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
    with (experiment_dir / "validation_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)

    return {
        **asdict(spec),
        "selected": selected,
        "val_accuracy": selected["val_accuracy"],
        "val_macro_recall": selected["val_macro_recall"],
        "val_predicted_class_count": selected["val_predicted_class_count"],
        "test_accuracy": test_accuracy,
        "test_macro_recall": test_macro_recall,
        "test_predicted_class_count": int(len(np.unique(predictions))),
        "feature_dimension": int(x_test.shape[1]),
    }


def _plot_accuracy_event_rate(
    results: list[dict[str, object]], preprocess_config: dict[str, object], output: Path
) -> None:
    points = []
    stats = preprocess_config["event_statistics_train"]
    for result in results:
        tag = result["quantile_tag"]
        if result["name"] == f"full_heterogeneous_{tag}":
            points.append((stats[tag]["event_rate"], result["val_accuracy"], tag))
    if not points:
        return
    points.sort()
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([p[0] for p in points], [p[1] for p in points], marker="o")
    for event_rate, accuracy, tag in points:
        axis.annotate(tag, (event_rate, accuracy))
    axis.set_xlabel("Training event rate")
    axis.set_ylabel("Validation accuracy")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run_all_experiments(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    specs: Iterable[ExperimentSpec] = DEFAULT_EXPERIMENTS,
    ridge_alphas: Iterable[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    mask_seed: int = 42,
) -> list[dict[str, object]]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = generate_mask(mask_seed)
    np.save(output_dir / "mask.npy", mask)
    results = [
        run_experiment(
            spec,
            input_dir,
            output_dir,
            ridge_alphas=ridge_alphas,
            mask=mask,
        )
        for spec in specs
    ]
    with (input_dir / "preprocess_config.json").open(encoding="utf-8") as handle:
        preprocess_config = json.load(handle)
    with (output_dir / "experiment_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    primary_results = [
        result
        for result in results
        if str(result["name"]).startswith("full_heterogeneous_q")
    ]
    primary = max(
        primary_results,
        key=lambda result: (
            float(result["val_macro_recall"]),
            float(result["val_accuracy"]),
        ),
    ) if primary_results else None
    acceptance = {
        "primary_experiment": None if primary is None else primary["name"],
        "validation_accuracy_target": 0.45,
        "validation_macro_recall_target": 0.45,
        "validation_predicted_class_target": 35,
        "passed": bool(
            primary is not None
            and float(primary["val_accuracy"]) >= 0.45
            and float(primary["val_macro_recall"]) >= 0.45
            and int(primary["val_predicted_class_count"]) >= 35
        ),
    }
    with (output_dir / "primary_selection.json").open("w", encoding="utf-8") as handle:
        json.dump({"acceptance": acceptance, "result": primary}, handle, indent=2)
    with (output_dir / "experiment_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "name",
            "variant",
            "quantile_tag",
            "nonlinearity",
            "feature_dimension",
            "val_accuracy",
            "val_macro_recall",
            "val_predicted_class_count",
            "test_accuracy",
            "test_macro_recall",
            "test_predicted_class_count",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    _plot_accuracy_event_rate(
        results, preprocess_config, output_dir / "accuracy_vs_event_rate.png"
    )
    return results
