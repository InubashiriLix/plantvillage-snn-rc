from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plantvillage-rc-matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .reservoir import dual_pass_states, generate_mask, reservoir_states, row_spectrum


def _model(ridge_alpha: float):
    return make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=ridge_alpha, class_weight="balanced", solver="svd"),
    )


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "predicted_class_count": int(len(np.unique(predictions))),
    }


def _plot_matrix(matrix: np.ndarray, classes: list[str], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title="Row-normalized recall",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    axis.tick_params(axis="y", labelsize=6)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _load(input_dir: Path, split: str, kind: str, tag: str = "q060") -> np.ndarray:
    infix = "" if kind == "rgb" else "_texture"
    return np.load(input_dir / f"{split}{infix}_inputs_{tag}.npy", mmap_mode="r")


def _dual_features(
    input_dir: Path,
    split: str,
    texture_tag: str,
    mask: np.ndarray,
    alpha: np.ndarray,
    *,
    gamma: float,
    gap_steps: int,
    carry_state: bool,
) -> dict[str, np.ndarray]:
    return dual_pass_states(
        _load(input_dir, split, "rgb", "q060"),
        _load(input_dir, split, "texture", texture_tag),
        mask,
        alpha=alpha,
        gamma=gamma,
        gap_steps=gap_steps,
        carry_state=carry_state,
    )


def run_v3_experiments(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    texture_tags: Iterable[str] = ("q060", "q070", "q080"),
    gap_steps: Iterable[int] = (0, 1, 2),
    ridge_alphas: Iterable[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    mask_seed: int = 42,
    gamma: float = 0.5,
    alpha_min: float = 0.0,
    alpha_max: float = 0.9,
    stage_gate: float = 0.65,
    target_macro_recall: float = 0.80,
) -> dict[str, object]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (input_dir / "preprocess_config.json").open(encoding="utf-8") as handle:
        preprocess_config = json.load(handle)
    classes: list[str] = preprocess_config["classes"]
    y_train = np.load(input_dir / "train_labels.npy")
    y_val = np.load(input_dir / "val_labels.npy")
    mask = generate_mask(mask_seed)
    alpha = row_spectrum(alpha_min, alpha_max)
    np.save(output_dir / "mask.npy", mask)
    np.save(output_dir / "alpha_nodes.npy", alpha)

    trials: list[dict[str, object]] = []
    for texture_tag in texture_tags:
        for gap in gap_steps:
            train_states = _dual_features(
                input_dir,
                "train",
                texture_tag,
                mask,
                alpha,
                gamma=gamma,
                gap_steps=gap,
                carry_state=True,
            )
            val_states = _dual_features(
                input_dir,
                "val",
                texture_tag,
                mask,
                alpha,
                gamma=gamma,
                gap_steps=gap,
                carry_state=True,
            )
            for ridge_alpha in ridge_alphas:
                classifier = _model(float(ridge_alpha))
                classifier.fit(train_states["combined_states"], y_train)
                predictions = classifier.predict(val_states["combined_states"])
                trials.append(
                    {
                        "texture_tag": texture_tag,
                        "gap_steps": int(gap),
                        "ridge_alpha": float(ridge_alpha),
                        **{f"val_{key}": value for key, value in _metrics(y_val, predictions).items()},
                    }
                )
    selected = max(
        trials,
        key=lambda row: (
            float(row["val_macro_recall"]),
            float(row["val_accuracy"]),
            int(row["val_predicted_class_count"]),
        ),
    )
    selected_train = _dual_features(
        input_dir,
        "train",
        str(selected["texture_tag"]),
        mask,
        alpha,
        gamma=gamma,
        gap_steps=int(selected["gap_steps"]),
        carry_state=True,
    )
    selected_val = _dual_features(
        input_dir,
        "val",
        str(selected["texture_tag"]),
        mask,
        alpha,
        gamma=gamma,
        gap_steps=int(selected["gap_steps"]),
        carry_state=True,
    )
    primary_model = _model(float(selected["ridge_alpha"]))
    primary_model.fit(selected_train["combined_states"], y_train)
    primary_val_predictions = primary_model.predict(selected_val["combined_states"])

    reset_train = _dual_features(
        input_dir,
        "train",
        str(selected["texture_tag"]),
        mask,
        alpha,
        gamma=gamma,
        gap_steps=0,
        carry_state=False,
    )
    reset_val = _dual_features(
        input_dir,
        "val",
        str(selected["texture_tag"]),
        mask,
        alpha,
        gamma=gamma,
        gap_steps=0,
        carry_state=False,
    )
    ablation_features = {
        "rgb_only": (
            selected_train["rgb_states"],
            selected_val["rgb_states"],
        ),
        "texture_only": (
            reset_train["texture_states"],
            reset_val["texture_states"],
        ),
        "dual_reset": (
            reset_train["combined_states"],
            reset_val["combined_states"],
        ),
        "dual_carry": (
            selected_train["combined_states"],
            selected_val["combined_states"],
        ),
    }
    ablations: dict[str, dict[str, float | int]] = {}
    for name, (train_features, val_features) in ablation_features.items():
        classifier = _model(float(selected["ridge_alpha"]))
        classifier.fit(train_features, y_train)
        ablations[name] = _metrics(y_val, classifier.predict(val_features))

    states_dir = output_dir / "states"
    states_dir.mkdir(exist_ok=True)
    for split, arrays in (("train", selected_train), ("val", selected_val)):
        for state_name, values in arrays.items():
            np.save(states_dir / f"{split}_{state_name}.npy", values)

    matrix = confusion_matrix(
        y_val,
        primary_val_predictions,
        labels=np.arange(len(classes)),
        normalize="true",
    )
    np.save(output_dir / "validation_confusion_matrix_balanced.npy", matrix)
    _plot_matrix(matrix, classes, output_dir / "validation_confusion_matrix_balanced.png")
    with (output_dir / "validation_classification_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            classification_report(
                y_val,
                primary_val_predictions,
                labels=np.arange(len(classes)),
                target_names=classes,
                output_dict=True,
                zero_division=0,
            ),
            handle,
            indent=2,
        )
    with (output_dir / "validation_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)

    passed_gate = float(selected["val_macro_recall"]) >= stage_gate
    reached_target = float(selected["val_macro_recall"]) >= target_macro_recall
    result: dict[str, object] = {
        "selected": selected,
        "dynamics": {
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "gamma": gamma,
            "mask_seed": mask_seed,
            "rgb_tag": "q060",
            "carry_state": True,
        },
        "feature_dimension": 216,
        "ablations_validation": ablations,
        "v2_validation_macro_recall": 0.4925877943357165,
        "macro_recall_gain_over_v2": float(selected["val_macro_recall"])
        - 0.4925877943357165,
        "state_inheritance_note": (
            "With both pass tails concatenated and a linear readout, carried and reset "
            "representations can be linearly equivalent under the Level-1 state equation."
        ),
        "stage_gate": stage_gate,
        "target_macro_recall": target_macro_recall,
        "passed_stage_gate": passed_gate,
        "reached_target": reached_target,
        "test_evaluated": False,
    }
    if reached_target:
        selected_test = _dual_features(
            input_dir,
            "test",
            str(selected["texture_tag"]),
            mask,
            alpha,
            gamma=gamma,
            gap_steps=int(selected["gap_steps"]),
            carry_state=True,
        )
        y_test = np.load(input_dir / "test_labels.npy")
        final_model = _model(float(selected["ridge_alpha"]))
        final_model.fit(
            np.concatenate(
                [selected_train["combined_states"], selected_val["combined_states"]]
            ),
            np.concatenate([y_train, y_val]),
        )
        test_predictions = final_model.predict(selected_test["combined_states"])
        result["test"] = _metrics(y_test, test_predictions)
        result["test_evaluated"] = True
        for state_name, values in selected_test.items():
            np.save(states_dir / f"test_{state_name}.npy", values)
        joblib.dump(final_model, output_dir / "readout.joblib")
    else:
        joblib.dump(primary_model, output_dir / "validation_readout.joblib")

    with (output_dir / "v3_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
