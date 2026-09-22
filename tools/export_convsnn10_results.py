#!/usr/bin/env python3
"""Export paper plotting data from the original, frozen ConvSNN Top10 runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


CAMPAIGN = Path("artifacts/hardware_refresh_2026-08-06")
PHASES = ("ideal", "hw_finetune")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(source_root: Path, output: Path):
    source_root = source_root.resolve()
    sources = {}

    def record(path):
        relative = path.relative_to(source_root).as_posix()
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return relative

    mapping_path = source_root / CAMPAIGN / "top10_classes.json"
    record(mapping_path)
    selection = json.loads(mapping_path.read_text())
    mapping = sorted(selection["classes"], key=lambda row: row["new_index"])
    if [row["new_index"] for row in mapping] != list(range(10)):
        raise ValueError("expected ten contiguous class indices")
    names = [row["name"] for row in mapping]
    metrics, summary, per_class, confusion, temporal, epochs = {}, [], [], [], [], []

    for phase in PHASES:
        directory = source_root / CAMPAIGN / "final" / f"conv_snn_top10_{phase}_seed42"
        path = directory / "metrics.json"
        source = record(path)
        raw = json.loads(path.read_text())
        metrics[phase] = {
            "source": source,
            "source_sha256": sources[source],
            "best_epoch": raw["best_epoch"],
            "executed_epochs": raw["executed_epochs"],
            "final_evaluation": raw["final_evaluation"],
            "model_cost": raw["model_cost"],
            "pulse_statistics": raw["pulse_statistics"],
            "conductance_statistics": raw["conductance_statistics"],
        }
        for split, filename in (("selection", "selection_predictions.csv"), ("final_test", "final_predictions.csv")):
            values = raw[f"{split}_metrics"]
            matrix = values["confusion_matrix"]
            if len(matrix) != 10 or any(len(row) != 10 for row in matrix):
                raise ValueError(f"invalid confusion matrix: {phase}/{split}")
            # Independent reconciliation with the recorded per-image predictions.
            prediction_path = directory / "records" / filename
            record(prediction_path)
            reconstructed = [[0] * 10 for _ in range(10)]
            for prediction in read_csv(prediction_path):
                truth, predicted = int(prediction["true_index"]), int(prediction["predicted_index"])
                if not (0 <= truth < 10 and 0 <= predicted < 10):
                    raise ValueError("prediction class index out of range")
                if prediction["true_class"] != names[truth] or prediction["predicted_class"] != names[predicted]:
                    raise ValueError("prediction class names disagree with mapping")
                reconstructed[truth][predicted] += 1
            if reconstructed != matrix:
                raise ValueError(f"predictions disagree with metrics: {phase}/{split}")
            supports = [sum(row) for row in matrix]
            total = sum(supports)
            accuracy = sum(matrix[i][i] for i in range(10)) / total
            recall = sum(matrix[i][i] / supports[i] for i in range(10)) / 10
            if total != values["sample_count"] or not math.isclose(accuracy, values["top1_accuracy"], abs_tol=1e-12) or not math.isclose(recall, values["macro_recall"], abs_tol=1e-12):
                raise ValueError(f"metric reconciliation failed: {phase}/{split}")
            metrics[phase][f"{split}_metrics"] = values
            summary.append(dict(phase=phase, split=split, sample_count=total,
                                accuracy=accuracy, macro_recall=recall,
                                best_epoch=raw["best_epoch"], executed_epochs=raw["executed_epochs"]))
            for i, name in enumerate(names):
                entry = values["per_class"][name]
                predicted_total = sum(row[i] for row in matrix)
                precision = matrix[i][i] / predicted_total if predicted_total else 0.0
                if entry["index"] != i or entry["samples"] != supports[i] or entry["correct"] != matrix[i][i] or not math.isclose(entry["recall"], matrix[i][i] / supports[i], abs_tol=1e-12) or not math.isclose(entry["precision"], precision, abs_tol=1e-12):
                    raise ValueError(f"per-class metric reconciliation failed: {phase}/{split}/{name}")
                per_class.append(dict(phase=phase, split=split, class_name=name, **entry))
                for j, predicted_name in enumerate(names):
                    confusion.append(dict(phase=phase, split=split, true_index=i,
                                          true_class=name, predicted_index=j, predicted_class=predicted_name,
                                          count=matrix[i][j], row_fraction=matrix[i][j] / supports[i]))
            for step, step_accuracy in enumerate(values["temporal_accuracy"], 1):
                temporal.append(dict(phase=phase, split=split, time_step=step, accuracy=step_accuracy))
        epoch_path = directory / "records" / "epoch_metrics.csv"
        record(epoch_path)
        records = read_csv(epoch_path)
        if [int(row["epoch"]) for row in records] != list(range(1, raw["executed_epochs"] + 1)):
            raise ValueError(f"incomplete epoch history: {phase}")
        for row in records:
            epochs.append(dict(phase=phase, is_best_epoch=int(row["epoch"]) == raw["best_epoch"], **row))

    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summary), ("per_class_metrics", per_class),
                       ("confusion_matrix_long", confusion), ("temporal_accuracy", temporal),
                       ("epoch_metrics", epochs), ("class_mapping", mapping)):
        write_csv(output / f"{name}.csv", list(rows[0]), rows)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    provenance = {
        "source_project": "original PlantVillage project",
        "source_files_sha256": sources,
        "class_selection": {
            "ranking_metric": selection["ranking_metric"],
            "tie_breakers": selection["tie_breakers"],
            "source_metrics": Path(selection["source_metrics"]).relative_to(source_root).as_posix(),
            "source_metrics_sha256": selection["source_metrics_sha256"],
        },
        "validation": "Both selection and final_test confusion matrices reconciled against per-image predictions; accuracy, macro recall, precision and supports recomputed.",
        "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(output.iterdir()) if path.suffix in (".csv", ".json") and path.name != "provenance.json"},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    for row in summary:
        print(f"{row['phase']}/{row['split']}: N={row['sample_count']}, accuracy={row['accuracy']:.8%}, macro_recall={row['macro_recall']:.8%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Root of the original PlantVillage project")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results/convsnn10")
    args = parser.parse_args()
    try:
        export(args.source_root, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Export failed: {error}\n")


if __name__ == "__main__":
    main()
