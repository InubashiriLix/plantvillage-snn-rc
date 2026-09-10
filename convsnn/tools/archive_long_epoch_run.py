#!/usr/bin/env python3
"""Create a complete retrospective record bundle for the 100-epoch all38 run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

import torch

from plantvillage_snn.config import ExperimentConfig
from plantvillage_snn.data import build_dataloaders, class_weights
from plantvillage_snn.recording import (
    SCHEMA_VERSION, batchnorm_summaries, dataset_inventory, environment_snapshot,
    export_metrics_tables, parameter_summaries, sha256_file, write_csv,
)
from plantvillage_snn.train import (
    _build_from_config, evaluate, load_checkpoint, ordered_class_names,
)
from plantvillage_snn.utils import jsonable, resolve_device, seed_everything, write_json


def prediction_summary(path: Path, class_names: list[str]) -> dict:
    confusion = [[0 for _ in class_names] for _ in class_names]
    samples = correct = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            true_index = int(row["true_index"])
            predicted_index = int(row["predicted_index"])
            confusion[true_index][predicted_index] += 1
            samples += 1
            correct += true_index == predicted_index
    recalls = [
        confusion[i][i] / max(sum(confusion[i]), 1) for i in range(len(class_names))
    ]
    return {
        "sample_count": samples, "top1_accuracy": correct / max(samples, 1),
        "macro_recall": sum(recalls) / max(len(recalls), 1), "confusion_matrix": confusion,
    }


def source_snapshot(workspace: Path, destination: Path) -> list[str]:
    candidates = [workspace / "plantvillage_snn", workspace / "tests", workspace / "configs",
                  workspace / "pyproject.toml", workspace / "README.md"]
    included = []
    with tarfile.open(destination, "w:gz") as archive:
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.is_dir():
                paths = sorted(p for p in candidate.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
            else:
                paths = [candidate]
            for path in paths:
                archive.add(path, arcname=path.relative_to(workspace))
                included.append(path.relative_to(workspace).as_posix())
    return included


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--records-dir")
    args = parser.parse_args()
    workspace = Path.cwd().resolve()
    metrics_path = Path(args.metrics).resolve()
    artifact = json.loads(metrics_path.read_text())
    run_dir = metrics_path.parent
    records_dir = Path(args.records_dir).resolve() if args.records_dir else run_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(**artifact["config"]).with_overrides({
        "evaluate_final": True, "max_selection_samples": None, "max_final_samples": None,
        "num_workers": 0,
        "recording_enabled": True, "record_sample_predictions": True,
        "recording_dir": str(records_dir),
    })
    seed_everything(config.seed)
    device = resolve_device(config.device)
    data = build_dataloaders(config)
    class_names = ordered_class_names(data.class_to_index)
    checkpoint_path = Path(artifact["checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = _build_from_config(config, len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    weight = class_weights(data.sample_counts["train"], device) if config.class_balance == "class_weights" else None

    selection = evaluate(model, data.selection, class_names, device, weight, "selection",
                         config.membrane_loss_weight, records_dir / "selection_predictions.csv")
    final_test = evaluate(model, data.final_test, class_names, device, weight, "final_test",
                          config.membrane_loss_weight, records_dir / "final_predictions.csv")
    history = artifact.get("executed_history", artifact["history"])
    export_metrics_tables(records_dir, history, selection, final_test)
    write_json(records_dir / "best_internal_summaries.json", jsonable({
        "schema_version": SCHEMA_VERSION, "parameters": parameter_summaries(model),
        "batchnorm": batchnorm_summaries(model),
    }))
    write_json(records_dir / "environment.json", environment_snapshot(device))

    inventory, aggregate = dataset_inventory(config.data_root)
    write_csv(records_dir / "dataset_files.csv", inventory, ["path", "size_bytes", "sha256"])
    snapshot_files = source_snapshot(workspace, records_dir / "source_snapshot.tar.gz")
    selection_csv = prediction_summary(records_dir / "selection_predictions.csv", class_names)
    final_csv = prediction_summary(records_dir / "final_predictions.csv", class_names)
    unavailable = {
        "schema_version": SCHEMA_VERSION,
        "fields": {
            "epoch_gradient_summaries": "not captured during the completed historical run",
            "epoch_parameter_summaries": "not captured during the completed historical run",
            "epoch_update_ratios": "not captured during the completed historical run",
            "epoch_duration_seconds": "not captured during the completed historical run",
            "epoch_cuda_peak_memory_bytes": "not captured during the completed historical run",
            "optimizer_state_at_each_epoch": "not captured; best checkpoint predates the recorder",
        },
    }
    write_json(records_dir / "unavailable_fields.json", unavailable)
    provenance_files = [metrics_path, checkpoint_path, Path(data.split_manifest_path), Path(config.device_csv)]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "retrospective record bundle for completed 100-epoch ConvSNN all38 ideal run",
        "run": artifact["run"], "best_epoch": artifact["best_epoch"],
        "executed_epochs": artifact.get("executed_epochs", len(history)),
        "dataset": {"hash_mode": "sha256", "file_count": len(inventory), "aggregate_sha256": aggregate},
        "artifacts": {str(path.resolve()): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                      for path in provenance_files},
        "source_snapshot": {"path": "source_snapshot.tar.gz", "file_count": len(snapshot_files)},
        "prediction_validation": {"selection": selection_csv, "final_test": final_csv},
        "metric_agreement_tolerance": 1e-12,
        "raw_activations_saved": False, "raw_gradients_saved": False,
    }
    for split, calculated, expected in (
        ("selection", selection_csv, selection), ("final_test", final_csv, final_test)
    ):
        for key in ("sample_count", "top1_accuracy", "macro_recall"):
            if abs(calculated[key] - expected[key]) > 1e-12:
                raise RuntimeError(f"{split} prediction CSV does not reproduce {key}")
        if calculated["confusion_matrix"] != expected["confusion_matrix"]:
            raise RuntimeError(f"{split} prediction CSV does not reproduce confusion matrix")
    write_json(records_dir / "recording_manifest.json", manifest)
    print(json.dumps({"records_dir": str(records_dir), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
