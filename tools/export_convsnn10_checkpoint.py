#!/usr/bin/env python3
"""Build a portable inference bundle from the verified original Top10 run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("artifacts/hardware_refresh_2026-08-06/final/conv_snn_top10_hw_finetune_seed42")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value, source):
    if isinstance(value, dict):
        return {key: portable(item, source) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(item, source) for item in value]
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, str):
        return value.replace(str(source) + "/", "")
    return value


def export(source, output):
    source = source.resolve()
    run = source / RUN
    provenance = json.loads((ROOT / "results/convsnn10/provenance.json").read_text())
    for name in ("metrics.json", "records/final_predictions.csv"):
        if sha256(run / name) != provenance["source_files_sha256"][(RUN / name).as_posix()]:
            raise ValueError(f"source does not match published result: {name}")
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    # Only load the locally owned, original training artifact. The exported
    # bundle contains tensors/basic types and supports weights_only=True.
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    metrics = json.loads((run / "metrics.json").read_text())
    if checkpoint["epoch"] != metrics["best_epoch"] or not checkpoint["hardware_state"]:
        raise ValueError("best checkpoint epoch or hardware state mismatch")
    state = portable({key: checkpoint[key] for key in
                      ("model_state", "hardware_state", "metadata", "config", "epoch")}, source)
    state["config"].update(data_root="data", device_csv="device.csv")
    with (run / "records/dataset_files.csv").open(newline="") as handle:
        inventory = {row["path"]: row for row in csv.DictReader(handle)}
    with (run / "records/final_predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    records = [{"path": row["path"], "class_name": row["true_class"],
                "target": int(row["true_index"]), "prediction": int(row["predicted_index"]),
                "sha256": inventory[row["path"]]["sha256"]} for row in predictions]
    if len(records) != metrics["final_test_metrics"]["sample_count"]:
        raise ValueError("frozen test sample count mismatch")
    output.mkdir(parents=True)
    torch.save(state, output / "checkpoint.pt")
    reloaded = torch.load(output / "checkpoint.pt", weights_only=True)
    for key, tensor in checkpoint["model_state"].items():
        assert torch.equal(tensor.cpu(), reloaded["model_state"][key])
    for side in ("conductance_plus", "conductance_minus"):
        for key, array in checkpoint["hardware_state"][side].items():
            np.testing.assert_array_equal(array, reloaded["hardware_state"][side][key].numpy())
    (output / "test_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    shutil.copyfile(source / "cnnRef/source/data.csv", output / "device.csv")
    manifest = {
        "format_version": 1,
        "source_run": RUN.as_posix(),
        "original_checkpoint_sha256": sha256(run / "best.pt"),
        "original_metrics_sha256": sha256(run / "metrics.json"),
        "export_note": "Original model tensors and G+/G- preserved exactly; inference-only export, no retraining. NumPy arrays converted to tensors; source-root prefixes removed; optimizer/RNG/history omitted.",
        "expected_accuracy": metrics["final_test_metrics"]["top1_accuracy"],
        "sample_count": len(records),
        "files_sha256": {name: sha256(output / name) for name in
                         ("checkpoint.pt", "test_manifest.json", "device.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/convsnn10_delivery")
    args = parser.parse_args()
    export(args.source_root, args.output)
