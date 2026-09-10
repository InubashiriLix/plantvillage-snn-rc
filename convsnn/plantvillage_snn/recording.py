from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


SCHEMA_VERSION = 1


def tensor_summary(tensor: torch.Tensor) -> dict[str, float | int]:
    value = tensor.detach().float()
    count = value.numel()
    finite = torch.isfinite(value)
    safe = value[finite]
    if safe.numel() == 0:
        return {"count": count, "mean": 0.0, "std": 0.0, "min": 0.0,
                "max": 0.0, "l1": 0.0, "l2": 0.0, "near_zero_rate": 0.0,
                "nonfinite_count": count}
    return {
        "count": count,
        "mean": float(safe.mean()),
        "std": float(safe.std(unbiased=False)),
        "min": float(safe.min()),
        "max": float(safe.max()),
        "l1": float(safe.abs().sum()),
        "l2": float(torch.linalg.vector_norm(safe)),
        "near_zero_rate": float((safe.abs() < 1e-8).float().mean()),
        "nonfinite_count": int(count - safe.numel()),
    }


def parameter_summaries(model: nn.Module) -> dict[str, dict[str, float | int]]:
    return {name: tensor_summary(parameter) for name, parameter in model.named_parameters()}


def batchnorm_summaries(model: nn.Module) -> dict[str, dict[str, Any]]:
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            result[name] = {
                "running_mean": tensor_summary(module.running_mean),
                "running_var": tensor_summary(module.running_var),
                "num_batches_tracked": int(module.num_batches_tracked),
            }
    return result


class GradientRecorder:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, float | int]] = {}

    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            summary = tensor_summary(parameter.grad)
            item = self.values.setdefault(name, {
                "batches": 0, "l2_sum": 0.0, "l2_max": 0.0,
                "mean_abs_sum": 0.0, "nonfinite_count": 0,
            })
            item["batches"] += 1
            item["l2_sum"] += summary["l2"]
            item["l2_max"] = max(item["l2_max"], summary["l2"])
            item["mean_abs_sum"] += summary["l1"] / max(summary["count"], 1)
            item["nonfinite_count"] += summary["nonfinite_count"]

    def compute(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "batches": value["batches"],
                "mean_l2": value["l2_sum"] / max(value["batches"], 1),
                "max_l2": value["l2_max"],
                "mean_abs": value["mean_abs_sum"] / max(value["batches"], 1),
                "nonfinite_count": value["nonfinite_count"],
            }
            for name, value in self.values.items()
        }


def update_ratios(before: dict[str, torch.Tensor], model: nn.Module) -> dict[str, float]:
    result = {}
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        old = before[name].to(parameter.device)
        delta = torch.linalg.vector_norm(parameter.detach() - old)
        denominator = torch.linalg.vector_norm(old).clamp_min(1e-12)
        result[name] = float(delta / denominator)
    return result


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "schema_version": SCHEMA_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": commit,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_inventory(data_root: str | Path) -> tuple[list[dict[str, Any]], str]:
    root = Path(data_root).resolve()
    rows = []
    aggregate = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        rows.append({"path": relative, "size_bytes": size, "sha256": digest})
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode())
    return rows, aggregate.hexdigest()


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def export_metrics_tables(records_dir: Path, history: list[dict], selection: dict,
                          final_test: dict | None) -> None:
    epoch_rows = []
    layer_rows = []
    temporal_rows = []
    for record in history:
        selection_epoch = record["selection"]
        row = {
            "epoch": record["epoch"], "train_loss": record["train_loss"],
            "train_accuracy": record["train_accuracy"],
            "selection_loss": selection_epoch["loss"],
            "selection_accuracy": selection_epoch["top1_accuracy"],
            "selection_macro_recall": selection_epoch["macro_recall"],
            "duration_seconds": record.get("duration_seconds"),
            "cuda_peak_memory_bytes": record.get("cuda_peak_memory_bytes"),
        }
        row.update({f"loss_{key}": value for key, value in record.get("loss_components", {}).items()})
        epoch_rows.append(row)
        for layer, stats in selection_epoch.get("snn_activity", {}).items():
            layer_rows.append({"epoch": record["epoch"], "layer": layer, **stats})
        for step, accuracy in enumerate(selection_epoch.get("temporal_accuracy", []), 1):
            temporal_rows.append({"epoch": record["epoch"], "time_step": step, "accuracy": accuracy})
    write_csv(records_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(records_dir / "layer_activity.csv", layer_rows)
    write_csv(records_dir / "temporal_accuracy.csv", temporal_rows)

    per_class_rows = []
    confusion_rows = []
    for split, metrics in (("selection", selection), ("final_test", final_test)):
        if not metrics:
            continue
        for class_name, values in metrics["per_class"].items():
            per_class_rows.append({"split": split, "class_name": class_name, **values})
        names = list(metrics["per_class"])
        for true_index, matrix_row in enumerate(metrics["confusion_matrix"]):
            for predicted_index, count in enumerate(matrix_row):
                confusion_rows.append({
                    "split": split, "true_index": true_index,
                    "true_class": names[true_index], "predicted_index": predicted_index,
                    "predicted_class": names[predicted_index], "count": count,
                })
    write_csv(records_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(records_dir / "confusion_matrix_long.csv", confusion_rows)
