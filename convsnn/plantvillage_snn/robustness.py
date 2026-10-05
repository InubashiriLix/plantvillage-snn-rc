"""Frozen-checkpoint conductance variation evaluation; never trains a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import PIL
import snntorch
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import PlantVillageRecords, _transform
from .device import DeviceModel, HardwareState
from .train import _build_from_config, ordered_class_names
from .utils import resolve_device, seed_everything


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_variation(hardware, original, sigma, seed):
    """Independent static additive Gaussian error per physical cell, clipped.

    sigma is a fraction of (Gmax-Gmin), not a fraction of G or weight.
    Same seed pairs standard normal draws across strengths; no error accumulates.
    """
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    clipped, count = 0, 0
    for side in ("conductance_plus", "conductance_minus"):
        arrays = {}
        for name, source in original[side].items():
            source = np.asarray(source)
            if sigma == 0:
                arrays[name] = source.copy()
            else:
                value = source.astype(np.float64) + sigma * hardware.device.g_span * rng.standard_normal(source.shape)
                clipped += int(((value < hardware.device.g_min) | (value > hardware.device.g_max)).sum())
                arrays[name] = np.clip(value, hardware.device.g_min, hardware.device.g_max).astype(np.float32)
            count += source.size
        setattr(hardware, side, arrays)
    return clipped / count


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    output, targets = [], []
    for images, labels in loader:
        model.reset_state()
        result = model(images.to(device))
        output.extend(result["spike_counts"].argmax(1).cpu().tolist())
        targets.extend(labels.tolist())
    return np.asarray(output), np.asarray(targets)


def run(args):
    if args.batch_size < 1 or args.threads < 1 or args.workers < 0:
        raise ValueError("invalid batch size, threads or workers")
    if not args.sigmas or any(not math.isfinite(s) or s < 0 for s in args.sigmas):
        raise ValueError("sigmas must be finite and nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if any(args.sigmas) and len(args.seeds) < 2:
        raise ValueError("use at least two repeats for a sample standard deviation")
    if args.output.exists():
        raise FileExistsError(f"choose a new output directory: {args.output}")
    bundle = args.bundle
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name in ("checkpoint.pt", "test_manifest.json", "device.csv"):
        if sha256(bundle / name) != manifest["files_sha256"][name]:
            raise ValueError(f"bundle hash mismatch: {name}")
    checkpoint = torch.load(bundle / "checkpoint.pt", map_location="cpu", weights_only=True)
    config = ExperimentConfig(**checkpoint["config"])
    mapping = checkpoint["metadata"]["class_mapping"]
    if config.phase != "hw_finetune" or config.stage != "top10":
        raise ValueError("expected the frozen Top10 hardware checkpoint")
    records = json.loads((bundle / "test_manifest.json").read_text())
    if len(records) != manifest["sample_count"] or len({r["path"] for r in records}) != len(records):
        raise ValueError("test count mismatch or duplicate images")
    for record in records:
        path = args.data_root / record["path"]
        if not path.resolve().is_relative_to(args.data_root.resolve()) or not record["path"].startswith("val/"):
            raise ValueError("invalid test image path")
        if mapping[record["class_name"]] != record["target"]:
            raise ValueError("test label disagrees with checkpoint")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"test image hash mismatch: {record['path']}")
    torch.set_num_threads(args.threads)
    seed_everything(config.seed)
    device = resolve_device(args.device)
    model = _build_from_config(config, len(mapping)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    physical = DeviceModel(str(bundle / "device.csv"), config.device_v_bias,
                           config.device_g_bins, config.device_max_pulses, config.device_preprocess)
    hardware = HardwareState(model, physical, config.hw_scale_margin)
    original = checkpoint["hardware_state"]
    original = {key: {k: v.numpy().copy() if torch.is_tensor(v) else v for k, v in value.items()}
                if isinstance(value, dict) else value for key, value in original.items()}
    hardware.load_state_dict(original)
    ds = PlantVillageRecords(args.data_root, records, mapping,
                             _transform(config.image_size, "none", False, config.input_encoding))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    args.output.mkdir(parents=True)
    protocol = {
        "status": "running", "noise_model": "G'=clip(G+sigma*(Gmax-Gmin)*N(0,1),Gmin,Gmax)",
        "noise_scope": "Independent G+ and G- cell errors, sampled once per repeat, fixed across all test images and time steps; restored from original for each trial.",
        "paired_seeds_across_strengths": True, "retrained": False,
        "uncertainty": "Sample SD (ddof=1) across conductance-noise realizations of ONE trained model; not training-seed uncertainty or measurement data.",
        "sigmas": sorted(set([0.0, *args.sigmas])), "seeds": args.seeds,
        "bundle_manifest_sha256": sha256(bundle / "manifest.json"), "sample_count": len(records),
        "g_min": physical.g_min, "g_max": physical.g_max,
        "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
        "snntorch": snntorch.__version__, "pillow": PIL.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batch_size": args.batch_size, "threads": args.threads,
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    protocol_path = args.output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    fields = ["sigma", "noise_seed", "sample_count", "correct", "accuracy", "macro_recall", "clipped_fraction", "seconds"]
    with (args.output / "repeat_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sigma in protocol["sigmas"]:
            for seed in ([args.seeds[0]] if sigma == 0 else args.seeds):
                start = time.monotonic()
                fraction = apply_variation(hardware, original, sigma, seed)
                hardware.sync_to_model(model)
                predictions, labels = predict(model, loader, device)
                correct = int((predictions == labels).sum())
                accuracy = correct / len(labels)
                matrix = np.bincount(labels * len(mapping) + predictions, minlength=len(mapping)**2).reshape(len(mapping), len(mapping))
                if sigma == 0:
                    mismatch = int((predictions != np.array([r["prediction"] for r in records])).sum())
                    protocol["baseline_prediction_mismatches"] = mismatch
                    protocol["baseline_accuracy"] = accuracy
                    if mismatch or not math.isclose(accuracy, manifest["expected_accuracy"], abs_tol=1e-12):
                        protocol["status"] = "baseline_mismatch"
                        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
                        raise RuntimeError(f"baseline differs from published predictions: {mismatch} images, accuracy={accuracy}; no sweep performed")
                row = dict(sigma=sigma, noise_seed=seed, sample_count=len(labels), correct=correct,
                           accuracy=accuracy, macro_recall=float(np.mean(matrix.diagonal() / matrix.sum(1))),
                           clipped_fraction=fraction, seconds=time.monotonic()-start)
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                (args.output / f"confusion_sigma{sigma:g}_seed{seed}.json").write_text(json.dumps({"class_names": ordered_class_names(mapping), "matrix": matrix.tolist()}) + "\n")
                print(json.dumps(row), flush=True)
    summary = []
    for sigma in protocol["sigmas"]:
        group = [r for r in rows if r["sigma"] == sigma]
        summary.append(dict(sigma=sigma, repeats=len(group),
                            accuracy_mean=float(np.mean([r["accuracy"] for r in group])),
                            accuracy_std=float(np.std([r["accuracy"] for r in group], ddof=1)) if len(group)>1 else 0.0,
                            macro_recall_mean=float(np.mean([r["macro_recall"] for r in group]))))
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    protocol["status"] = "complete"
    protocol["files_sha256"] = {p.name: sha256(p) for p in args.output.iterdir() if p != protocol_path}
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0, .01, .02, .05, .1, .2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
