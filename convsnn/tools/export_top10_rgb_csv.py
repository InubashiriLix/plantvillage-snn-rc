#!/usr/bin/env python3
"""Export a deterministic, balanced Top-10 PlantVillage RGB dataset as CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranked_top10(metrics: dict) -> list[dict]:
    per_class = metrics["selection_metrics"]["per_class"]
    ranked = sorted(
        per_class.items(),
        key=lambda item: (-item[1]["recall"], -item[1]["precision"], item[0]),
    )[:10]
    return [
        {
            "label_index": new_index,
            "label_name": name,
            "original_index": values["index"],
            "selection_recall": values["recall"],
            "selection_precision": values["precision"],
        }
        for new_index, (name, values) in enumerate(ranked)
    ]


def select_samples(data_root: Path, classes: list[dict], per_class: int,
                   seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    selected = []
    for class_info in classes:
        directory = data_root / "train" / class_info["label_name"]
        candidates = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(candidates) < per_class:
            raise RuntimeError(
                f"{class_info['label_name']} has {len(candidates)} images; {per_class} required"
            )
        indices = rng.choice(len(candidates), size=per_class, replace=False)
        for path in sorted(candidates[int(index)] for index in indices):
            selected.append({
                "label_index": class_info["label_index"],
                "label_name": class_info["label_name"],
                "source_path": path.relative_to(data_root).as_posix(),
                "absolute_path": path,
            })
    # Keep classes balanced while avoiding a block of 30 identical labels.
    order = rng.permutation(len(selected))
    shuffled = [selected[int(index)] for index in order]
    for sample_id, sample in enumerate(shuffled):
        sample["sample_id"] = sample_id
    return shuffled


def pixel_columns(size: int):
    for channel in "RGB":
        for y in range(size):
            for x in range(size):
                yield f"{channel}_{y:03d}_{x:03d}"


def channel_first_flat(image: Image.Image, size: int) -> np.ndarray:
    rgb = image.convert("RGB")
    if rgb.size != (size, size):
        rgb = rgb.resize((size, size), resample=Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.uint8)
    return np.transpose(array, (2, 0, 1)).reshape(-1)


def csv_prefix(sample: dict) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow([
        sample["sample_id"], sample["label_index"],
        sample["label_name"], sample["source_path"],
    ])
    return buffer.getvalue()


def write_header(handle, size: int) -> None:
    handle.write("sample_id,label_index,label_name,source_path,")
    first = True
    for column in pixel_columns(size):
        if not first:
            handle.write(",")
        handle.write(column)
        first = False
    handle.write("\n")


def export_csvs(samples: list[dict], outputs: dict[int, Path]) -> dict[int, dict]:
    handles = {}
    try:
        for size, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            handles[size] = path.open("w", encoding="utf-8", newline="")
            write_header(handles[size], size)
        for sample in samples:
            with Image.open(sample["absolute_path"]) as image:
                prefix = csv_prefix(sample)
                for size, handle in handles.items():
                    pixels = channel_first_flat(image, size)
                    if pixels.size != size * size * 3 or pixels.min() < 0 or pixels.max() > 255:
                        raise RuntimeError(f"invalid RGB pixels for {sample['source_path']}")
                    handle.write(prefix)
                    handle.write(",")
                    handle.write(",".join(str(int(value)) for value in pixels))
                    handle.write("\n")
    finally:
        for handle in handles.values():
            handle.close()
    return {
        size: {
            "path": str(path.resolve()),
            "rows": len(samples),
            "columns": 4 + size * size * 3,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for size, path in outputs.items()
    }


def validate_csv(path: Path, size: int, expected_samples: list[dict]) -> dict:
    expected_commas = 3 + size * size * 3
    counts = Counter()
    paths = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline()
        if header.count(",") != expected_commas:
            raise RuntimeError(f"wrong header width in {path}")
        for row_index, line in enumerate(handle):
            if line.count(",") != expected_commas:
                raise RuntimeError(f"wrong column count in {path} row {row_index + 2}")
            # Current PlantVillage labels and relative paths contain no commas;
            # limiting the split avoids materializing ~200k pixel strings merely
            # to validate four metadata fields.
            metadata = line.split(",", 4)[:4]
            expected = expected_samples[row_index]
            actual = (int(metadata[0]), int(metadata[1]), metadata[2], metadata[3])
            wanted = (
                expected["sample_id"], expected["label_index"],
                expected["label_name"], expected["source_path"],
            )
            if actual != wanted:
                raise RuntimeError(f"metadata mismatch in {path} row {row_index + 2}")
            counts[int(metadata[1])] += 1
            paths.append(metadata[3])
    if len(paths) != len(expected_samples) or len(paths) != len(set(paths)):
        raise RuntimeError(f"row count or uniqueness check failed for {path}")
    expected_per_class = len(expected_samples) // 10
    if set(counts) != set(range(10)) or set(counts.values()) != {expected_per_class}:
        raise RuntimeError(f"class-balance check failed for {path}: {dict(counts)}")
    return {"rows": len(paths), "class_counts": dict(sorted(counts.items())), "paths": paths}


def validate_reconstruction(path: Path, size: int, samples: list[dict],
                            selected_rows: set[int]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        next(handle)
        for row_index, line in enumerate(handle):
            if row_index not in selected_rows:
                continue
            fields = next(csv.reader([line], strict=True))
            pixels = np.asarray(fields[4:], dtype=np.uint8)
            with Image.open(samples[row_index]["absolute_path"]) as image:
                expected = channel_first_flat(image, size)
            if not np.array_equal(pixels, expected):
                raise RuntimeError(f"pixel reconstruction mismatch in {path} row {row_index + 2}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default="artifacts/accuracy_investigation/long_epoch/conv_snn_all38_ideal_seed42/metrics.json",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="artifacts/exports/top10_rgb_csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=30)
    args = parser.parse_args()

    metrics_path = Path(args.metrics).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    classes = ranked_top10(metrics)
    samples = select_samples(data_root, classes, args.samples_per_class, args.seed)
    if len(samples) > 300:
        raise RuntimeError("export exceeds the agreed total limit of 300 samples")

    outputs = {size: output_dir / f"top10_rgb_{size}.csv" for size in (64, 256)}
    output_metadata = export_csvs(samples, outputs)
    validations = {size: validate_csv(path, size, samples) for size, path in outputs.items()}
    if validations[64]["paths"] != validations[256]["paths"]:
        raise RuntimeError("64 and 256 CSV sample order differs")
    reconstruction_rows = {0, len(samples) // 2, len(samples) - 1}
    for size, path in outputs.items():
        validate_reconstruction(path, size, samples, reconstruction_rows)

    manifest = {
        "format_version": 1,
        "description": "Balanced RGB export of the latest 100-epoch all38 ConvSNN Top-10 classes",
        "source_metrics": str(metrics_path),
        "source_metrics_sha256": sha256_file(metrics_path),
        "source_best_epoch": metrics["best_epoch"],
        "ranking": {
            "primary": "selection_recall_descending",
            "tie_breakers": ["selection_precision_descending", "class_name_ascending"],
            "classes": classes,
        },
        "sampling": {
            "source_partition": "complete original train directory",
            "seed": args.seed,
            "samples_per_class": args.samples_per_class,
            "total_samples": len(samples),
            "without_replacement": True,
        },
        "encoding": {
            "color_mode": "RGB", "value_type": "uint8", "value_range": [0, 255],
            "flatten_order": "channel-first: all R, then all G, then all B; row-major within each channel",
            "resize_64": "Pillow bilinear",
            "resize_256": "original pixels when already 256x256; otherwise Pillow bilinear",
        },
        "metadata_columns": ["sample_id", "label_index", "label_name", "source_path"],
        "outputs": {str(size): metadata for size, metadata in output_metadata.items()},
        "validation": {
            "row_and_column_counts": True, "unique_source_paths": True,
            "balanced_class_counts": True, "identical_sample_order": True,
            "pixel_range_checked_during_export": True,
            "reconstruction_rows": sorted(reconstruction_rows),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **output_metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
