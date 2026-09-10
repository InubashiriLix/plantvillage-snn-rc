#!/usr/bin/env python3
"""Copy the exact source images referenced by a Top-10 RGB export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_samples(csv_path: Path) -> list[dict]:
    samples = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(handle, None)
        if not header or not header.startswith("sample_id,label_index,label_name,source_path,"):
            raise ValueError("unexpected RGB CSV header")
        for line in handle:
            values = line.split(",", 4)[:4]
            samples.append({
                "sample_id": int(values[0]), "label_index": int(values[1]),
                "label_name": values[2], "source_path": values[3],
            })
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="exports/top10_rgb_csv/top10_rgb_64.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="exports/top10_rgb_csv/original_images")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    samples = read_samples(csv_path)
    if len(samples) != 300 or len({item["source_path"] for item in samples}) != 300:
        raise RuntimeError("expected exactly 300 unique exported samples")
    counts = Counter(item["label_index"] for item in samples)
    if set(counts) != set(range(10)) or set(counts.values()) != {30}:
        raise RuntimeError(f"expected 30 samples in each of 10 classes, found {dict(counts)}")

    rows = []
    for sample in samples:
        source = data_root / sample["source_path"]
        class_dir = output_dir / f"{sample['label_index']:02d}_{sample['label_name']}"
        class_dir.mkdir(parents=True, exist_ok=True)
        destination = class_dir / source.name
        if destination.exists() and destination.resolve() != source.resolve():
            # Existing files are overwritten only after their exact target has
            # been deterministically resolved from the export manifest.
            destination.unlink()
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        copied_hash = sha256_file(destination)
        if source_hash != copied_hash:
            raise RuntimeError(f"copy verification failed for {source}")
        rows.append({
            **sample,
            "copied_path": destination.relative_to(output_dir.parent).as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": copied_hash,
        })

    manifest_path = output_dir.parent / "original_images_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_csv": str(csv_path), "output_dir": str(output_dir),
        "manifest": str(manifest_path), "image_count": len(rows),
        "class_counts": dict(sorted(counts.items())), "sha256_verified": True,
    }
    summary_path = output_dir.parent / "original_images_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
