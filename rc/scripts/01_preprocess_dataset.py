#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO_ROOT = Path(__file__).resolve().parents[2]

from plantvillage_rc.preprocessing import PreprocessSettings, preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PlantVillage for RGB event RC")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/preprocessed")
    parser.add_argument(
        "--max-per-class",
        type=int,
        help="Deterministic per-class cap for smoke tests; omit for the full dataset",
    )
    class_filter = parser.add_mutually_exclusive_group()
    class_filter.add_argument(
        "--class-prefix",
        help="Keep class directories beginning with this prefix, e.g. Tomato___",
    )
    class_filter.add_argument(
        "--classes",
        nargs="+",
        help="Explicit class directory names to keep",
    )
    class_filter.add_argument("--classes-file", type=Path, help="JSON list of fixed class names")
    parser.add_argument(
        "--patch-grid",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLUMNS"),
        default=(3, 4),
        help="Spatial patch grid; use 8 8 for the 64-step v4 scan",
    )
    parser.add_argument("--active-columns", type=int, choices=(9, 12), default=9)
    args = parser.parse_args()
    config = preprocess_dataset(
        args.data_root,
        args.output_dir,
        max_per_class=args.max_per_class,
        class_prefix=args.class_prefix,
        selected_classes=(json.loads(args.classes_file.read_text()) if args.classes_file else args.classes),
        settings=PreprocessSettings(
            patch_grid=tuple(args.patch_grid),
            active_columns=args.active_columns,
        ),
    )
    print(json.dumps({"split_sizes": config["split_sizes"]}, indent=2))


if __name__ == "__main__":
    main()
