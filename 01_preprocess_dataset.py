#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from plantvillage_rc.preprocessing import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PlantVillage for RGB event RC")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/preprocessed"))
    parser.add_argument(
        "--max-per-class",
        type=int,
        help="Deterministic per-class cap for smoke tests; omit for the full dataset",
    )
    args = parser.parse_args()
    config = preprocess_dataset(
        args.data_root, args.output_dir, max_per_class=args.max_per_class
    )
    print(json.dumps({"split_sizes": config["split_sizes"]}, indent=2))


if __name__ == "__main__":
    main()

