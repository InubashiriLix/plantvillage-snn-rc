#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plantvillage_rc.hardware_preview import read_preview


def main():
    base = Path(__file__).resolve().parents[1] / "hardware_example"
    parser = argparse.ArgumentParser(description="Validate preview and reconstruct N x 192 RGB values")
    parser.add_argument("--csv", type=Path, default=base / "preview_10_samples.csv")
    parser.add_argument("--mapping", type=Path, default=base / "class_mapping.csv")
    args = parser.parse_args()
    try:
        rgb, labels, signals = read_preview(args.csv, args.mapping)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Invalid hardware preview: {error}\n")
    print(f"RGB shape: {rgb.shape}; signals: {signals.shape}; labels: {labels.tolist()}")


if __name__ == "__main__":
    main()
