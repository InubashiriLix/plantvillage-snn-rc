#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO_ROOT = Path(__file__).resolve().parents[2]

from plantvillage_rc.hardware_export import export_hardware_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the selected ten-class 8x8 inputs for hardware playback"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "artifacts_best10/preprocessed_v4",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=REPO_ROOT / "artifacts_best10/experiments_v5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts_best10/hardware_export_v1",
    )
    args = parser.parse_args()
    manifest = export_hardware_dataset(
        args.input_dir,
        args.experiment_dir,
        args.output_dir,
    )
    print(json.dumps({"splits": manifest["splits"], "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
