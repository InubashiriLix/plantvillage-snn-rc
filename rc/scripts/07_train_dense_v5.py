#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO_ROOT = Path(__file__).resolve().parents[2]

from plantvillage_rc.v5_experiments import run_v5_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dense-read 64-step v5 RC")
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "artifacts/preprocessed_v4")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/experiments_v5")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()
    options = {}
    if args.quick:
        options.update(
            encodings=("graded_q060",),
            tau_max_values=(128.0,),
            gamma_values=(2.0,),
            weight_decays=(1e-4,),
            top_k=1,
            max_epochs=5,
            patience=2,
        )
    result = run_v5_experiments(
        args.input_dir,
        args.output_dir,
        target_macro_recall=args.target,
        device=args.device,
        enable_augmentation=not args.no_augmentation,
        **options,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
