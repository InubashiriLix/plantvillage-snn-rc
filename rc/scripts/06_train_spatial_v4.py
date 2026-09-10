#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO_ROOT = Path(__file__).resolve().parents[2]

from plantvillage_rc.v4_experiments import run_v4_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 8x8 64-step spatial RC")
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "artifacts/preprocessed_v4")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/experiments_v4")
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="device used for the balanced ridge readout (default: cuda)",
    )
    args = parser.parse_args()
    kwargs = {}
    if args.quick:
        kwargs.update(
            quantile_tags=("q060",),
            tau_max_values=(128.0,),
            gamma_values=(1.0,),
            ridge_alphas=(1e-2,),
        )
    result = run_v4_experiments(
        args.input_dir,
        args.output_dir,
        target_macro_recall=args.target,
        device=args.device,
        **kwargs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
