#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from plantvillage_rc.v3_experiments import run_v3_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v3 RGB+texture dual-pass RC")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments_v3"))
    parser.add_argument("--stage-gate", type=float, default=0.65)
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    kwargs = {}
    if args.quick:
        kwargs.update(texture_tags=("q060",), gap_steps=(0,), ridge_alphas=(1e-4, 0.1))
    result = run_v3_experiments(
        args.input_dir,
        args.output_dir,
        stage_gate=args.stage_gate,
        target_macro_recall=args.target,
        **kwargs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

