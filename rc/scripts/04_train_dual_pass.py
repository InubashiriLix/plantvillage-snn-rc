#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO_ROOT = Path(__file__).resolve().parents[2]

from plantvillage_rc.v3_experiments import run_v3_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v3 RGB+texture dual-pass RC")
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "artifacts/preprocessed")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/experiments_v3")
    parser.add_argument("--stage-gate", type=float, default=0.65)
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    kwargs = {}
    v2_selection = args.input_dir.parent / "experiments_v2" / "primary_selection.json"
    if v2_selection.is_file():
        v2_result = json.loads(v2_selection.read_text())["result"]
        v2_parameters = v2_result["selected"]
        if v2_parameters["device_profile"] == "row-spectrum":
            kwargs.update(
                alpha_min=float(v2_parameters["alpha_min"]),
                alpha_max=float(v2_parameters["alpha_max"]),
                gamma=float(v2_parameters["gamma"]),
            )
        kwargs["v2_validation_macro_recall"] = float(
            v2_result["val_macro_recall"]
        )
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
