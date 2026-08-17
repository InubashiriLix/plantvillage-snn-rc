#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from plantvillage_rc.experiments import (
    DEFAULT_EXPERIMENTS,
    ExperimentSpec,
    run_all_experiments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and evaluate RC linear readouts")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments_v2"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the default full q=0.70 model with a minimal search grid",
    )
    parser.add_argument(
        "--device-profile",
        choices=("uniform", "row-spectrum"),
        default="row-spectrum",
        help="Device dynamics used by --quick; the full suite runs both profiles",
    )
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="balanced",
        help="Readout weighting used by --quick; the full suite runs the required ablations",
    )
    args = parser.parse_args()
    if args.quick:
        specs = (
            ExperimentSpec(
                "quick_v2",
                quantile_tag="q060",
                device_profile=args.device_profile,
                class_weight=None if args.class_weight == "none" else "balanced",
                uniform_alphas=(0.9,),
                alpha_ranges=((0.05, 0.98),),
                gammas=(2.0,),
            ),
        )
    else:
        specs = DEFAULT_EXPERIMENTS
    ridge_alphas = (0.1, 1.0, 10.0) if args.quick else (
        1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0
    )
    results = run_all_experiments(
        args.input_dir,
        args.output_dir,
        specs=specs,
        ridge_alphas=ridge_alphas,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
