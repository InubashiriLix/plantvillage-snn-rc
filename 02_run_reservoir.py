#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from plantvillage_rc.reservoir import save_reservoir_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen reservoir states")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/states"))
    parser.add_argument("--quantile", choices=("q060", "q070", "q080"), default="q070")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.98)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--variant", choices=("full", "tonic", "event"), default="full")
    parser.add_argument(
        "--nonlinearity", choices=("saturating", "linear"), default="saturating"
    )
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--mask-kind", choices=("continuous", "binary"), default="continuous")
    parser.add_argument(
        "--device-profile",
        choices=("uniform", "row-spectrum"),
        default="row-spectrum",
    )
    args = parser.parse_args()
    config = save_reservoir_states(
        args.input_dir,
        args.output_dir,
        quantile_tag=args.quantile,
        alpha=args.alpha,
        gamma=args.gamma,
        variant=args.variant,
        nonlinearity=args.nonlinearity,
        mask_seed=args.mask_seed,
        mask_kind=args.mask_kind,
        device_profile=args.device_profile,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
    )
    with (args.output_dir / "reservoir_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
