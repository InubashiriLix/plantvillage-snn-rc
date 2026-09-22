#!/usr/bin/env python3
"""Audit, evaluate and fit classifiers using only measured hardware current."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plantvillage_rc.hardware_data import load_dataset, read_responses, write_csv
from plantvillage_rc.hardware_learning import audit_sources, evaluate, fit_final, unit_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "evaluate", "fit"):
        command = sub.add_parser(name)
        command.add_argument("--responses", required=True, type=Path)
        command.add_argument("--inputs", required=True, type=Path)
        command.add_argument("--workbook", required=True, type=Path)
        command.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts_hardware300")
        if name == "audit":
            command.add_argument("--unit-diagnostic", action="store_true")
        else:
            command.add_argument("--workers", type=int, default=6)
            command.add_argument("--hours", type=float, default=7.0 if name == "evaluate" else 1.0)
    predict = sub.add_parser("predict")
    predict.add_argument("--model", required=True, type=Path, help="Trusted local model.joblib only")
    predict.add_argument("--responses", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "predict":
            import joblib
            from threadpoolctl import threadpool_limits
            model = joblib.load(args.model)
            x, metadata = read_responses(args.responses, require_labels=False)
            with threadpool_limits(limits=1):
                probability = model.predict_proba(x)
            rows = []
            for meta, prob in zip(metadata, probability):
                prediction = int(prob.argmax())
                rows.append({"sample_index": meta["subset_sample_index"], "sample_id": meta["sample_id"],
                             "predicted_label": prediction, "predicted_class": model.classes[prediction],
                             **{f"probability_{i}": float(value) for i, value in enumerate(prob)}})
            write_csv(args.output, rows)
            print(f"Wrote {len(rows)} predictions to {args.output}")
            return
        if hasattr(args, "workers") and (args.workers < 1 or args.hours <= 0):
            raise ValueError("workers and hours must be positive")
        paths = {key: getattr(args, key) for key in ("responses", "inputs", "workbook")}
        dataset = load_dataset(**paths)
        audit_sources(dataset, paths, args.output_dir)
        if args.command == "audit":
            print(f"Aligned {len(dataset.y)} samples: {dataset.x.shape}; historical mean={dataset.audit['historical_mean_accuracy']:.6%}")
            if args.unit_diagnostic:
                unit_diagnostic(dataset, args.output_dir)
        elif args.command == "evaluate":
            evaluate(dataset, args.output_dir, workers=args.workers, hours=args.hours)
        else:
            fit_final(dataset, args.output_dir, workers=args.workers, hours=args.hours)
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        parser.exit(2, f"Hardware experiment stopped: {error}\n")


if __name__ == "__main__":
    main()
