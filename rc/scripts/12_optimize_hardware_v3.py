#!/usr/bin/env python3
"""Third-round pure-current hardware optimization and delivery CLI.

The model sees only the nine measured current channels.  The paired optical
inputs are used solely by the historical audit loader to verify alignment and
are never passed to fitting or prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rc"))

from plantvillage_rc.hardware_data import load_dataset, read_responses, write_csv, write_json
from plantvillage_rc.hardware_refinement_learning import evaluate, fit_final
from plantvillage_rc.hardware_models import probabilities


def dataset_from(args):
    return load_dataset(args.responses, args.inputs, args.workbook)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("audit", "search", "evaluate", "fit", "predict", "report"))
    p.add_argument("--responses", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--workbook", type=Path)
    p.add_argument("--configs", type=Path, default=ROOT / "rc/configs/hardware_refinement_v3.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts_hardware300_v3/run")
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--hours", type=float, default=8.0)
    args = p.parse_args()

    if args.action == "predict":
        if not args.responses or not args.model or not args.output:
            p.error("predict requires --responses, --model and --output")
        x, metadata = read_responses(args.responses, require_labels=False)
        import joblib
        model = joblib.load(args.model)
        prediction = model.predict(x)
        rows = [{**meta, "prediction": int(label)} for meta, label in zip(metadata, prediction)]
        write_csv(args.output, rows)
        print(json.dumps({"count": len(rows), "output": str(args.output)}, ensure_ascii=False))
        return

    if not args.responses or not args.inputs or not args.workbook:
        p.error(f"{args.action} requires --responses, --inputs and --workbook")
    dataset = dataset_from(args)
    configs = json.loads(args.configs.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "audit":
        write_json(args.output_dir / "audit.json", dataset.audit)
        print(json.dumps(dataset.audit, ensure_ascii=False))
    elif args.action == "search":
        from plantvillage_rc.hardware_refinement_learning import freeze, run_inner
        import time
        fingerprint = freeze(dataset, configs, args.output_dir)
        mask = dataset.folds[20260909] != 0
        rows = run_inner(configs, dataset.x[mask], dataset.y[mask], 20260909,
                         args.output_dir / "development_inner", fingerprint,
                         time.time() + args.hours * 3600, args.workers)
        write_json(args.output_dir / "screen_scores.json", [
            {"config": r["config"], "scores": r["scores"], "seconds": r["seconds"], "warnings": r["warnings"]}
            for r in rows])
        print(json.dumps({"status": "complete", "development_samples": int(mask.sum()), "candidates": len(rows)}))
    elif args.action == "evaluate":
        result = evaluate(dataset, configs, args.output_dir, workers=args.workers, hours=args.hours)
        print(json.dumps({k: v for k, v in result.items() if k != "paired_predictions"}, ensure_ascii=False))
    elif args.action == "fit":
        result = fit_final(dataset, configs, args.output_dir, workers=args.workers, hours=args.hours)
        print(json.dumps(result, ensure_ascii=False))
    elif args.action == "report":
        summary_path = args.output_dir / "evaluation_summary.json"
        if not summary_path.exists():
            p.error(f"missing {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines = ["# Hardware300 v3 纯电流结果", "", "模型只接收九路实测电流；没有独立测试集。", ""]
        repeat_accuracy = ", ".join(format(r["accuracy"], ".4%") for r in summary["repeats"])
        lines += [f"- 平均准确率：{summary['mean_accuracy']:.4%}",
                  f"- 平均宏 F1：{summary['mean_macro_f1']:.4%}",
                  f"- 平均宏召回率：{summary['mean_macro_recall']:.4%}",
                  f"- 重复准确率：{repeat_accuracy}",
                  f"- 外层预测数：{summary['prediction_count']}"]
        (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(str(args.output_dir / "report.md"))


if __name__ == "__main__":
    main()
