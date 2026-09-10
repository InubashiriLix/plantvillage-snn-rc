#!/usr/bin/env python3
"""Reproduce the original end-to-end ConvSNN on all 38 and selected Top-10 classes."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from plantvillage_snn.artifacts import build_report  # noqa: E402
from plantvillage_snn.train import evaluate_saved_run  # noqa: E402
from tools.run_best_pipeline import (  # noqa: E402
    BudgetReached,
    PipelineRunner,
    atomic_json,
    load_json,
)


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def build_detailed_report(metrics: dict[tuple[str, str], str], output_dir: Path) -> dict:
    """Write overall and per-class results for both stages and phases."""
    artifacts = {
        key: load_json(Path(path)) for key, path in metrics.items()
    }
    result = {
        "format_version": 1,
        "architecture": "conv_snn",
        "class_accuracy_definition": "per-class recall",
        "final_test_used_for_selection": False,
        "runs": {},
    }
    lines = [
        "# PlantVillage ConvSNN 38 类与 Top-10 复现报告",
        "",
        "类别准确率按该类别召回率定义。所有超参数、早停和 Top-10 排序只使用选择验证集；最终测试在选择冻结后加载。",
        "",
        "## 总览",
        "",
        "| 任务 | 相位 | 选择集准确率 | 选择集宏召回率 | 最终准确率 | 最终宏召回率 | 差分物理单元 | 写入脉冲 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("all38", "top10"):
        for phase in ("ideal", "hw_finetune"):
            artifact = artifacts[(stage, phase)]
            selection = artifact["selection_metrics"]
            final = artifact.get("final_test_metrics")
            if not final:
                raise ValueError(f"missing frozen final evaluation for {stage}/{phase}")
            pulse = artifact.get("pulse_statistics") or {}
            cost = artifact.get("model_cost") or {}
            run_key = f"{stage}/{phase}"
            result["runs"][run_key] = {
                "metrics_path": str(Path(metrics[(stage, phase)]).resolve()),
                "selection": selection,
                "final_test": final,
                "model_cost": cost,
                "pulse_statistics": artifact.get("pulse_statistics"),
                "conductance_statistics": artifact.get("conductance_statistics"),
                "ideal_to_hardware_penalty": artifact.get("ideal_to_hardware_penalty"),
            }
            lines.append(
                f"| {stage} | {phase} | {_percent(selection['top1_accuracy'])} | "
                f"{_percent(selection['macro_recall'])} | {_percent(final['top1_accuracy'])} | "
                f"{_percent(final['macro_recall'])} | "
                f"{cost.get('differential_physical_cells', '—')} | "
                f"{pulse.get('total_pulses', '—')} |"
            )

    for stage, title in (("all38", "38 类逐类结果"), ("top10", "Top-10 逐类结果")):
        lines.extend([
            "", f"## {title}", "",
            "| 相位 | 类别 | 样本数 | 正确数 | 类别准确率/召回率 | 精确率 |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for phase in ("ideal", "hw_finetune"):
            per_class = artifacts[(stage, phase)]["final_test_metrics"]["per_class"]
            for name, values in sorted(per_class.items(), key=lambda item: item[1]["index"]):
                lines.append(
                    f"| {phase} | {name} | {values['samples']} | {values['correct']} | "
                    f"{_percent(values['recall'])} | {_percent(values['precision'])} |"
                )

    lines.extend([
        "",
        "## 解释限制",
        "",
        "器件 CSV 没有同步光刺激日志，因此仅作为经验电导更新曲线。报告不估算缺少实测依据的芯片能耗或面积。",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "convsnn_38_to_top10.json", result)
    (output_dir / "convsnn_38_to_top10.zh-CN.md").write_text("\n".join(lines) + "\n")
    return result


class ConvSNNReproduction(PipelineRunner):
    def run(self) -> None:
        self.state["status"] = "running"
        self.state.pop("last_error", None)
        self._save_state()
        self.event(
            "convsnn_reproduction_started",
            cuda=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            completed_tasks=len(self.state.get("tasks", {})),
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to start the ConvSNN reproduction")

        results: dict[tuple[str, str], dict] = {}
        all38_ideal_config = self.ideal_search("conv_snn", "all38")
        all38_ideal = self.formal_ideal("conv_snn", "all38", all38_ideal_config)
        results[("all38", "ideal")] = all38_ideal
        all38_hw_config = self.hardware_search("conv_snn", "all38", all38_ideal)
        all38_hw = self.formal_hardware("conv_snn", "all38", all38_hw_config, all38_ideal)
        results[("all38", "hw_finetune")] = all38_hw

        self.make_top10(all38_hw)

        top10_ideal_config = self.ideal_search("conv_snn", "top10")
        top10_ideal = self.formal_ideal("conv_snn", "top10", top10_ideal_config)
        results[("top10", "ideal")] = top10_ideal
        top10_hw_config = self.hardware_search("conv_snn", "top10", top10_ideal)
        top10_hw = self.formal_hardware("conv_snn", "top10", top10_hw_config, top10_ideal)
        results[("top10", "hw_finetune")] = top10_hw

        # No final-test data is loaded until every selection decision above is frozen.
        for (stage, phase), summary in results.items():
            evaluated = evaluate_saved_run(summary["metrics_path"])
            self.event(
                "final_test_evaluated", stage=stage, phase=phase,
                top1=eventual(evaluated, "top1_accuracy"),
                macro_recall=eventual(evaluated, "macro_recall"),
            )

        metrics_paths = [summary["metrics_path"] for summary in results.values()]
        build_report(metrics_paths, self.root / "report")
        build_detailed_report(
            {key: summary["metrics_path"] for key, summary in results.items()},
            self.root / "report",
        )
        self.state["status"] = "complete"
        self.state["completed_at"] = datetime.now().astimezone().isoformat()
        self._save_state()
        self.event(
            "convsnn_reproduction_completed",
            report=str(self.root / "report/convsnn_38_to_top10.zh-CN.md"),
        )


def eventual(artifact: dict, key: str) -> float:
    return artifact["final_test_metrics"][key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/convsnn_reproduction")
    parser.add_argument("--wait-for-window", action="store_true")
    parser.add_argument("--ignore-window", action="store_true")
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--end-hour", type=int, default=8)
    parser.add_argument("--launch-guard-minutes", type=int, default=60)
    parser.add_argument("--budget-hours", type=float, default=48.0)
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = (REPO_ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    if args.status:
        state_path = root / "state.json"
        if not state_path.is_file():
            print("no state")
            return 0
        state = load_json(state_path)
        print(json.dumps({
            "status": state.get("status"),
            "completed_tasks": len(state.get("tasks", {})),
            "compute_hours": float(state.get("compute_seconds", 0)) / 3600,
            "last_error": state.get("last_error"),
            "completed_at": state.get("completed_at"),
        }, indent=2, ensure_ascii=False))
        return 0

    root.mkdir(parents=True, exist_ok=True)
    lock_stream = (root / ".runner.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another runner holds {root / '.runner.lock'}", file=sys.stderr)
        return 3
    lock_stream.write(str(os.getpid()) + "\n")
    lock_stream.flush()
    runner = ConvSNNReproduction(
        root=root,
        wait_for_window=args.wait_for_window,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        launch_guard_minutes=args.launch_guard_minutes,
        budget_hours=args.budget_hours,
        ignore_window=args.ignore_window,
    )

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        runner.run()
    except KeyboardInterrupt:
        runner.state["status"] = "interrupted"
        runner.state["interrupted_at"] = datetime.now().astimezone().isoformat()
        runner._save_state()
        runner.event("convsnn_reproduction_interrupted")
        return 130
    except BudgetReached as error:
        runner.state["status"] = "budget_reached"
        runner._save_state()
        runner.event("pipeline_paused", reason=str(error))
        return 2
    except Exception as error:
        runner.state["status"] = "failed"
        runner.state["last_error"] = repr(error)
        runner._save_state()
        runner.event("pipeline_failed", error=repr(error))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
