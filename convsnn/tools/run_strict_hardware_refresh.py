#!/usr/bin/env python3
"""Refresh hardware-aware ConvSNN results from the retained 100-epoch all38 model.

This runner never retrains the all38 ideal source. It performs validation-only
hardware search, creates a new hardware-ranked Top-10, conditionally reuses an
exactly matching fresh Top-10 ideal run, and delays final-test evaluation until
all selection decisions are frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
from itertools import product
import json
import os
from pathlib import Path
import signal
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from plantvillage_snn.artifacts import build_report, select_top10  # noqa: E402
from plantvillage_snn.config import ExperimentConfig  # noqa: E402
from plantvillage_snn.train import evaluate_saved_run, load_checkpoint  # noqa: E402
from tools.run_best_pipeline import (  # noqa: E402
    BudgetReached, PipelineRunner, atomic_json, load_json, safe_key,
)
from tools.run_convsnn_reproduction import build_detailed_report  # noqa: E402


ALL38_IDEAL_METRICS = REPO_ROOT / "artifacts/current_sources/all38_ideal_100ep/metrics.json"
CURRENT_TOP10_IDEAL_METRICS = REPO_ROOT / "artifacts/hardware_refresh_2026-08-06/final/conv_snn_top10_ideal_seed42/metrics.json"
HW_LEARNING_RATES = (0.01, 0.02, 0.05, 0.08)
HW_MAX_PULSES = (4, 8)


def ordered_top10(artifact: dict) -> list[str]:
    return [item["name"] for item in sorted(artifact["classes"], key=lambda item: item["new_index"])]


def same_top10(left: dict, right: dict) -> bool:
    return ordered_top10(left) == ordered_top10(right)


def compatible_top10_config(all38: dict, top10: dict) -> bool:
    shared = (
        "model", "seed", "split_seed", "selection_fraction", "split_manifest_path",
        "image_size", "time_steps", "beta", "threshold", "input_encoding",
        "learning_rate", "weight_decay", "augmentation", "class_balance",
    )
    return (
        all38.get("stage") == "all38"
        and top10.get("stage") == "top10"
        and top10.get("phase") == "ideal"
        and top10.get("checkpoint") is None
        and all(all38.get(key) == top10.get(key) for key in shared)
    )


def best_hardware_result(results: list[dict]) -> dict:
    valid = [
        result for result in results
        if (result.get("total_pulses") or 0) > 0
        and (result.get("total_updated_cells") or 0) > 0
    ]
    if not valid:
        raise RuntimeError("all hardware candidates produced zero device updates")
    return min(valid, key=lambda result: (
        -result["selection_macro_recall"],
        -result["selection_top1_accuracy"],
        result["total_pulses"],
        load_json(Path(result["config_path"]))["learning_rate"],
        load_json(Path(result["config_path"]))["max_pulses_per_update"],
    ))


class StrictHardwareRefresh(PipelineRunner):
    def completed(self, key: str) -> dict | None:
        summary = super().completed(key)
        if summary is None:
            return None
        try:
            artifact = load_json(Path(summary["metrics_path"]))
            checkpoint = load_checkpoint(summary["checkpoint"])
            run = artifact["run"]
            metadata = checkpoint["metadata"]
            if (
                metadata["architecture"] != run["model"]
                or metadata["stage"] != run["stage"]
                or metadata["training_phase"] != run["phase"]
                or metadata["class_mapping"] != artifact["metadata"]["class_mapping"]
            ):
                return None
            if run["phase"] == "hw_finetune" and not checkpoint.get("hardware_state"):
                return None
            if run["stage"] == "top10":
                selected = load_json(self.root / "top10_classes.json")
                expected_mapping = {
                    item["name"]: item["new_index"] for item in selected["classes"]
                }
                if artifact["metadata"]["class_mapping"] != expected_mapping:
                    return None
        except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError):
            return None
        return summary

    def _adopt_ideal(self, metrics_path: Path, expected_stage: str,
                     top10_path: Path | None = None) -> dict:
        artifact = load_json(metrics_path)
        run = artifact["run"]
        if (run.get("model"), run.get("stage"), run.get("phase")) != (
            "conv_snn", expected_stage, "ideal"
        ):
            raise ValueError(f"invalid retained ideal source: {metrics_path}")
        checkpoint_path = Path(artifact["checkpoint"])
        checkpoint = load_checkpoint(checkpoint_path)
        metadata = checkpoint.get("metadata", {})
        if (
            metadata.get("architecture") != "conv_snn"
            or metadata.get("stage") != expected_stage
            or metadata.get("training_phase") != "ideal"
            or metadata.get("class_mapping") != artifact["metadata"]["class_mapping"]
        ):
            raise ValueError(f"retained checkpoint does not match metrics: {checkpoint_path}")
        config = ExperimentConfig(**artifact["config"])
        overrides = {
            "evaluate_final": False,
            "max_train_samples": None,
            "max_selection_samples": None,
            "max_final_samples": None,
            "recording_dir": str((self.root / "retained" / expected_stage / "records").resolve()),
        }
        if top10_path is not None:
            overrides["top10_path"] = str(top10_path.resolve())
        config = config.with_overrides(overrides)
        config_path = self.root / "inputs" / f"retained_{expected_stage}_ideal.json"
        atomic_json(config_path, asdict(config))
        retained_metrics = self.root / "retained" / expected_stage / "metrics.json"
        artifact["config"] = config.to_dict()
        atomic_json(retained_metrics, artifact)
        return {
            "status": "retained",
            "config_path": str(config_path.resolve()),
            "metrics_path": str(retained_metrics.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "selection_macro_recall": artifact["selection_metrics"]["macro_recall"],
            "selection_top1_accuracy": artifact["selection_metrics"]["top1_accuracy"],
        }

    def hardware_search_strict(self, stage: str, ideal: dict) -> ExperimentConfig:
        train_limit, selection_limit = self.limits(stage)
        ideal_config = ExperimentConfig.from_json(ideal["config_path"])
        results = []
        for index, (learning_rate, max_pulses) in enumerate(product(
            HW_LEARNING_RATES, HW_MAX_PULSES
        )):
            key = f"search/{stage}/conv_snn/hardware/{index:03d}"
            config = ideal_config.with_overrides({
                "output_dir": str(self.root / "tasks" / safe_key(key)),
                "phase": "hw_finetune",
                "checkpoint": ideal["checkpoint"],
                "learning_rate": learning_rate,
                "max_pulses_per_update": max_pulses,
                "epochs": 3,
                "patience": 2,
                "evaluate_final": False,
                "max_train_samples": train_limit,
                "max_selection_samples": selection_limit,
                "max_final_samples": None,
                "recording_enabled": False,
                "record_internal_summaries": False,
                "record_sample_predictions": False,
                "dataset_hash_mode": "none",
                "recording_dir": None,
            })
            results.append(self.run_task(key, config))
        winner = best_hardware_result(results)
        winning_config = ExperimentConfig.from_json(winner["config_path"])
        winner_path = self.root / "winners" / f"conv_snn_{stage}_hardware.json"
        atomic_json(winner_path, asdict(winning_config))
        self.event(
            "hardware_search_winner", stage=stage,
            macro_recall=winner["selection_macro_recall"],
            top1_accuracy=winner["selection_top1_accuracy"],
            pulses=winner["total_pulses"], config_path=str(winner_path),
        )
        return winning_config

    def formal_hardware_strict(self, stage: str, winner: ExperimentConfig,
                               ideal: dict) -> dict:
        key = f"formal/{stage}/conv_snn/hardware"
        config = winner.with_overrides({
            "output_dir": str(self.root / "final"),
            "checkpoint": ideal["checkpoint"],
            "epochs": 10,
            "patience": 3,
            "max_train_samples": None,
            "max_selection_samples": None,
            "max_final_samples": None,
            "evaluate_final": False,
            "recording_enabled": True,
            "record_internal_summaries": True,
            "record_sample_predictions": True,
            "dataset_hash_mode": "sha256",
            "recording_dir": None,
        })
        result = self.run_task(key, config)
        if (result.get("total_pulses") or 0) <= 0 or (result.get("total_updated_cells") or 0) <= 0:
            raise RuntimeError(f"formal hardware run had zero device updates: {stage}")
        return result

    def fresh_top10_ideal(self, all38_config: ExperimentConfig) -> dict:
        key = "formal/top10/conv_snn/ideal"
        config = all38_config.with_overrides({
            "output_dir": str(self.root / "final"),
            "stage": "top10",
            "phase": "ideal",
            "checkpoint": None,
            "top10_path": str((self.root / "top10_classes.json").resolve()),
            "epochs": 100,
            "patience": 20,
            "max_train_samples": None,
            "max_selection_samples": None,
            "max_final_samples": None,
            "evaluate_final": False,
            "recording_enabled": True,
            "record_internal_summaries": True,
            "record_sample_predictions": True,
            "dataset_hash_mode": "sha256",
            "recording_dir": None,
        })
        return self.run_task(key, config)

    def _top10_ideal(self, all38_ideal: dict, selected: dict) -> dict:
        retained_path = CURRENT_TOP10_IDEAL_METRICS.resolve()
        all38_config = ExperimentConfig.from_json(all38_ideal["config_path"])
        if retained_path.is_file():
            retained_artifact = load_json(retained_path)
            retained_list_path = Path(retained_artifact["config"]["top10_path"])
            retained_list = load_json(retained_list_path) if retained_list_path.is_file() else None
            same_classes = retained_list is not None and same_top10(selected, retained_list)
            same_config = compatible_top10_config(
                all38_config.to_dict(), retained_artifact["config"]
            )
            if same_classes and same_config:
                self.event("top10_ideal_reused", metrics_path=str(retained_path))
                return self._adopt_ideal(retained_path, "top10", self.root / "top10_classes.json")
            self.event(
                "top10_fresh_training_required",
                reason="class_list_changed" if not same_classes else "configuration_mismatch",
                old=ordered_top10(retained_list) if retained_list else None,
                new=ordered_top10(selected),
            )
        else:
            self.event("top10_fresh_training_required", reason="retained_run_missing")
        return self.fresh_top10_ideal(all38_config)

    def run(self) -> None:
        self.state["status"] = "running"
        self.state.pop("last_error", None)
        self._save_state()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to start hardware refresh")
        self.event("strict_hardware_refresh_started", gpu=torch.cuda.get_device_name(0))

        all38_ideal = self._adopt_ideal(ALL38_IDEAL_METRICS.resolve(), "all38")
        all38_winner = self.hardware_search_strict("all38", all38_ideal)
        all38_hardware = self.formal_hardware_strict("all38", all38_winner, all38_ideal)

        top10_path = self.root / "top10_classes.json"
        selected = select_top10(all38_hardware["metrics_path"], top10_path)
        self.event("top10_created", classes=ordered_top10(selected), path=str(top10_path))
        top10_ideal = self._top10_ideal(all38_ideal, selected)
        top10_winner = self.hardware_search_strict("top10", top10_ideal)
        top10_hardware = self.formal_hardware_strict("top10", top10_winner, top10_ideal)

        # No final-test access occurs above this point.
        results = {
            ("all38", "ideal"): all38_ideal,
            ("all38", "hw_finetune"): all38_hardware,
            ("top10", "ideal"): top10_ideal,
            ("top10", "hw_finetune"): top10_hardware,
        }
        for (stage, phase), summary in results.items():
            artifact = load_json(Path(summary["metrics_path"]))
            if not artifact.get("final_test_metrics"):
                evaluate_saved_run(summary["metrics_path"])
                self.event("final_test_evaluated", stage=stage, phase=phase)

        metrics = {key: value["metrics_path"] for key, value in results.items()}
        build_report(list(metrics.values()), self.root / "report")
        build_detailed_report(metrics, self.root / "report")
        self.state["status"] = "complete"
        self.state["completed_at"] = datetime.now().astimezone().isoformat()
        self._save_state()
        self.event("strict_hardware_refresh_completed", report=str(self.root / "report"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/hardware_refresh_2026-08-06")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--budget-hours", type=float, default=72.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    root = (REPO_ROOT / root).resolve() if not root.is_absolute() else root.resolve()
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
    runner = StrictHardwareRefresh(
        root=root, wait_for_window=False, start_hour=0, end_hour=24,
        launch_guard_minutes=0, budget_hours=args.budget_hours,
        ignore_window=True,
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
        runner.event("strict_hardware_refresh_interrupted")
        return 130
    except BudgetReached as error:
        runner.state["status"] = "budget_reached"
        runner.state["last_error"] = str(error)
        runner._save_state()
        return 2
    except Exception as error:
        runner.state["status"] = "failed"
        runner.state["last_error"] = repr(error)
        runner._save_state()
        runner.event("strict_hardware_refresh_failed", error=repr(error))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
