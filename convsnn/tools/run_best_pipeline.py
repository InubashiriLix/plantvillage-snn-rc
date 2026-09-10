#!/usr/bin/env python3
"""Resumable, validation-only hyperparameter search and final PlantVillage runs.

The runner is intentionally artifact-driven: each completed trial is recorded in
``state.json`` and is skipped after a restart.  High-load work can be constrained
to a local-time window without changing experiment semantics.
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from datetime import datetime, timedelta
import fcntl
from itertools import product
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from plantvillage_snn.artifacts import build_report, select_top10  # noqa: E402
from plantvillage_snn.config import ExperimentConfig  # noqa: E402
from plantvillage_snn.train import evaluate_saved_run, run_experiment  # noqa: E402


MODELS = ("conv_snn", "compact_snn")
COMPACT_WIDTHS = ((8, 16, 32), (12, 24, 48), (16, 32, 64))
LEARNING_RATES = (5e-4, 1e-3)
TIME_STEPS = (4, 6, 8)
BETAS = (0.85, 0.9)
AUGMENTATIONS = ("none", "basic", "strong")
BALANCES = ("none", "class_weights", "weighted_sampler")
HW_LEARNING_RATES = (0.02, 0.05, 0.08, 0.12)
HW_MAX_PULSES = (4, 8)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def safe_key(key: str) -> str:
    return key.replace("/", "__")


def summary_from_result(result: dict, config_path: Path, elapsed: float) -> dict:
    pulses = result.get("pulse_statistics") or {}
    return {
        "status": "complete",
        "config_path": str(config_path.resolve()),
        "metrics_path": result["metrics_path"],
        "checkpoint": result["checkpoint"],
        "selection_macro_recall": result["selection_metrics"]["macro_recall"],
        "selection_top1_accuracy": result["selection_metrics"]["top1_accuracy"],
        "final_macro_recall": (
            result["final_test_metrics"]["macro_recall"]
            if result.get("final_test_metrics") else None
        ),
        "final_top1_accuracy": (
            result["final_test_metrics"]["top1_accuracy"]
            if result.get("final_test_metrics") else None
        ),
        "total_pulses": pulses.get("total_pulses"),
        "total_updated_cells": pulses.get("total_updated_cells"),
        "logical_synapses": (result.get("model_cost") or {}).get("logical_synapses"),
        "physical_cells": (result.get("model_cost") or {}).get("differential_physical_cells"),
        "synaptic_events": result["selection_metrics"].get(
            "inference_cost_per_sample", {}
        ).get("spike_driven_synaptic_events_upper_bound"),
        "time_steps": result.get("config", {}).get("time_steps"),
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now().astimezone().isoformat(),
    }


def best_result(results: list[dict]) -> dict:
    if not results:
        raise RuntimeError("cannot choose a winner from an empty result set")
    return min(
        enumerate(results),
        key=lambda item: (
            -item[1]["selection_macro_recall"],
            -item[1]["selection_top1_accuracy"],
            item[0],
        ),
    )[1]


class BudgetReached(RuntimeError):
    pass


class PipelineRunner:
    def __init__(self, root: Path, wait_for_window: bool, start_hour: int,
                 end_hour: int, launch_guard_minutes: int, budget_hours: float,
                 ignore_window: bool = False):
        self.root = root.resolve()
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self.wait_for_window = wait_for_window
        self.ignore_window = ignore_window
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.launch_guard = launch_guard_minutes * 60
        self.budget_seconds = budget_hours * 3600
        if self.state_path.is_file():
            self.state = load_json(self.state_path)
        else:
            self.state = {
                "version": 1,
                "started_at": datetime.now().astimezone().isoformat(),
                "compute_seconds": 0.0,
                "tasks": {},
            }
            self._save_state()

    def _save_state(self) -> None:
        atomic_json(self.state_path, self.state)

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "time": datetime.now().astimezone().isoformat(),
            "event": event,
            **fields,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        print(json.dumps(record, ensure_ascii=False, default=str), flush=True)

    def _seconds_until_window(self, now: datetime) -> float:
        start = now.replace(hour=self.start_hour, minute=0, second=0, microsecond=0)
        if now.hour < self.start_hour:
            target = start
        else:
            target = start + timedelta(days=1)
        return max((target - now).total_seconds(), 0)

    def _window_remaining(self, now: datetime) -> float:
        if self.start_hour <= now.hour < self.end_hour:
            end = now.replace(hour=self.end_hour, minute=0, second=0, microsecond=0)
            return max((end - now).total_seconds(), 0)
        return 0

    def ensure_window(self, task: str) -> None:
        if self.state["compute_seconds"] >= self.budget_seconds:
            raise BudgetReached(
                f"compute budget reached before {task}: "
                f"{self.state['compute_seconds'] / 3600:.2f} hours"
            )
        if self.ignore_window:
            return
        announced = False
        while True:
            now = datetime.now().astimezone()
            remaining = self._window_remaining(now)
            if remaining > self.launch_guard:
                return
            if not self.wait_for_window:
                raise RuntimeError(
                    f"outside permitted {self.start_hour:02d}:00–{self.end_hour:02d}:00 "
                    f"window (or inside its launch guard) before {task}"
                )
            wait_seconds = self._seconds_until_window(now)
            if not announced:
                self.event(
                    "waiting_for_window", task=task,
                    resume_at=(now + timedelta(seconds=wait_seconds)).isoformat(),
                )
                announced = True
            time.sleep(min(max(wait_seconds, 1), 60))

    def completed(self, key: str) -> dict | None:
        summary = self.state["tasks"].get(key)
        if not summary or summary.get("status") != "complete":
            return None
        if not Path(summary["metrics_path"]).is_file() or not Path(summary["checkpoint"]).is_file():
            return None
        return summary

    def run_task(self, key: str, config: ExperimentConfig) -> dict:
        existing = self.completed(key)
        if existing:
            self.event("task_skipped", task=key, reason="already_complete")
            return existing
        config_path = self.root / "configs" / f"{safe_key(key)}.json"
        atomic_json(config_path, asdict(config))
        self.ensure_window(key)
        attempt = 0
        while True:
            attempt += 1
            self.event(
                "task_started", task=key, attempt=attempt,
                model=config.model, stage=config.stage, phase=config.phase,
                batch_size=config.batch_size,
            )
            started = time.monotonic()
            try:
                result = run_experiment(config)
                elapsed = time.monotonic() - started
                self.state["compute_seconds"] += elapsed
                summary = summary_from_result(result, config_path, elapsed)
                self.state["tasks"][key] = summary
                self._save_state()
                self.event(
                    "task_completed", task=key, elapsed_seconds=round(elapsed, 2),
                    macro_recall=summary["selection_macro_recall"],
                    top1_accuracy=summary["selection_top1_accuracy"],
                    pulses=summary["total_pulses"],
                )
                self.release_snn_cuda()
                return summary
            except RuntimeError as error:
                elapsed = time.monotonic() - started
                self.state["compute_seconds"] += elapsed
                self._save_state()
                is_oom = "out of memory" in str(error).lower()
                if is_oom and config.batch_size > 64 and attempt == 1:
                    self.event("cuda_oom_retry", task=key, old_batch=config.batch_size, new_batch=64)
                    config = config.with_overrides({"batch_size": 64})
                    atomic_json(config_path, asdict(config))
                    self.release_snn_cuda()
                    continue
                self.event("task_failed", task=key, error=repr(error))
                raise

    @staticmethod
    def release_snn_cuda() -> None:
        """Drop snnTorch's global neuron references between independent trials."""
        from snntorch._neurons.neurons import SpikingNeuron

        SpikingNeuron.instances.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def base_config(self, model: str, stage: str) -> ExperimentConfig:
        return ExperimentConfig(
            data_root=str((REPO_ROOT.parent / "data").resolve()),
            output_dir=str(self.root / "unused"),
            model=model,
            stage=stage,
            phase="ideal",
            seed=42,
            split_seed=42,
            selection_fraction=0.15,
            split_manifest_path=str(self.root / "splits/seed42_val0p15.json"),
            image_size=64,
            batch_size=128,
            num_workers=4,
            time_steps=8,
            beta=0.9,
            threshold=1.0,
            channels=[12, 24, 48] if model == "compact_snn" else None,
            learning_rate=1e-3,
            weight_decay=1e-4,
            epochs=5,
            patience=2,
            augmentation="basic",
            class_balance="class_weights",
            top10_path=str(self.root / "top10_classes.json"),
            evaluate_final=False,
            device_csv=str((REPO_ROOT / "cnnRef/source/data.csv").resolve()),
            device_v_bias=-5.0,
            device_g_bins=200,
            device_max_pulses=200,
            max_pulses_per_update=8,
            hw_scale_margin=1.5,
            accum_decay=0.9,
            accum_decay_interval=10,
            device="auto",
        )

    def limits(self, stage: str) -> tuple[int, int]:
        return (7600, 3800) if stage == "all38" else (5000, 2000)

    def ideal_search(self, model: str, stage: str,
                     teacher_checkpoint: str | None = None,
                     fixed_channels: tuple[int, int, int] | None = None) -> ExperimentConfig:
        train_limit, selection_limit = self.limits(stage)
        base = self.base_config(model, stage).with_overrides({
            "max_train_samples": train_limit,
            "max_selection_samples": selection_limit,
            "teacher_checkpoint": teacher_checkpoint,
        })
        core_results = []
        widths = (fixed_channels,) if model == "compact_snn" else (None,)
        if model == "compact_snn" and fixed_channels is None:
            raise ValueError("compact_snn search requires an explicit channel width")
        variant = (
            "w" + "_".join(str(value) for value in fixed_channels)
            if fixed_channels is not None else "baseline"
        )
        for index, (lr, steps, beta, channels) in enumerate(product(
            LEARNING_RATES, TIME_STEPS, BETAS, widths
        )):
            key = f"search/{stage}/{model}/{variant}/ideal/core/{index:03d}"
            config = base.with_overrides({
                "learning_rate": lr, "time_steps": steps, "beta": beta,
                "channels": list(channels) if channels is not None else None,
                "augmentation": "basic", "class_balance": "class_weights",
                "output_dir": str(self.root / "tasks" / safe_key(key)),
            })
            core_results.append(self.run_task(key, config))
        core_winner = best_result(core_results)
        winning_core_config = ExperimentConfig.from_json(core_winner["config_path"])

        regularization_results = []
        for index, (augmentation, balance) in enumerate(product(
            AUGMENTATIONS, BALANCES
        )):
            key = f"search/{stage}/{model}/{variant}/ideal/regularization/{index:03d}"
            config = winning_core_config.with_overrides({
                "augmentation": augmentation,
                "class_balance": balance,
                "output_dir": str(self.root / "tasks" / safe_key(key)),
            })
            regularization_results.append(self.run_task(key, config))
        winner = best_result(regularization_results)
        winning_config = ExperimentConfig.from_json(winner["config_path"])
        winner_path = self.root / "winners" / f"{model}_{stage}_{variant}_ideal.json"
        atomic_json(winner_path, asdict(winning_config))
        self.event("ideal_search_winner", model=model, stage=stage,
                   config_path=str(winner_path), macro_recall=winner["selection_macro_recall"])
        return winning_config

    def formal_ideal(self, model: str, stage: str, winner: ExperimentConfig,
                     variant: str = "baseline") -> dict:
        key = f"formal/{stage}/{model}/{variant}/ideal"
        config = winner.with_overrides({
            "output_dir": str(self.root / "final" / variant),
            "phase": "ideal", "checkpoint": None,
            "epochs": 20, "patience": 5,
            "max_train_samples": None, "max_selection_samples": None,
            "max_final_samples": None, "evaluate_final": False,
        })
        return self.run_task(key, config)

    def hardware_search(self, model: str, stage: str, ideal: dict,
                        variant: str = "baseline") -> ExperimentConfig:
        train_limit, selection_limit = self.limits(stage)
        ideal_config = ExperimentConfig.from_json(ideal["config_path"])
        results = []
        candidates = list(product(HW_LEARNING_RATES, HW_MAX_PULSES))
        for index, (learning_rate, max_pulses) in enumerate(candidates):
            key = f"search/{stage}/{model}/{variant}/hardware/{index:03d}"
            config = ideal_config.with_overrides({
                "output_dir": str(self.root / "tasks" / safe_key(key)),
                "phase": "hw_finetune", "checkpoint": ideal["checkpoint"],
                "learning_rate": learning_rate,
                "max_pulses_per_update": max_pulses,
                "epochs": 3, "patience": 2, "evaluate_final": False,
                "max_train_samples": train_limit,
                "max_selection_samples": selection_limit,
                "max_final_samples": None,
            })
            results.append(self.run_task(key, config))
        valid = [
            result for result in results
            if (result.get("total_pulses") or 0) > 0
            and (result.get("total_updated_cells") or 0) > 0
        ]
        if not valid:
            for index, max_pulses in enumerate(HW_MAX_PULSES):
                key = f"search/{stage}/{model}/{variant}/hardware/fallback_{index:03d}"
                config = ideal_config.with_overrides({
                    "output_dir": str(self.root / "tasks" / safe_key(key)),
                    "phase": "hw_finetune", "checkpoint": ideal["checkpoint"],
                    "learning_rate": 0.2,
                    "max_pulses_per_update": max_pulses,
                    "epochs": 3, "patience": 2, "evaluate_final": False,
                    "max_train_samples": train_limit,
                    "max_selection_samples": selection_limit,
                    "max_final_samples": None,
                })
                result = self.run_task(key, config)
                if (result.get("total_pulses") or 0) > 0 and (result.get("total_updated_cells") or 0) > 0:
                    valid.append(result)
        if not valid:
            raise RuntimeError(f"all hardware candidates produced zero updates for {model}/{stage}")
        winner = best_result(valid)
        winning_config = ExperimentConfig.from_json(winner["config_path"])
        winner_path = self.root / "winners" / f"{model}_{stage}_{variant}_hardware.json"
        atomic_json(winner_path, asdict(winning_config))
        self.event("hardware_search_winner", model=model, stage=stage,
                   config_path=str(winner_path), macro_recall=winner["selection_macro_recall"],
                   pulses=winner["total_pulses"])
        return winning_config

    def formal_hardware(self, model: str, stage: str, winner: ExperimentConfig,
                        ideal: dict, variant: str = "baseline") -> dict:
        key = f"formal/{stage}/{model}/{variant}/hardware"
        config = winner.with_overrides({
            "output_dir": str(self.root / "final" / variant),
            "checkpoint": ideal["checkpoint"],
            "epochs": 10, "patience": 3,
            "max_train_samples": None, "max_selection_samples": None,
            "max_final_samples": None, "evaluate_final": False,
        })
        result = self.run_task(key, config)
        if (result.get("total_pulses") or 0) <= 0 or (result.get("total_updated_cells") or 0) <= 0:
            raise RuntimeError(f"formal hardware run had zero device updates: {model}/{stage}")
        return result

    def make_top10(self, conv_hardware: dict) -> Path:
        output = self.root / "top10_classes.json"
        selected = select_top10(conv_hardware["metrics_path"], output)
        self.event("top10_created_or_refreshed", path=str(output),
                   classes=[item["name"] for item in selected["classes"]])
        return output

    def select_deployment(self, stage: str, candidates: list[dict],
                          tolerance: float = 0.02) -> dict:
        best_macro = max(item["selection_macro_recall"] for item in candidates)
        eligible = [
            item for item in candidates
            if item["selection_macro_recall"] >= best_macro - tolerance
        ]
        winner = min(
            eligible,
            key=lambda item: (
                item.get("physical_cells") or float("inf"),
                item.get("synaptic_events") or float("inf"),
                item.get("time_steps") or float("inf"),
                item.get("total_pulses") or float("inf"),
                load_json(Path(item["config_path"]))["model"],
            ),
        )
        model = load_json(Path(winner["config_path"]))["model"]
        decision = {
            "format_version": 1,
            "stage": stage,
            "accuracy_metric": "selection_macro_recall",
            "accuracy_tolerance_absolute": tolerance,
            "best_macro_recall": best_macro,
            "minimum_eligible_macro_recall": best_macro - tolerance,
            "winner_model": model,
            "winner": winner,
            "eligible": eligible,
            "all_candidates": candidates,
            "tie_break_order": [
                "differential_physical_cells", "synaptic_events", "time_steps",
                "write_pulses", "model_name",
            ],
        }
        atomic_json(self.root / "deployment" / f"{stage}.json", decision)
        self.event(
            "deployment_selected", stage=stage, model=model,
            macro_recall=winner["selection_macro_recall"],
            physical_cells=winner.get("physical_cells"),
        )
        return winner

    def run(self) -> None:
        self.state["status"] = "running"
        self.state.pop("last_error", None)
        self._save_state()
        self.event(
            "pipeline_started", cuda=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            compute_hours_so_far=round(self.state["compute_seconds"] / 3600, 3),
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to start the planned GPU pipeline")

        final_results: dict[tuple[str, str, str, str], dict] = {}

        def run_stage(stage: str) -> dict:
            baseline_winner = self.ideal_search("conv_snn", stage)
            baseline = self.formal_ideal("conv_snn", stage, baseline_winner, "baseline")
            final_results[("conv_snn", stage, "ideal", "baseline")] = baseline

            ideal_candidates = [("conv_snn", "baseline", baseline)]
            for channels in COMPACT_WIDTHS:
                variant = "w" + "_".join(str(value) for value in channels)
                compact_winner = self.ideal_search(
                    "compact_snn", stage, baseline["checkpoint"], channels
                )
                compact = self.formal_ideal(
                    "compact_snn", stage, compact_winner, variant
                )
                ideal_candidates.append(("compact_snn", variant, compact))
                final_results[("compact_snn", stage, "ideal", variant)] = compact

            hardware_candidates = []
            completed_candidates = []
            for model, variant, ideal in ideal_candidates:
                hw_winner = self.hardware_search(model, stage, ideal, variant)
                hardware = self.formal_hardware(
                    model, stage, hw_winner, ideal, variant
                )
                final_results[(model, stage, "hw_finetune", variant)] = hardware
                hardware_candidates.append(hardware)
                completed_candidates.append((model, variant, ideal, hardware))
            deployment = self.select_deployment(stage, hardware_candidates)
            selected_metrics = Path(deployment["metrics_path"]).resolve()
            retained = [
                item for item in completed_candidates
                if item[0] == "conv_snn"
                or Path(item[3]["metrics_path"]).resolve() == selected_metrics
            ]
            evaluated_paths = set()
            for model, variant, ideal, hardware in retained:
                for phase, summary in (("ideal", ideal), ("hw_finetune", hardware)):
                    metrics_path = str(Path(summary["metrics_path"]).resolve())
                    if metrics_path in evaluated_paths:
                        continue
                    evaluate_saved_run(metrics_path)
                    evaluated_paths.add(metrics_path)
                    self.event(
                        "final_test_evaluated", stage=stage, model=model,
                        variant=variant, phase=phase, metrics_path=metrics_path,
                    )
            return deployment

        all38_deployment = run_stage("all38")
        self.make_top10(all38_deployment)
        run_stage("top10")

        metrics = [result["metrics_path"] for result in final_results.values()]
        build_report(metrics, self.root / "report")
        self.state["status"] = "complete"
        self.state["completed_at"] = datetime.now().astimezone().isoformat()
        self._save_state()
        self.event(
            "pipeline_completed",
            compute_hours=round(self.state["compute_seconds"] / 3600, 3),
            report=str(self.root / "report/comparison.json"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/compact_pipeline")
    parser.add_argument("--wait-for-window", action="store_true")
    parser.add_argument("--ignore-window", action="store_true")
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--end-hour", type=int, default=8)
    parser.add_argument("--launch-guard-minutes", type=int, default=60)
    parser.add_argument("--budget-hours", type=float, default=12.0)
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = (REPO_ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    if args.status:
        state_path = root / "state.json"
        print(state_path.read_text() if state_path.is_file() else "no state")
        return 0
    root.mkdir(parents=True, exist_ok=True)
    lock_stream = (root / ".runner.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another pipeline runner already holds {root / '.runner.lock'}", file=sys.stderr)
        return 3
    lock_stream.write(str(os.getpid()) + "\n")
    lock_stream.flush()
    runner = PipelineRunner(
        root=root,
        wait_for_window=args.wait_for_window,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        launch_guard_minutes=args.launch_guard_minutes,
        budget_hours=args.budget_hours,
        ignore_window=args.ignore_window,
    )
    try:
        runner.run()
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
