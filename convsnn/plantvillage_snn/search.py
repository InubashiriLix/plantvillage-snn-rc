from __future__ import annotations

from itertools import product
from pathlib import Path

from .config import ExperimentConfig
from .train import run_experiment
from .utils import write_json


def focused_search(base: ExperimentConfig, learning_rates: list[float],
                   time_steps: list[int], betas: list[float],
                   augmentations: list[str], balances: list[str]) -> dict:
    trials = []
    root = Path(base.output_dir) / f"search_{base.model}_{base.stage}_{base.phase}_seed{base.seed}"
    for index, (lr, steps, beta, augmentation, balance) in enumerate(product(
        learning_rates, time_steps, betas, augmentations, balances
    )):
        config = base.with_overrides({
            "learning_rate": lr,
            "time_steps": steps,
            "beta": beta,
            "augmentation": augmentation,
            "class_balance": balance,
            "evaluate_final": False,  # final-test data is categorically excluded from search
            "output_dir": str(root / f"trial_{index:03d}"),
        })
        result = run_experiment(config)
        trials.append({
            "trial": index,
            "config": config.to_dict(),
            "selection_macro_recall": result["selection_metrics"]["macro_recall"],
            "selection_top1_accuracy": result["selection_metrics"]["top1_accuracy"],
            "checkpoint": result["checkpoint"],
            "metrics_path": result["metrics_path"],
        })
    winner = min(
        trials,
        key=lambda t: (-t["selection_macro_recall"], -t["selection_top1_accuracy"], t["trial"]),
    )
    artifact = {
        "format_version": 1,
        "selection_split": "selection",
        "final_test_used": False,
        "objective": "macro_recall",
        "winner": winner,
        "trials": trials,
    }
    write_json(root / "search_results.json", artifact)
    write_json(root / "winning_config.json", winner["config"])
    artifact["results_path"] = str((root / "search_results.json").resolve())
    artifact["winning_config_path"] = str((root / "winning_config.json").resolve())
    return artifact
