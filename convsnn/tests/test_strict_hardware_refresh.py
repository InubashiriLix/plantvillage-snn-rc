from __future__ import annotations

import json

import pytest

from tools.run_strict_hardware_refresh import (
    best_hardware_result, compatible_top10_config, same_top10,
)


def top10(names):
    return {"classes": [{"name": name, "new_index": index} for index, name in enumerate(names)]}


def test_top10_identity_includes_order():
    names = [f"class_{index}" for index in range(10)]
    assert same_top10(top10(names), top10(names))
    assert not same_top10(top10(names), top10(list(reversed(names))))


def test_retained_top10_config_must_match_training_protocol():
    all38 = {
        "model": "conv_snn", "stage": "all38", "phase": "ideal", "checkpoint": None,
        "seed": 42, "split_seed": 42, "selection_fraction": 0.15,
        "split_manifest_path": "split.json", "image_size": 64, "time_steps": 8,
        "beta": 0.85, "threshold": 1.0, "input_encoding": "direct_current",
        "learning_rate": 0.001, "weight_decay": 0.0001,
        "augmentation": "none", "class_balance": "none",
    }
    retained = {**all38, "stage": "top10"}
    assert compatible_top10_config(all38, retained)
    assert not compatible_top10_config(all38, {**retained, "beta": 0.9})
    assert not compatible_top10_config(all38, {**retained, "checkpoint": "old.pt"})


def test_hardware_winner_and_zero_update_exclusion(tmp_path):
    def candidate(name, macro, accuracy, pulses, updated, lr, max_pulses):
        config = tmp_path / f"{name}.json"
        config.write_text(json.dumps({
            "learning_rate": lr, "max_pulses_per_update": max_pulses,
        }))
        return {
            "config_path": str(config), "selection_macro_recall": macro,
            "selection_top1_accuracy": accuracy, "total_pulses": pulses,
            "total_updated_cells": updated,
        }

    zero = candidate("zero", 1.0, 1.0, 0, 0, 0.01, 4)
    more_pulses = candidate("more", 0.9, 0.8, 200, 2, 0.02, 4)
    fewer_pulses = candidate("fewer", 0.9, 0.8, 100, 2, 0.05, 8)
    assert best_hardware_result([zero, more_pulses, fewer_pulses]) == fewer_pulses
    with pytest.raises(RuntimeError, match="zero device updates"):
        best_hardware_result([zero])
