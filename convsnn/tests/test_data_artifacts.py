from __future__ import annotations

import json
from pathlib import Path

from plantvillage_snn.artifacts import select_top10
from plantvillage_snn.config import ExperimentConfig
from plantvillage_snn.data import build_dataloaders, create_split_manifest, load_stage_mapping
from plantvillage_snn.search import focused_search
from tools.run_convsnn_reproduction import build_detailed_report


def test_stratification_mapping_and_final_exclusion(tiny_plantvillage, tmp_path):
    manifest_path = tmp_path / "split.json"
    first = create_split_manifest(tiny_plantvillage, manifest_path, 9, 0.34)
    second = create_split_manifest(tiny_plantvillage, tmp_path / "split2.json", 9, 0.34)
    assert first["partitions"] == second["partitions"]
    assert len(first["class_to_original_index"]) == 38
    for partition in ("train", "selection"):
        assert {r["class_name"] for r in first["partitions"][partition]} == set(first["class_to_original_index"])
        assert all(r["path"].startswith("train/") for r in first["partitions"][partition])
        assert not any(r["path"].startswith("val/") for r in first["partitions"][partition])
    assert first["final_test_used_for_selection"] is False


def test_top10_ranking_ties_and_label_remap(tiny_plantvillage, tmp_path):
    names = [f"class_{i:02d}" for i in range(38)]
    original = {name: i for i, name in enumerate(names)}
    per_class = {
        name: {"recall": (i % 5) / 4, "precision": (37 - i) / 37, "samples": 1, "correct": 0, "index": i}
        for i, name in enumerate(names)
    }
    metrics = {
        "run": {"model": "conv_snn", "stage": "all38", "phase": "hw_finetune", "seed": 1},
        "metadata": {"class_to_original_index": original},
        "selection_metrics": {"split": "selection", "per_class": per_class},
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    top_path = tmp_path / "top10_classes.json"
    selected = select_top10(metrics_path, top_path)
    expected = sorted(names, key=lambda n: (-per_class[n]["recall"], -per_class[n]["precision"], n))[:10]
    assert [item["name"] for item in selected["classes"]] == expected
    config = ExperimentConfig(data_root=str(tiny_plantvillage), stage="top10", top10_path=str(top_path))
    mapping = load_stage_mapping(config, original)
    assert mapping == {name: i for i, name in enumerate(expected)}


def test_top10_accepts_selected_compact_pure_snn(tmp_path):
    names = [f"class_{i:02d}" for i in range(38)]
    artifact = {
        "run": {"model": "compact_snn", "stage": "all38", "phase": "hw_finetune", "seed": 3},
        "metadata": {"class_to_original_index": {name: i for i, name in enumerate(names)}},
        "selection_metrics": {
            "split": "selection",
            "per_class": {
                name: {"recall": i / 38, "precision": i / 38, "samples": 1, "correct": 0, "index": i}
                for i, name in enumerate(names)
            },
        },
    }
    source = tmp_path / "compact_metrics.json"
    source.write_text(json.dumps(artifact))
    selected = select_top10(source, tmp_path / "top10.json")
    assert selected["source_model"] == "compact_snn"


def test_final_loader_is_opt_in(tiny_plantvillage, tmp_path):
    base = ExperimentConfig(data_root=str(tiny_plantvillage), output_dir=str(tmp_path),
                            image_size=16, batch_size=8, augmentation="none")
    without_final = build_dataloaders(base)
    assert without_final.final_test is None
    images, _ = next(iter(without_final.selection))
    assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0
    with_final = build_dataloaders(base.with_overrides({"evaluate_final": True}))
    assert with_final.final_test is not None
    assert sum(with_final.sample_counts["final_test"]) == 38


def test_explicit_none_override_clears_search_sample_limits():
    search = ExperimentConfig(max_train_samples=100, max_selection_samples=50)
    formal = search.with_overrides({
        "max_train_samples": None, "max_selection_samples": None,
    })
    assert formal.max_train_samples is None
    assert formal.max_selection_samples is None


def test_explicit_split_manifest_is_shared_across_output_directories(tiny_plantvillage, tmp_path):
    shared = tmp_path / "shared" / "split.json"
    first = ExperimentConfig(
        data_root=str(tiny_plantvillage), output_dir=str(tmp_path / "trial_a"),
        split_manifest_path=str(shared), image_size=16, batch_size=8,
        augmentation="none",
    )
    second = first.with_overrides({"output_dir": str(tmp_path / "trial_b")})
    first_bundle = build_dataloaders(first)
    second_bundle = build_dataloaders(second)
    assert shared.is_file()
    assert first_bundle.split_manifest_path == second_bundle.split_manifest_path
    assert first_bundle.sample_counts == second_bundle.sample_counts


def test_focused_search_forces_final_test_off(monkeypatch, tmp_path):
    observed = []
    def fake_run(config):
        observed.append(config)
        return {
            "selection_metrics": {"macro_recall": config.beta, "top1_accuracy": 0.5},
            "checkpoint": "best.pt", "metrics_path": "metrics.json",
        }
    monkeypatch.setattr("plantvillage_snn.search.run_experiment", fake_run)
    base = ExperimentConfig(output_dir=str(tmp_path), evaluate_final=True)
    result = focused_search(base, [0.001], [2], [0.8, 0.9], ["none"], ["none"])
    assert all(config.evaluate_final is False for config in observed)
    assert result["final_test_used"] is False
    assert result["winner"]["config"]["beta"] == 0.9
    assert Path(result["winning_config_path"]).is_file()


def test_convsnn_detailed_report_contains_all38_and_top10_classes(tmp_path):
    paths = {}
    for stage, count in (("all38", 38), ("top10", 10)):
        for phase in ("ideal", "hw_finetune"):
            per_class = {
                f"class_{index:02d}": {
                    "index": index, "samples": 2, "correct": 1,
                    "recall": 0.5, "precision": 0.5,
                }
                for index in range(count)
            }
            artifact = {
                "selection_metrics": {"top1_accuracy": 0.5, "macro_recall": 0.5},
                "final_test_metrics": {
                    "top1_accuracy": 0.5, "macro_recall": 0.5,
                    "per_class": per_class,
                },
                "model_cost": {"differential_physical_cells": 100},
                "pulse_statistics": {"total_pulses": 10} if phase == "hw_finetune" else None,
                "conductance_statistics": None,
                "ideal_to_hardware_penalty": None,
            }
            path = tmp_path / f"{stage}_{phase}.json"
            path.write_text(json.dumps(artifact))
            paths[(stage, phase)] = str(path)
    report = build_detailed_report(paths, tmp_path / "report")
    markdown = (tmp_path / "report/convsnn_38_to_top10.zh-CN.md").read_text()
    assert report["class_accuracy_definition"] == "per-class recall"
    assert "class_37" in markdown and "class_09" in markdown
