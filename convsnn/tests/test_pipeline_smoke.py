from __future__ import annotations

from pathlib import Path

import pytest

from plantvillage_snn.artifacts import build_report, select_top10
from plantvillage_snn.config import ExperimentConfig
from plantvillage_snn.train import evaluate_saved_run, load_checkpoint, run_experiment


@pytest.mark.slow
def test_every_model_stage_phase_smoke(tiny_plantvillage, tmp_path):
    common = dict(
        data_root=str(tiny_plantvillage), output_dir=str(tmp_path / "runs"),
        seed=5, split_seed=5, selection_fraction=0.34,
        image_size=16, batch_size=38, time_steps=2, beta=0.8, threshold=0.1,
        epochs=1, patience=1, augmentation="none", class_balance="none",
        learning_rate=0.001, evaluate_final=True,
        device_csv=str(Path(__file__).resolve().parents[1] / "cnnRef/source/data.csv"), device_g_bins=12,
        device_max_pulses=12, max_pulses_per_update=2,
    )
    results = {}
    for model_type in ("conv_snn", "compact_snn"):
        model_common = {
            **common,
            "channels": [8, 16, 32] if model_type == "compact_snn" else None,
            "teacher_checkpoint": (
                results[("conv_snn", "all38", "ideal")]["checkpoint"]
                if model_type == "compact_snn" else None
            ),
        }
        ideal = run_experiment(ExperimentConfig(**model_common, model=model_type, stage="all38", phase="ideal"))
        results[(model_type, "all38", "ideal")] = ideal
        hw = run_experiment(ExperimentConfig(
            **model_common, model=model_type, stage="all38", phase="hw_finetune",
            checkpoint=ideal["checkpoint"],
        ))
        results[(model_type, "all38", "hw_finetune")] = hw

    top10_path = tmp_path / "top10_classes.json"
    select_top10(results[("conv_snn", "all38", "hw_finetune")]["metrics_path"], top10_path)
    top_common = {**common, "top10_path": str(top10_path)}
    for model_type in ("conv_snn", "compact_snn"):
        model_common = {
            **top_common,
            "channels": [8, 16, 32] if model_type == "compact_snn" else None,
            "teacher_checkpoint": (
                results[("conv_snn", "top10", "ideal")]["checkpoint"]
                if model_type == "compact_snn" else None
            ),
        }
        ideal = run_experiment(ExperimentConfig(**model_common, model=model_type, stage="top10", phase="ideal"))
        results[(model_type, "top10", "ideal")] = ideal
        hw = run_experiment(ExperimentConfig(
            **model_common, model=model_type, stage="top10", phase="hw_finetune",
            checkpoint=ideal["checkpoint"],
        ))
        results[(model_type, "top10", "hw_finetune")] = hw

    assert len(results) == 8
    for (model_type, stage, phase), result in results.items():
        assert result["selection_metrics"]["split"] == "selection"
        assert result["final_test_metrics"]["split"] == "final_test"
        assert Path(result["checkpoint"]).is_file()
        checkpoint = load_checkpoint(result["checkpoint"])
        assert checkpoint["metadata"]["architecture"] == model_type
        assert checkpoint["metadata"]["stage"] == stage
        assert checkpoint["metadata"]["training_phase"] == phase
        expected_classes = 38 if stage == "all38" else 10
        assert len(checkpoint["metadata"]["class_mapping"]) == expected_classes
        if phase == "hw_finetune":
            assert result["pulse_statistics"] is not None
            assert result["conductance_statistics"] is not None
            assert result["conductance_statistics"]["mapping_mode"] == "differential_pair"
            assert result["device_curve_sensitivity"]["official_preprocess"] == "raw"
            assert result["ideal_to_hardware_penalty"] is not None

    report_dir = tmp_path / "report"
    report = build_report([result["metrics_path"] for result in results.values()], report_dir)
    assert len(report["runs"]) == 8
    assert (report_dir / "comparison.md").is_file()
    assert (report_dir / "comparison.zh-CN.md").is_file()
    frozen = evaluate_saved_run(
        results[("compact_snn", "top10", "hw_finetune")]["metrics_path"]
    )
    assert frozen["final_evaluation"]["performed_after_selection"] is True
    assert frozen["final_evaluation"]["checkpoint_retrained"] is False
