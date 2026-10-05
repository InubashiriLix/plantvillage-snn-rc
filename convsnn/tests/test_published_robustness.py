"""Independently reconcile published noise trials, matrices and error bars."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/convsnn10_robustness"


def csv_rows(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_portable_original_weights_and_frozen_images():
    bundle = RESULTS / "model"
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == expected
    checkpoint = torch.load(bundle / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 3
    assert checkpoint["metadata"]["training_phase"] == "hw_finetune"
    assert checkpoint["hardware_state"]["mapping_mode"] == "differential_pair"
    records = json.loads((bundle / "test_manifest.json").read_text())
    assert len(records) == len({r["path"] for r in records}) == 4894
    assert sum(r["target"] == r["prediction"] for r in records) == 4593
    assert {r["target"] for r in records} == set(range(10))


def test_trials_matrices_means_and_sample_standard_deviations():
    protocol = json.loads((RESULTS / "protocol.json").read_text())
    assert protocol["status"] == "complete"
    assert protocol["baseline_prediction_mismatches"] == 0
    assert protocol["retrained"] is False
    for name, expected in protocol["files_sha256"].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == expected
    trials = csv_rows("repeat_metrics.csv")
    assert len(trials) == 26
    assert len({(r["sigma"], r["noise_seed"]) for r in trials}) == 26
    for row in trials:
        sigma, seed = float(row["sigma"]), int(row["noise_seed"])
        matrix = json.loads((RESULTS / f"confusion_sigma{sigma:g}_seed{seed}.json").read_text())["matrix"]
        assert sum(map(sum, matrix)) == int(row["sample_count"]) == 4894
        correct = sum(matrix[i][i] for i in range(10))
        assert correct == int(row["correct"])
        assert float(row["accuracy"]) == pytest.approx(correct / 4894, abs=1e-12)
        recall = statistics.mean(matrix[i][i] / sum(matrix[i]) for i in range(10))
        assert float(row["macro_recall"]) == pytest.approx(recall, abs=1e-12)
    for row in csv_rows("summary.csv"):
        sigma = float(row["sigma"])
        group = [r for r in trials if float(r["sigma"]) == sigma]
        expected_seeds = {100} if sigma == 0 else {100, 101, 102, 103, 104}
        assert {int(r["noise_seed"]) for r in group} == expected_seeds
        values = [float(r["accuracy"]) for r in group]
        assert float(row["accuracy_mean"]) == pytest.approx(statistics.mean(values), abs=1e-12)
        assert float(row["accuracy_std"]) == pytest.approx(statistics.stdev(values) if sigma else 0., abs=1e-12)
