"""Guard published plotting tables against split/order/normalization mistakes."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[2] / "results/convsnn10"


def table(name):
    with (BASE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("phase", ["ideal", "hw_finetune"])
@pytest.mark.parametrize("split", ["selection", "final_test"])
def test_published_confusion_and_class_metrics(phase, split):
    source = json.loads((BASE / "metrics.json").read_text())[phase][f"{split}_metrics"]
    rows = [r for r in table("confusion_matrix_long.csv") if r["phase"] == phase and r["split"] == split]
    assert len(rows) == 100
    assert len({(r["true_index"], r["predicted_index"]) for r in rows}) == 100
    matrix = [[0] * 10 for _ in range(10)]
    for row in rows:
        i, j = int(row["true_index"]), int(row["predicted_index"])
        matrix[i][j] = int(row["count"])
        assert float(row["row_fraction"]) == pytest.approx(int(row["count"]) / sum(source["confusion_matrix"][i]))
    assert matrix == source["confusion_matrix"]
    assert sum(map(sum, matrix)) == source["sample_count"]
    summary = next(r for r in table("summary.csv") if r["phase"] == phase and r["split"] == split)
    assert float(summary["accuracy"]) == pytest.approx(sum(matrix[i][i] for i in range(10)) / source["sample_count"])
    assert float(summary["macro_recall"]) == pytest.approx(sum(matrix[i][i] / sum(matrix[i]) for i in range(10)) / 10)
    classes = [r for r in table("per_class_metrics.csv") if r["phase"] == phase and r["split"] == split]
    assert len(classes) == 10
    for row in classes:
        expected = source["per_class"][row["class_name"]]
        for key in ("index", "samples", "correct"):
            assert int(row[key]) == expected[key]
        for key in ("precision", "recall"):
            assert float(row[key]) == pytest.approx(expected[key])


def test_provenance_and_full_training_curves():
    manifest = json.loads((BASE / "provenance.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest
    metrics = json.loads((BASE / "metrics.json").read_text())
    for phase, result in metrics.items():
        epochs = [r for r in table("epoch_metrics.csv") if r["phase"] == phase]
        assert [int(r["epoch"]) for r in epochs] == list(range(1, result["executed_epochs"] + 1))
        assert [int(r["epoch"]) for r in epochs if r["is_best_epoch"] == "True"] == [result["best_epoch"]]
