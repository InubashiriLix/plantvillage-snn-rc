from __future__ import annotations

import csv

import torch
from torch import nn

from plantvillage_snn.recording import (
    GradientRecorder, dataset_inventory, parameter_summaries, update_ratios, write_csv,
)


def test_internal_summaries_and_updates():
    model = nn.Linear(3, 2)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    model(torch.ones(4, 3)).sum().backward()
    recorder = GradientRecorder()
    recorder.update(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)
    summaries = parameter_summaries(model)
    assert summaries["weight"]["count"] == 6
    assert recorder.compute()["weight"]["mean_l2"] > 0
    assert update_ratios(before, model)["weight"] > 0


def test_inventory_and_utf8_csv(tmp_path):
    (tmp_path / "类别").mkdir()
    (tmp_path / "类别" / "图像.bin").write_bytes(b"plant")
    rows, digest = dataset_inventory(tmp_path)
    assert len(rows) == 1 and len(digest) == 64 and len(rows[0]["sha256"]) == 64
    output = tmp_path / "result.csv"
    write_csv(output, [{"class": "类别", "value": 1}])
    with output.open(encoding="utf-8", newline="") as handle:
        assert next(csv.DictReader(handle))["class"] == "类别"
