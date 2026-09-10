from __future__ import annotations

import torch
import json
from snntorch._neurons.neurons import SpikingNeuron

from plantvillage_snn.metrics import hardware_aware_snn_loss, spike_count_loss
from plantvillage_snn.models import build_model
from tools.run_best_pipeline import PipelineRunner


def test_model_shapes_temporal_state_gradients_and_activity():
    torch.manual_seed(3)
    images = torch.randn(2, 3, 16, 16)
    targets = torch.tensor([0, 4])
    for kind in ("conv_snn", "compact_snn"):
        model = build_model(
            kind, 5, time_steps=4, beta=0.8, threshold=0.2,
            channels=[8, 16, 32] if kind == "compact_snn" else None,
        )
        model.eval()
        first = model(images)
        model.reset_state()
        second = model(images)
        assert first["spikes"].shape == (4, 2, 5)
        assert first["spike_counts"].shape == (2, 5)
        assert torch.equal(first["spikes"], second["spikes"])
        assert sum(float(spikes.detach().sum()) for spikes in first["activity"].values()) > 0
        loss = spike_count_loss(first, targets)
        loss.backward()
        synapse_gradients = [
            parameter.grad for name, parameter in model.named_parameters()
            if ("conv" in name or "head" in name or "fc" in name or "output" in name)
            and name.endswith("weight")
        ]
        assert synapse_gradients and all(g is not None for g in synapse_gradients)
        assert all(torch.isfinite(g).all() for g in synapse_gradients)


def test_compact_snn_removes_large_hidden_fc_and_supports_hybrid_loss():
    compact = build_model("compact_snn", 38, time_steps=2, channels=[8, 16, 32])
    baseline = build_model("conv_snn", 38, time_steps=2)
    assert compact.synapse_cost()["logical_synapses"] < baseline.synapse_cost()["logical_synapses"] / 10
    assert compact.synapse_cost()["differential_physical_cells"] == 2 * compact.synapse_cost()["logical_synapses"]
    images = torch.randn(2, 3, 16, 16)
    result = compact(images)
    loss, parts = hardware_aware_snn_loss(
        result, torch.tensor([1, 2]), membrane_weight=0.25,
        teacher_result=result, distill_weight=0.5,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {"spike", "membrane", "distillation", "firing_regularization"}


def test_campaign_cleanup_releases_snntorch_global_instances():
    build_model("conv_snn", 5, time_steps=2)
    assert SpikingNeuron.instances
    PipelineRunner.release_snn_cuda()
    assert SpikingNeuron.instances == []


def test_deployment_selector_prefers_smallest_model_within_two_points(tmp_path):
    runner = PipelineRunner(tmp_path / "campaign", False, 0, 8, 60, 1, True)
    candidates = []
    for model, recall, cells, events in (
        ("conv_snn", 0.80, 300_000, 1_000_000),
        ("compact_snn", 0.785, 30_000, 200_000),
    ):
        config = tmp_path / f"{model}.json"
        config.write_text(json.dumps({"model": model}))
        candidates.append({
            "config_path": str(config), "selection_macro_recall": recall,
            "physical_cells": cells, "synaptic_events": events,
            "time_steps": 8, "total_pulses": 100,
        })
    winner = runner.select_deployment("all38", candidates)
    assert json.loads(open(winner["config_path"]).read())["model"] == "compact_snn"


def test_device_selection_and_unavailable_cuda(monkeypatch):
    from plantvillage_snn.utils import resolve_device
    import pytest
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")
