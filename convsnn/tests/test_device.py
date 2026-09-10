from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from plantvillage_snn.device import DeviceModel, HardwareState


@pytest.fixture(scope="module")
def device():
    return DeviceModel(str(Path(__file__).resolve().parents[1] / "cnnRef/source/data.csv"), n_g_bins=24, max_pulses=30)


def test_measured_data_parsing_and_directions(device):
    assert len(device.current) == 4096
    assert device.g_min > 0 and device.g_max > device.g_min
    assert device.ltp_segments and device.ltd_segments
    start = device.g_mid
    up, up_pulses = device.apply_pulses(start, device.g_span * 0.3)
    down, down_pulses = device.apply_pulses(start, -device.g_span * 0.3)
    assert up >= start and down <= start
    assert 0 <= up_pulses <= device.max_pulses
    assert 0 <= down_pulses <= device.max_pulses


def test_lut_bounds_clipping_and_weight_round_trip(device):
    current = np.array([device.g_min, device.g_mid, device.g_max])
    delta = np.array([-10 * device.g_span, 0, 10 * device.g_span])
    new, pulses = device.apply_pulses_vectorized(current, delta)
    assert np.all(new >= device.g_min) and np.all(new <= device.g_max)
    assert np.all(pulses >= 0) and np.all(pulses <= device.max_pulses)
    weights = np.linspace(-1, 1, 11)
    restored = device.conductance_to_weight(device.weight_to_conductance(weights), 1.0)
    np.testing.assert_allclose(restored, weights, atol=1e-6)


def test_hardware_state_maps_only_synapses_and_updates(device):
    model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3), nn.Linear(3, 2))
    state = HardwareState(model, device, scale_margin=2.0)
    assert state.mapped_names == ["0.weight", "2.weight"]
    before_plus = state.conductance_plus["0.weight"].copy()
    before_minus = state.conductance_minus["0.weight"].copy()
    expected = model[0].weight.detach().clone()
    state.sync_to_model(model)
    torch.testing.assert_close(model[0].weight, expected, atol=1e-6, rtol=1e-5)
    stats = state.update("0.weight", np.full_like(before_plus, -100.0), 0.1, 4)
    assert stats.updated_cells == 2 * before_plus.size
    assert stats.updated_synapses == before_plus.size
    assert stats.pulses >= 0
    assert np.all(state.conductance_plus["0.weight"] >= before_plus)
    assert np.all(state.conductance_minus["0.weight"] <= before_minus)
    for conductance in (state.conductance_plus, state.conductance_minus):
        assert np.all(conductance["0.weight"] >= device.g_min)
        assert np.all(conductance["0.weight"] <= device.g_max)
    state.sync_to_model(model)
    assert torch.isfinite(model[0].weight).all()
    summary = state.statistics()
    assert summary["mapped_layers"] == 2
    assert summary["mapping_mode"] == "differential_pair"
    assert summary["logical_synapses"] == 18
    assert summary["physical_cells"] == 36
    assert 0 <= summary["saturation_low_rate"] <= 1


def test_robust_device_preprocessing_is_bounded_and_explicit():
    robust = DeviceModel(
        str(Path(__file__).resolve().parents[1] / "cnnRef/source/data.csv"), n_g_bins=12, max_pulses=12, preprocess="robust"
    )
    parameters = robust.parameters_dict()
    assert parameters["preprocess"] == "robust"
    assert "no synchronized optical-stimulus log" in parameters["measurement_assumption"]
    assert robust.g_min <= robust.g_raw.min() <= robust.g_raw.max() <= robust.g_max
