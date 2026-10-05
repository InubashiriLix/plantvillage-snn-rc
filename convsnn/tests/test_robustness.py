from types import SimpleNamespace

import numpy as np
import pytest

from plantvillage_snn.robustness import apply_variation


def fixture_state():
    device = SimpleNamespace(g_min=1., g_max=3., g_span=2.)
    hardware = SimpleNamespace(device=device)
    original = {side: {"layer.weight": np.full(10000, 2., dtype=np.float32)}
                for side in ("conductance_plus", "conductance_minus")}
    return hardware, original


def test_static_noise_reproducible_independent_and_not_cumulative():
    hardware, original = fixture_state()
    apply_variation(hardware, original, .01, 42)
    first = hardware.conductance_plus["layer.weight"].copy()
    minus = hardware.conductance_minus["layer.weight"].copy()
    assert abs(np.std(first - 2.) - .02) < .001
    assert abs(np.corrcoef(first, minus)[0, 1]) < .04
    apply_variation(hardware, original, .02, 42)
    np.testing.assert_allclose(hardware.conductance_plus["layer.weight"] - 2., 2*(first - 2.), atol=3e-7)
    apply_variation(hardware, original, .01, 42)
    np.testing.assert_array_equal(first, hardware.conductance_plus["layer.weight"])
    apply_variation(hardware, original, 0, 99)
    for side in original:
        np.testing.assert_array_equal(getattr(hardware, side)["layer.weight"], original[side]["layer.weight"])
        assert not np.shares_memory(getattr(hardware, side)["layer.weight"], original[side]["layer.weight"])
        assert np.all(original[side]["layer.weight"] == 2.)


def test_physical_bounds_and_invalid_strength():
    hardware, original = fixture_state()
    clipped = apply_variation(hardware, original, 10, 5)
    assert .9 < clipped < 1
    for side in original:
        values = getattr(hardware, side)["layer.weight"]
        assert np.all((values >= 1.) & (values <= 3.))
    for sigma in (-1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            apply_variation(hardware, original, sigma, 0)
