"""Empirical pulse-programmed conductance model generalized from cnnRef/device.py."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import torch
from torch import nn


class DeviceModel:
    """Build LTP/LTD dG(G), conductance curves, and pulse LUTs from measurements."""

    def __init__(self, csv_path: str | Path, v_bias: float = -5.0,
                 n_g_bins: int = 200, max_pulses: int = 200,
                 preprocess: str = "raw"):
        if v_bias == 0:
            raise ValueError("v_bias cannot be zero")
        if n_g_bins < 2 or max_pulses < 1:
            raise ValueError("n_g_bins >= 2 and max_pulses >= 1 are required")
        self.csv_path = str(Path(csv_path).resolve())
        self.v_bias = float(v_bias)
        self.n_g_bins = int(n_g_bins)
        self.max_pulses = int(max_pulses)
        if preprocess not in {"raw", "robust"}:
            raise ValueError("preprocess must be raw or robust")
        self.preprocess = preprocess
        self._load_data()
        self._extract_segments()
        self._build_dg_functions()
        self._build_curves()
        self._build_lut()

    def _load_data(self) -> None:
        frame = pd.read_csv(self.csv_path)
        if "Id" not in frame:
            raise ValueError("device CSV must contain an Id current column")
        current = pd.to_numeric(frame["Id"], errors="raise").to_numpy(np.float64)
        if len(current) < 10 or not np.isfinite(current).all():
            raise ValueError("device current data must contain at least 10 finite values")
        self.current = current
        self.g_measured = np.abs(current / self.v_bias)
        if self.preprocess == "robust":
            window = min(31, len(current) - (1 - len(current) % 2))
            self.g_raw = savgol_filter(self.g_measured, window, 3, mode="interp")
            self.g_raw = np.clip(self.g_raw, self.g_measured.min(), self.g_measured.max())
        else:
            self.g_raw = self.g_measured.copy()
        self.g_min = float(self.g_measured.min())
        self.g_max = float(self.g_measured.max())
        self.g_span = self.g_max - self.g_min
        self.g_mid = (self.g_min + self.g_max) / 2.0
        if self.g_span <= 0:
            raise ValueError("device measurements have no conductance range")

    def _extract_segments(self) -> None:
        delta = np.diff(self.g_raw)
        signs = np.sign(delta)
        signs[signs == 0] = 1
        ltp: list[np.ndarray] = []
        ltd: list[np.ndarray] = []
        start = 0
        for i in range(1, len(signs) + 1):
            boundary = i == len(signs) or signs[i] != signs[i - 1]
            if boundary:
                segment = self.g_raw[start:i + 1]
                if len(segment) >= 5:
                    (ltp if signs[i - 1] > 0 else ltd).append(segment)
                start = i
        if not ltp or not ltd:
            raise ValueError("device data must contain both LTP and LTD segments")
        self.ltp_segments = ltp
        self.ltd_segments = ltd

    def _build_dg_functions(self) -> None:
        def samples(segments: list[np.ndarray], direction: int):
            gs, dgs = [], []
            for segment in segments:
                raw_delta = direction * np.diff(segment)
                mask = raw_delta > 0
                gs.append(segment[:-1][mask])
                dgs.append(raw_delta[mask])
            return np.concatenate(gs), np.concatenate(dgs)

        g_ltp, dg_ltp = samples(self.ltp_segments, 1)
        g_ltd, dg_ltd = samples(self.ltd_segments, -1)
        n_bins = 50
        edges = np.linspace(self.g_min, self.g_max, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        def smooth(g_values: np.ndarray, dg_values: np.ndarray):
            average = np.zeros(n_bins)
            count = np.zeros(n_bins, dtype=int)
            for index in range(n_bins):
                mask = (g_values >= edges[index]) & (g_values < edges[index + 1])
                if mask.any():
                    average[index] = dg_values[mask].mean()
                    count[index] = int(mask.sum())
            valid = count > 0
            if valid.sum() < 2:
                raise ValueError("insufficient measured conductance coverage")
            fill = interp1d(centers[valid], average[valid], kind="linear",
                            bounds_error=False, fill_value="extrapolate")
            average = fill(centers)
            average = np.convolve(average, np.ones(5) / 5, mode="same")
            average = np.maximum(average, 0)
            return interp1d(centers, average, kind="linear", bounds_error=False,
                            fill_value="extrapolate")

        self._dg_ltp = smooth(g_ltp, dg_ltp)
        self._dg_ltd = smooth(g_ltd, dg_ltd)

    def _build_curves(self) -> None:
        self.g_centers = np.linspace(self.g_min, self.g_max, self.n_g_bins)
        shape = (self.n_g_bins, self.max_pulses + 1)
        self.ltp_curves = np.zeros(shape, dtype=np.float64)
        self.ltd_curves = np.zeros(shape, dtype=np.float64)
        for index, initial in enumerate(self.g_centers):
            up = down = initial
            self.ltp_curves[index, 0] = initial
            self.ltd_curves[index, 0] = initial
            for pulse in range(1, self.max_pulses + 1):
                up = min(up + max(float(self._dg_ltp(up)), 0), self.g_max)
                down = max(down - max(float(self._dg_ltd(down)), 0), self.g_min)
                self.ltp_curves[index, pulse] = up
                self.ltd_curves[index, pulse] = down

    def _build_lut(self) -> None:
        shape = (self.n_g_bins, self.n_g_bins)
        self.lut_pulses_ltp = np.zeros(shape, dtype=np.int32)
        self.lut_pulses_ltd = np.zeros(shape, dtype=np.int32)
        self.lut_new_g_ltp = np.zeros(shape, dtype=np.float64)
        self.lut_new_g_ltd = np.zeros(shape, dtype=np.float64)
        for current_index in range(self.n_g_bins):
            up = self.ltp_curves[current_index]
            down = self.ltd_curves[current_index]
            for target_index in range(current_index, self.n_g_bins):
                hits = np.flatnonzero(up >= self.g_centers[target_index] - 1e-15)
                pulse = int(hits[0]) if len(hits) else self.max_pulses
                self.lut_pulses_ltp[current_index, target_index] = pulse
                self.lut_new_g_ltp[current_index, target_index] = up[pulse]
            for target_index in range(current_index + 1):
                hits = np.flatnonzero(down <= self.g_centers[target_index] + 1e-15)
                pulse = int(hits[0]) if len(hits) else self.max_pulses
                self.lut_pulses_ltd[current_index, target_index] = pulse
                self.lut_new_g_ltd[current_index, target_index] = down[pulse]

    @property
    def effective_ltp_levels(self) -> int:
        step = float(self._dg_ltp(self.g_mid))
        return max(int(self.g_span / max(step, 1e-15)), 1)

    def apply_pulses_vectorized(self, g_current: np.ndarray, delta_g: np.ndarray):
        if g_current.shape != delta_g.shape:
            raise ValueError("g_current and delta_g must have identical shapes")
        shape = g_current.shape
        current = np.asarray(g_current, dtype=np.float64).ravel()
        delta = np.asarray(delta_g, dtype=np.float64).ravel()
        current = np.clip(current, self.g_min, self.g_max)
        target = np.clip(current + delta, self.g_min, self.g_max)
        edges = np.linspace(self.g_min, self.g_max, self.n_g_bins + 1)
        source_bin = np.clip(np.digitize(current, edges) - 1, 0, self.n_g_bins - 1)
        target_bin = np.clip(np.digitize(target, edges) - 1, 0, self.n_g_bins - 1)
        new_g = current.copy()
        pulses = np.zeros(len(current), dtype=np.int32)
        up = delta > 0
        down = delta < 0
        if up.any():
            pulses[up] = self.lut_pulses_ltp[source_bin[up], target_bin[up]]
            # LUT rows start at bin centers, which may lie slightly below an
            # actual value in the same bin. Never let an LTP request decrease G.
            new_g[up] = np.maximum(
                current[up], self.lut_new_g_ltp[source_bin[up], target_bin[up]]
            )
        if down.any():
            pulses[down] = self.lut_pulses_ltd[source_bin[down], target_bin[down]]
            # Likewise an LTD request must never increase conductance.
            new_g[down] = np.minimum(
                current[down], self.lut_new_g_ltd[source_bin[down], target_bin[down]]
            )
        return np.clip(new_g, self.g_min, self.g_max).reshape(shape), pulses.reshape(shape)

    def apply_pulses(self, g_current: float, delta_g: float) -> tuple[float, int]:
        new, pulses = self.apply_pulses_vectorized(
            np.asarray([g_current]), np.asarray([delta_g])
        )
        return float(new[0]), int(pulses[0])

    def conductance_to_weight(self, conductance: np.ndarray, scale: float = 1.0):
        return ((conductance - self.g_mid) / (self.g_span / 2.0)) * scale

    def weight_to_conductance(self, weight: np.ndarray, scale: float = 1.0):
        normalized = np.asarray(weight) / scale
        return np.clip(self.g_mid + normalized * (self.g_span / 2.0), self.g_min, self.g_max)

    def parameters_dict(self) -> dict:
        return {
            "csv_path": self.csv_path,
            "v_bias": self.v_bias,
            "n_g_bins": self.n_g_bins,
            "max_pulses": self.max_pulses,
            "preprocess": self.preprocess,
            "measurement_assumption": (
                "adjacent conductance samples are treated as empirical update increments; "
                "the CSV contains no synchronized optical-stimulus log"
            ),
            "g_min": self.g_min,
            "g_max": self.g_max,
            "ltp_segments": len(self.ltp_segments),
            "ltd_segments": len(self.ltd_segments),
            "dg_ltp_mid": float(self._dg_ltp(self.g_mid)),
            "dg_ltd_mid": float(self._dg_ltd(self.g_mid)),
        }


@dataclass
class UpdateStats:
    pulses: int = 0
    updated_cells: int = 0
    updated_synapses: int = 0


class HardwareState:
    """Differential-pair pulse mapping for signed Conv2d/Linear weights.

    Each logical synapse is represented by two positive conductances and
    decoded as ``scale * (G+ - G-) / G_span``. Biases, normalization and LIF
    dynamics remain ideal peripheral state.
    """

    def __init__(self, model: nn.Module, device_model: DeviceModel,
                 scale_margin: float = 1.5):
        self.device = device_model
        self.scale_margin = scale_margin
        self.conductance_plus: dict[str, np.ndarray] = {}
        self.conductance_minus: dict[str, np.ndarray] = {}
        self.accumulator_plus: dict[str, np.ndarray] = {}
        self.accumulator_minus: dict[str, np.ndarray] = {}
        self.scales: dict[str, float] = {}
        module_names = {
            f"{name}.weight" if name else "weight"
            for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        }
        for name, parameter in model.named_parameters():
            if name not in module_names:
                continue
            weight = parameter.detach().cpu().numpy().astype(np.float64)
            scale = max(float(np.abs(weight).max()) * scale_margin, 1e-6)
            normalized = np.clip(weight / scale, -1.0, 1.0)
            offset = normalized * (self.device.g_span / 2.0)
            plus = np.clip(self.device.g_mid + offset, self.device.g_min, self.device.g_max)
            minus = np.clip(self.device.g_mid - offset, self.device.g_min, self.device.g_max)
            self.conductance_plus[name] = plus.astype(np.float32)
            self.conductance_minus[name] = minus.astype(np.float32)
            self.accumulator_plus[name] = np.zeros_like(plus, dtype=np.float64)
            self.accumulator_minus[name] = np.zeros_like(minus, dtype=np.float64)
            self.scales[name] = scale
        if not self.conductance_plus:
            raise ValueError("model contains no Conv2d or Linear synaptic weights")

    @property
    def mapped_names(self) -> list[str]:
        return list(self.conductance_plus)

    @property
    def logical_synapses(self) -> int:
        return sum(value.size for value in self.conductance_plus.values())

    @property
    def physical_cells(self) -> int:
        return 2 * self.logical_synapses

    def sync_to_model(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, plus in self.conductance_plus.items():
                minus = self.conductance_minus[name]
                weight = self.scales[name] * (plus - minus) / self.device.g_span
                tensor = torch.as_tensor(weight, dtype=parameters[name].dtype,
                                         device=parameters[name].device)
                parameters[name].copy_(tensor)

    def update(self, name: str, gradient: np.ndarray, learning_rate: float,
               max_pulses_per_update: int = 8) -> UpdateStats:
        scale = self.scales[name]
        delta_weight = -learning_rate * np.asarray(gradient, dtype=np.float64)
        delta_g = (delta_weight / scale) * (self.device.g_span / 2.0)
        accumulated_plus = self.accumulator_plus[name]
        accumulated_minus = self.accumulator_minus[name]
        # Push-pull programming splits the requested signed update equally
        # across both positive conductance devices.
        accumulated_plus += delta_g
        accumulated_minus -= delta_g
        minimum_step = self.device.g_span / self.device.n_g_bins
        mask_plus = np.abs(accumulated_plus) >= minimum_step
        mask_minus = np.abs(accumulated_minus) >= minimum_step
        updated_cells = int(mask_plus.sum() + mask_minus.sum())
        updated_synapses = int(np.logical_or(mask_plus, mask_minus).sum())
        pulses_total = 0
        max_delta = (
            self.device.g_span * max_pulses_per_update
            / max(self.device.effective_ltp_levels, 1)
        )

        def apply(conductance: dict[str, np.ndarray], accumulated: np.ndarray,
                  mask: np.ndarray) -> int:
            if not mask.any():
                return 0
            current = conductance[name][mask].astype(np.float64)
            requested = np.clip(accumulated[mask], -max_delta, max_delta)
            new_g, pulses = self.device.apply_pulses_vectorized(current, requested)
            conductance[name][mask] = new_g.astype(np.float32)
            accumulated[mask] -= new_g - current
            return int(pulses.sum())

        if updated_cells:
            pulses_total += apply(self.conductance_plus, accumulated_plus, mask_plus)
            pulses_total += apply(self.conductance_minus, accumulated_minus, mask_minus)
        return UpdateStats(pulses_total, updated_cells, updated_synapses)

    def decay_accumulators(self, factor: float) -> None:
        for accumulator in (*self.accumulator_plus.values(), *self.accumulator_minus.values()):
            accumulator *= factor

    def statistics(self) -> dict:
        plus_values = np.concatenate([g.ravel() for g in self.conductance_plus.values()])
        minus_values = np.concatenate([g.ravel() for g in self.conductance_minus.values()])
        values = np.concatenate((plus_values, minus_values))
        common_mode = 0.5 * (plus_values + minus_values)
        differential = plus_values - minus_values
        epsilon = self.device.g_span / self.device.n_g_bins
        histogram_counts, histogram_edges = np.histogram(
            values, bins=20, range=(self.device.g_min, self.device.g_max)
        )
        return {
            "mapped_layers": len(self.conductance_plus),
            "mapping_mode": "differential_pair",
            "logical_synapses": self.logical_synapses,
            "physical_cells": self.physical_cells,
            "mapped_cells": self.physical_cells,
            "conductance_min": float(values.min()),
            "conductance_max": float(values.max()),
            "conductance_mean": float(values.mean()),
            "conductance_std": float(values.std()),
            "saturation_low_rate": float(np.mean(values <= self.device.g_min + epsilon)),
            "saturation_high_rate": float(np.mean(values >= self.device.g_max - epsilon)),
            "common_mode_mean": float(common_mode.mean()),
            "common_mode_std": float(common_mode.std()),
            "differential_mean": float(differential.mean()),
            "differential_std": float(differential.std()),
            "histogram": {
                "counts": histogram_counts.astype(int).tolist(),
                "conductance_edges": histogram_edges.tolist(),
            },
            "per_layer": {
                name: {
                    "plus_min": float(plus.min()), "plus_max": float(plus.max()),
                    "minus_min": float(self.conductance_minus[name].min()),
                    "minus_max": float(self.conductance_minus[name].max()),
                    "common_mode_mean": float(
                        (0.5 * (plus + self.conductance_minus[name])).mean()
                    ),
                }
                for name, plus in self.conductance_plus.items()
            },
        }

    def state_dict(self) -> dict:
        return {
            "mapping_mode": "differential_pair",
            "conductance_plus": self.conductance_plus,
            "conductance_minus": self.conductance_minus,
            "accumulator_plus": self.accumulator_plus,
            "accumulator_minus": self.accumulator_minus,
            "scales": self.scales,
            "scale_margin": self.scale_margin,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("mapping_mode") != "differential_pair":
            raise ValueError("hardware checkpoint is not a differential-pair state")
        if set(state["conductance_plus"]) != set(self.conductance_plus):
            raise ValueError("hardware state layer names do not match model")
        self.conductance_plus = state["conductance_plus"]
        self.conductance_minus = state["conductance_minus"]
        self.accumulator_plus = state["accumulator_plus"]
        self.accumulator_minus = state["accumulator_minus"]
        self.scales = state["scales"]
