from __future__ import annotations

from typing import Any, Sequence

import snntorch as snn
from snntorch import surrogate
import torch
from torch import nn
import torch.nn.functional as F


class TemporalSNN(nn.Module):
    """Base class whose states live only inside a forward call.

    Membranes are passed explicitly through the unrolled loop. ``reset_state``
    clears snnTorch's internal mirror at every batch boundary so no state can
    leak into the next sample batch.
    """

    def __init__(self, time_steps: int):
        super().__init__()
        self.time_steps = time_steps

    def reset_state(self) -> None:
        # snnTorch keeps a mirror of explicitly passed membrane tensors on each
        # neuron module. Clear those mirrors at every batch boundary as well.
        for module in self.modules():
            if isinstance(module, snn.Leaky):
                module.reset_mem()

    @staticmethod
    def _result(output_spikes, output_membranes, activity):
        spikes = torch.stack(output_spikes)
        membranes = torch.stack(output_membranes)
        return {
            "spikes": spikes,
            "spike_counts": spikes.sum(0),
            "membranes": membranes,
            "activity": {name: torch.stack(values) for name, values in activity.items()},
        }

    def synapse_cost(self) -> dict[str, int]:
        logical = sum(
            module.weight.numel() for module in self.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        )
        return {
            "logical_synapses": logical,
            "differential_physical_cells": 2 * logical,
        }


def _lif_parameters(count: int, beta: float, threshold: float,
                    layer_betas: Sequence[float] | None,
                    layer_thresholds: Sequence[float] | None) -> tuple[list[float], list[float]]:
    betas = list(layer_betas) if layer_betas is not None else [beta] * count
    thresholds = list(layer_thresholds) if layer_thresholds is not None else [threshold] * count
    if len(betas) != count or len(thresholds) != count:
        raise ValueError(f"expected {count} layer LIF parameters")
    return betas, thresholds


class ConvSNN(TemporalSNN):
    """Convolutional SNN: every learned block is followed by a LIF neuron."""

    def __init__(self, num_classes: int, time_steps: int = 8, beta: float = 0.9,
                 threshold: float = 1.0, layer_betas: Sequence[float] | None = None,
                 layer_thresholds: Sequence[float] | None = None):
        super().__init__(time_steps)
        spike_grad = surrogate.fast_sigmoid(slope=25)
        betas, thresholds = _lif_parameters(
            5, beta, threshold, layer_betas, layer_thresholds
        )
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.lif1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], spike_grad=spike_grad)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.lif3 = snn.Leaky(beta=betas[2], threshold=thresholds[2], spike_grad=spike_grad)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.lif4 = snn.Leaky(beta=betas[3], threshold=thresholds[3], spike_grad=spike_grad)
        self.output = nn.Linear(128, num_classes)
        self.lif_out = snn.Leaky(beta=betas[4], threshold=thresholds[4], spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        mem1 = mem2 = mem3 = mem4 = mem_out = None
        output_spikes, output_membranes = [], []
        activity = {"conv1": [], "conv2": [], "conv3": [], "hidden": [], "output": []}
        for _ in range(self.time_steps):
            spk1, mem1 = self.lif1(self.bn1(self.conv1(x)), mem1)
            spk2, mem2 = self.lif2(self.bn2(self.conv2(F.max_pool2d(spk1, 2))), mem2)
            spk3, mem3 = self.lif3(self.bn3(self.conv3(F.max_pool2d(spk2, 2))), mem3)
            flat = F.adaptive_avg_pool2d(F.max_pool2d(spk3, 2), (4, 4)).flatten(1)
            spk4, mem4 = self.lif4(self.fc1(flat), mem4)
            spk_out, mem_out = self.lif_out(self.output(spk4), mem_out)
            for name, spike in zip(activity, (spk1, spk2, spk3, spk4, spk_out)):
                activity[name].append(spike)
            output_spikes.append(spk_out)
            output_membranes.append(mem_out)
        return self._result(output_spikes, output_membranes, activity)


class CompactSNN(TemporalSNN):
    """Hardware-oriented end-to-end SNN without a flattened hidden FC layer."""

    def __init__(self, num_classes: int, time_steps: int = 8, beta: float = 0.9,
                 threshold: float = 1.0, channels: Sequence[int] = (12, 24, 48),
                 layer_betas: Sequence[float] | None = None,
                 layer_thresholds: Sequence[float] | None = None):
        super().__init__(time_steps)
        if len(channels) != 3:
            raise ValueError("CompactSNN requires three channel widths")
        c1, c2, c3 = (int(value) for value in channels)
        self.channels = (c1, c2, c3)
        spike_grad = surrogate.fast_sigmoid(slope=25)
        betas, thresholds = _lif_parameters(
            4, beta, threshold, layer_betas, layer_thresholds
        )
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.lif1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.lif2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], spike_grad=spike_grad)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)
        self.lif3 = snn.Leaky(beta=betas[2], threshold=thresholds[2], spike_grad=spike_grad)
        self.output = nn.Linear(c3, num_classes)
        self.lif_out = snn.Leaky(beta=betas[3], threshold=thresholds[3], spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        mem1 = mem2 = mem3 = mem_out = None
        output_spikes, output_membranes = [], []
        activity = {"conv1": [], "conv2": [], "conv3": [], "output": []}
        for _ in range(self.time_steps):
            spk1, mem1 = self.lif1(self.bn1(self.conv1(x)), mem1)
            pooled1 = F.max_pool2d(spk1, 2)
            spk2, mem2 = self.lif2(self.bn2(self.conv2(pooled1)), mem2)
            pooled2 = F.max_pool2d(spk2, 2)
            spk3, mem3 = self.lif3(self.bn3(self.conv3(pooled2)), mem3)
            # Pooling is non-learned spike aggregation. Every learned layer
            # still receives either input current (conv1) or upstream spikes.
            pooled3 = F.adaptive_avg_pool2d(spk3, (1, 1)).flatten(1)
            spk_out, mem_out = self.lif_out(self.output(pooled3), mem_out)
            for name, spike in zip(activity, (spk1, spk2, spk3, spk_out)):
                activity[name].append(spike)
            output_spikes.append(spk_out)
            output_membranes.append(mem_out)
        return self._result(output_spikes, output_membranes, activity)


def build_model(model_type: str, num_classes: int, time_steps: int = 8,
                beta: float = 0.9, threshold: float = 1.0,
                channels: Sequence[int] | None = None,
                layer_betas: Sequence[float] | None = None,
                layer_thresholds: Sequence[float] | None = None) -> TemporalSNN:
    kwargs = dict(
        num_classes=num_classes, time_steps=time_steps, beta=beta, threshold=threshold,
        layer_betas=layer_betas, layer_thresholds=layer_thresholds,
    )
    if model_type == "conv_snn":
        return ConvSNN(**kwargs)
    if model_type == "compact_snn":
        return CompactSNN(**kwargs, channels=channels or (12, 24, 48))
    raise ValueError(f"unknown model type: {model_type}")
