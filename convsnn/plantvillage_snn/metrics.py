from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class EvaluationAccumulator:
    def __init__(self, class_names: list[str], time_steps: int):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.time_steps = time_steps
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.temporal_correct = np.zeros(time_steps, dtype=np.int64)
        self.loss_sum = 0.0
        self.samples = 0
        self.activity_spikes: dict[str, float] = {}
        self.activity_elements: dict[str, int] = {}
        self.activity_neuron_counts: dict[str, np.ndarray] = {}
        self.membrane_sum = 0.0
        self.membrane_sum_sq = 0.0
        self.membrane_count = 0
        self.membrane_min = np.inf
        self.membrane_max = -np.inf
        self.cost_totals: dict[str, float] = {}

    def update(self, result: dict, targets: torch.Tensor, loss: torch.Tensor) -> None:
        spikes = result["spikes"].detach()
        counts = result["spike_counts"].detach()
        predictions = counts.argmax(1)
        target_np = targets.detach().cpu().numpy()
        pred_np = predictions.cpu().numpy()
        np.add.at(self.confusion, (target_np, pred_np), 1)
        batch_size = len(targets)
        self.loss_sum += float(loss.detach()) * batch_size
        self.samples += batch_size
        cumulative = spikes.cumsum(0)
        for step in range(self.time_steps):
            self.temporal_correct[step] += int(
                (cumulative[step].argmax(1) == targets).sum().item()
            )
        for name, layer_spikes in result["activity"].items():
            detached = layer_spikes.detach()
            self.activity_spikes[name] = self.activity_spikes.get(name, 0.0) + float(detached.sum())
            self.activity_elements[name] = self.activity_elements.get(name, 0) + detached.numel()
            if detached.ndim == 5:
                per_neuron = detached.sum(dim=(0, 1, 3, 4)).cpu().numpy()
            else:
                per_neuron = detached.sum(dim=(0, 1)).cpu().numpy()
            if name not in self.activity_neuron_counts:
                self.activity_neuron_counts[name] = per_neuron
            else:
                self.activity_neuron_counts[name] += per_neuron
        membrane = result["membranes"].detach().float()
        self.membrane_sum += float(membrane.sum())
        self.membrane_sum_sq += float((membrane * membrane).sum())
        self.membrane_count += membrane.numel()
        self.membrane_min = min(self.membrane_min, float(membrane.min()))
        self.membrane_max = max(self.membrane_max, float(membrane.max()))
        for name, value in result.get("inference_cost", {}).items():
            self.cost_totals[name] = self.cost_totals.get(name, 0.0) + float(value)

    def compute(self, split: str) -> dict:
        true_counts = self.confusion.sum(1)
        predicted_counts = self.confusion.sum(0)
        correct = np.diag(self.confusion)
        recalls = np.divide(correct, true_counts, out=np.zeros_like(correct, dtype=float), where=true_counts > 0)
        precisions = np.divide(correct, predicted_counts, out=np.zeros_like(correct, dtype=float), where=predicted_counts > 0)
        per_class = {
            name: {
                "index": index,
                "recall": float(recalls[index]),
                "precision": float(precisions[index]),
                "samples": int(true_counts[index]),
                "correct": int(correct[index]),
            }
            for index, name in enumerate(self.class_names)
        }
        activity = {}
        for name in self.activity_spikes:
            neuron_counts = self.activity_neuron_counts[name]
            silent = int(np.sum(neuron_counts == 0))
            activity[name] = {
                "firing_rate": self.activity_spikes[name] / max(self.activity_elements[name], 1),
                "spikes_per_sample": self.activity_spikes[name] / max(self.samples, 1),
                "elements_per_sample": self.activity_elements[name] / max(self.samples, 1),
                "silent_neurons": silent,
                "neuron_count": int(neuron_counts.size),
                "silent_neuron_rate": silent / max(neuron_counts.size, 1),
            }
        mean = self.membrane_sum / max(self.membrane_count, 1)
        variance = self.membrane_sum_sq / max(self.membrane_count, 1) - mean * mean
        return {
            "split": split,
            "loss": self.loss_sum / max(self.samples, 1),
            "top1_accuracy": float(correct.sum() / max(self.samples, 1)),
            "macro_recall": float(recalls.mean()),
            "sample_count": self.samples,
            "sample_counts_per_class": true_counts.astype(int).tolist(),
            "per_class": per_class,
            "confusion_matrix": self.confusion.tolist(),
            "temporal_accuracy": (self.temporal_correct / max(self.samples, 1)).tolist(),
            "snn_activity": activity,
            "membrane": {
                "mean": mean,
                "std": float(np.sqrt(max(variance, 0))),
                "min": self.membrane_min if self.membrane_count else None,
                "max": self.membrane_max if self.membrane_count else None,
            },
            "inference_cost_per_sample": {
                name: value / max(self.samples, 1) for name, value in self.cost_totals.items()
            },
        }


def spike_count_loss(result: dict, targets: torch.Tensor, weight: torch.Tensor | None = None):
    # Dividing by T keeps the logit scale stable across time-step searches while
    # decoding still uses literal accumulated output spike counts.
    logits = result["spike_counts"] / result["spikes"].shape[0]
    return F.cross_entropy(logits, targets, weight=weight)


def hardware_aware_snn_loss(result: dict, targets: torch.Tensor,
                            weight: torch.Tensor | None = None,
                            membrane_weight: float = 0.25,
                            teacher_result: dict | None = None,
                            distill_weight: float = 0.5,
                            temperature: float = 2.0,
                            firing_weight: float = 1e-3,
                            firing_min: float = 0.01,
                            firing_max: float = 0.35) -> tuple[torch.Tensor, dict[str, float]]:
    spike_loss = spike_count_loss(result, targets, weight)
    membrane_logits = result["membranes"].mean(0)
    membrane_loss = F.cross_entropy(membrane_logits, targets, weight=weight)
    distill = spike_loss.new_zeros(())
    if teacher_result is not None and distill_weight > 0:
        student = result["spike_counts"] / result["spikes"].shape[0]
        teacher = teacher_result["spike_counts"].detach() / teacher_result["spikes"].shape[0]
        distill = F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)
    firing_penalty = spike_loss.new_zeros(())
    if firing_weight > 0:
        rates = torch.stack([spikes.float().mean() for spikes in result["activity"].values()])
        firing_penalty = (
            F.relu(firing_min - rates).square() + F.relu(rates - firing_max).square()
        ).mean()
    total = (
        spike_loss + membrane_weight * membrane_loss
        + distill_weight * distill + firing_weight * firing_penalty
    )
    components = {
        "spike": float(spike_loss.detach()),
        "membrane": float(membrane_loss.detach()),
        "distillation": float(distill.detach()),
        "firing_regularization": float(firing_penalty.detach()),
    }
    return total, components
