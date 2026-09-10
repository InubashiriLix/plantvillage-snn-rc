from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import hashlib
import time

import torch
from torch import nn

from .config import ExperimentConfig
from .data import DataBundle, build_dataloaders, class_weights
from .device import DeviceModel, HardwareState
from .metrics import EvaluationAccumulator, hardware_aware_snn_loss
from .models import build_model
from .recording import (
    SCHEMA_VERSION, GradientRecorder, batchnorm_summaries, dataset_inventory,
    environment_snapshot, export_metrics_tables, parameter_summaries, rng_state,
    update_ratios, write_csv,
)
from .utils import jsonable, read_json, resolve_device, seed_everything, write_json


def ordered_class_names(mapping: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]


@torch.no_grad()
def evaluate(model: nn.Module, loader, class_names: list[str], device: torch.device,
             loss_weight: torch.Tensor | None, split: str,
             membrane_loss_weight: float = 0.25,
             prediction_path: str | Path | None = None) -> dict:
    model.eval()
    accumulator = EvaluationAccumulator(class_names, model.time_steps)
    prediction_handle = None
    prediction_writer = None
    offset = 0
    if prediction_path is not None:
        target = Path(prediction_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        prediction_handle = target.open("w", newline="", encoding="utf-8")
        spike_fields = [f"spike_{index:03d}" for index in range(len(class_names))]
        prediction_writer = csv.DictWriter(prediction_handle, fieldnames=[
            "path", "true_index", "true_class", "predicted_index", "predicted_class",
            "correct", "predicted_spikes", "true_spikes", "top1_margin", *spike_fields,
        ])
        prediction_writer.writeheader()
    try:
        for images, targets in loader:
            model.reset_state()
            images, targets = images.to(device), targets.to(device)
            result = model(images)
            loss, _ = hardware_aware_snn_loss(
                result, targets, loss_weight, membrane_weight=membrane_loss_weight,
                distill_weight=0, firing_weight=0,
            )
            accumulator.update(result, targets, loss)
            if prediction_writer is not None:
                counts = result["spike_counts"].detach().cpu()
                targets_cpu = targets.detach().cpu()
                predictions = counts.argmax(1)
                top2 = torch.topk(counts, min(2, counts.shape[1]), dim=1).values
                margins = top2[:, 0] - top2[:, 1] if counts.shape[1] > 1 else top2[:, 0]
                records = getattr(loader.dataset, "records", None)
                for index in range(len(targets_cpu)):
                    true_index = int(targets_cpu[index])
                    predicted_index = int(predictions[index])
                    row = {
                        "path": records[offset + index]["path"] if records else str(offset + index),
                        "true_index": true_index, "true_class": class_names[true_index],
                        "predicted_index": predicted_index,
                        "predicted_class": class_names[predicted_index],
                        "correct": int(true_index == predicted_index),
                        "predicted_spikes": float(counts[index, predicted_index]),
                        "true_spikes": float(counts[index, true_index]),
                        "top1_margin": float(margins[index]),
                    }
                    row.update({f"spike_{j:03d}": float(value) for j, value in enumerate(counts[index])})
                    prediction_writer.writerow(row)
                offset += len(targets_cpu)
    finally:
        if prediction_handle is not None:
            prediction_handle.close()
    metrics = accumulator.compute(split)
    metrics["inference_cost_per_sample"] = _inference_cost(model, metrics)
    return metrics


def _inference_cost(model: nn.Module, metrics: dict) -> dict[str, float]:
    """Architecture-aware event proxy; no unsupported energy conversion."""
    activity = metrics["snn_activity"]
    if not activity:
        return {}
    first = model.conv1
    first_elements = activity["conv1"].get("elements_per_sample", 0.0)
    # Direct-current injection makes the first convolution dense each time.
    dense_first_macs = first_elements * first.in_channels * first.kernel_size[0] * first.kernel_size[1]
    events = 0.0
    if hasattr(model, "conv2"):
        events += activity["conv1"].get("spikes_per_sample", 0.0) * model.conv2.out_channels * 9
        events += activity["conv2"].get("spikes_per_sample", 0.0) * model.conv3.out_channels * 9
    if hasattr(model, "fc1"):
        events += activity["conv3"].get("spikes_per_sample", 0.0) * model.fc1.out_features
        events += activity["hidden"].get("spikes_per_sample", 0.0) * model.output.out_features
    else:
        events += activity["conv3"].get("spikes_per_sample", 0.0) * model.output.out_features
    return {
        "dense_first_layer_macs": float(dense_first_macs),
        "spike_driven_synaptic_events_upper_bound": float(events),
        "output_spikes": float(activity["output"].get("spikes_per_sample", 0.0)),
    }


def _build_from_config(config: ExperimentConfig, num_classes: int) -> nn.Module:
    return build_model(
        config.model, num_classes, config.time_steps, config.beta, config.threshold,
        config.channels, config.layer_betas, config.layer_thresholds,
    )


def _checkpoint_metadata(config: ExperimentConfig, data: DataBundle,
                         device_parameters: dict | None) -> dict:
    return {
        "format_version": 1,
        "architecture": config.model,
        "stage": config.stage,
        "training_phase": config.phase,
        "class_mapping": data.class_to_index,
        "class_to_original_index": data.class_to_original_index,
        "split_seed": config.split_seed,
        "split_manifest_path": data.split_manifest_path,
        "snn_parameters": {
            "time_steps": config.time_steps,
            "beta": config.beta,
            "threshold": config.threshold,
            "decoder": "accumulated_output_spike_count",
            "batch_state_reset": True,
            "layer_betas": config.layer_betas,
            "layer_thresholds": config.layer_thresholds,
        },
        "architecture_parameters": {"channels": config.channels},
        "preprocessing": {
            "image_size": config.image_size,
            "input_range": [0.0, 1.0],
            "normalization_mean": None,
            "normalization_std": None,
            "augmentation": config.augmentation,
            "class_balance": config.class_balance,
            "input_encoding": config.input_encoding,
            "input_note": (
                "non-negative RGB intensity repeated as direct current for every time step; "
                "signed accumulation is implemented by differential synaptic weights"
            ),
        },
        "device_parameters": device_parameters,
        "conductance_mapping_boundary": {
            "mapped": ["Conv2d.weight as differential G+/G- pairs", "Linear.weight as differential G+/G- pairs"],
            "ideal": ["biases", "normalization parameters and running state", "fixed LIF dynamics"],
        },
        "signed_weight_mapping": config.signed_mapping,
        "loss": {
            "membrane_weight": config.membrane_loss_weight,
            "distill_weight": config.distill_weight,
            "distill_temperature": config.distill_temperature,
            "firing_regularization_weight": config.firing_regularization_weight,
            "firing_rate_bounds": [config.firing_rate_min, config.firing_rate_max],
        },
    }


def save_checkpoint(path: Path, model: nn.Module, metadata: dict, config: ExperimentConfig,
                    epoch: int, selection_metrics: dict, hardware: HardwareState | None,
                    training_history: list[dict], source_ideal_metrics: dict | None,
                    optimizer: torch.optim.Optimizer | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "metadata": metadata,
        "config": config.to_dict(),
        "epoch": epoch,
        "best_selection_metrics": selection_metrics,
        "hardware_state": hardware.state_dict() if hardware else None,
        "training_history": training_history,
        "source_ideal_metrics": source_ideal_metrics,
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "rng_state": rng_state(),
    }, path)


def load_checkpoint(path: str | Path, map_location: torch.device | str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def _validate_source_checkpoint(checkpoint: dict, config: ExperimentConfig,
                                class_mapping: dict[str, int]) -> None:
    metadata = checkpoint.get("metadata", {})
    if metadata.get("architecture") != config.model:
        raise ValueError("checkpoint architecture does not match requested model")
    if metadata.get("stage") != config.stage:
        raise ValueError("checkpoint stage does not match requested stage")
    if metadata.get("class_mapping") != class_mapping:
        raise ValueError("checkpoint class mapping does not match current data")
    source_config = checkpoint.get("config", {})
    if (
        source_config.get("split_seed") != config.split_seed
        or source_config.get("selection_fraction") != config.selection_fraction
    ):
        raise ValueError("hardware fine-tuning must use the ideal checkpoint's data split")
    if config.phase == "hw_finetune" and metadata.get("training_phase") != "ideal":
        raise ValueError("hardware-aware fine-tuning must start from an ideal checkpoint")
    source_snn = metadata.get("snn_parameters", {})
    requested = (config.time_steps, config.beta, config.threshold)
    recorded = (source_snn.get("time_steps"), source_snn.get("beta"), source_snn.get("threshold"))
    if recorded != requested:
        raise ValueError("hardware fine-tuning must retain the ideal checkpoint's LIF dynamics")
    source_architecture = metadata.get("architecture_parameters", {})
    if source_architecture.get("channels") != config.channels:
        raise ValueError("hardware fine-tuning must retain the ideal checkpoint's channel widths")


def _load_teacher(config: ExperimentConfig, data: DataBundle, device: torch.device,
                  num_classes: int) -> nn.Module | None:
    if config.model != "compact_snn" or not config.teacher_checkpoint or config.phase != "ideal":
        return None
    checkpoint = load_checkpoint(config.teacher_checkpoint, device)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("architecture") != "conv_snn":
        raise ValueError("compact_snn teacher must be a pure conv_snn checkpoint")
    if metadata.get("stage") != config.stage or metadata.get("class_mapping") != data.class_to_index:
        raise ValueError("teacher stage/class mapping does not match the student")
    teacher_config = ExperimentConfig(**checkpoint["config"])
    teacher = _build_from_config(teacher_config, num_classes).to(device)
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    data = build_dataloaders(config)
    class_names = ordered_class_names(data.class_to_index)
    model = _build_from_config(config, len(class_names)).to(device)
    teacher = _load_teacher(config, data, device, len(class_names))
    source_checkpoint = None
    source_ideal_metrics = None
    if config.checkpoint:
        source_checkpoint = load_checkpoint(config.checkpoint, device)
        _validate_source_checkpoint(source_checkpoint, config, data.class_to_index)
        model.load_state_dict(source_checkpoint["model_state"])
        source_ideal_metrics = source_checkpoint.get("best_selection_metrics")

    weight = None
    if config.class_balance == "class_weights":
        weight = class_weights(data.sample_counts["train"], device)

    hardware = None
    device_model = None
    mapped_names: set[str] = set()
    if config.phase == "hw_finetune":
        device_model = DeviceModel(
            config.device_csv, config.device_v_bias,
            config.device_g_bins, config.device_max_pulses, config.device_preprocess,
        )
        hardware = HardwareState(model, device_model, config.hw_scale_margin)
        mapped_names = set(hardware.mapped_names)
        hardware.sync_to_model(model)

    ideal_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name not in mapped_names and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        ideal_parameters if config.phase == "hw_finetune" else model.parameters(),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    metadata = _checkpoint_metadata(
        config, data, device_model.parameters_dict() if device_model else None
    )
    run_dir = Path(config.output_dir) / f"{config.model}_{config.stage}_{config.phase}_seed{config.seed}"
    records_dir = Path(config.recording_dir) if config.recording_dir else run_dir / "records"
    if config.recording_enabled:
        records_dir.mkdir(parents=True, exist_ok=True)
        write_json(records_dir / "environment.json", environment_snapshot(device))
    best_path = run_dir / "best.pt"
    history: list[dict] = []
    best_recall = -1.0
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        parameter_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        } if config.recording_enabled and config.record_internal_summaries else {}
        gradient_recorder = GradientRecorder()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0
        epoch_pulses = {
            name: {"pulses": 0, "updated_cells": 0, "updated_synapses": 0}
            for name in mapped_names
        }
        epoch_loss_components: dict[str, float] = {}
        for batch_index, (images, targets) in enumerate(data.train):
            if hardware:
                hardware.sync_to_model(model)
            model.reset_state()
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(images)
            teacher_result = None
            if teacher is not None:
                teacher.reset_state()
                with torch.no_grad():
                    teacher_result = teacher(images)
            loss, components = hardware_aware_snn_loss(
                result, targets, weight,
                membrane_weight=config.membrane_loss_weight,
                teacher_result=teacher_result,
                distill_weight=config.distill_weight,
                temperature=config.distill_temperature,
                firing_weight=config.firing_regularization_weight,
                firing_min=config.firing_rate_min,
                firing_max=config.firing_rate_max,
            )
            loss.backward()
            if config.recording_enabled and config.record_internal_summaries:
                gradient_recorder.update(model)
            optimizer.step()
            if hardware:
                if config.accum_decay_interval > 0 and batch_index > 0 and batch_index % config.accum_decay_interval == 0:
                    hardware.decay_accumulators(config.accum_decay)
                parameters = dict(model.named_parameters())
                for name in hardware.mapped_names:
                    gradient = parameters[name].grad
                    if gradient is None:
                        continue
                    stats = hardware.update(
                        name, gradient.detach().cpu().numpy(), config.learning_rate,
                        config.max_pulses_per_update,
                    )
                    epoch_pulses[name]["pulses"] += stats.pulses
                    epoch_pulses[name]["updated_cells"] += stats.updated_cells
                    epoch_pulses[name]["updated_synapses"] += stats.updated_synapses
                hardware.sync_to_model(model)
            batch = len(targets)
            train_loss_sum += float(loss.detach()) * batch
            train_correct += int((result["spike_counts"].argmax(1) == targets).sum())
            train_samples += batch
            for name, value in components.items():
                epoch_loss_components[name] = epoch_loss_components.get(name, 0.0) + value * batch

        selection = evaluate(
            model, data.selection, class_names, device, weight, "selection",
            config.membrane_loss_weight,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_samples, 1),
            "train_accuracy": train_correct / max(train_samples, 1),
            "selection": selection,
            "pulse_updates": epoch_pulses,
            "loss_components": {
                name: value / max(train_samples, 1)
                for name, value in epoch_loss_components.items()
            },
            "duration_seconds": time.perf_counter() - epoch_started,
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        }
        if config.recording_enabled and config.record_internal_summaries:
            epoch_record["internal_summaries"] = {
                "parameters": parameter_summaries(model),
                "gradients": gradient_recorder.compute(),
                "update_ratio": update_ratios(parameter_before, model),
                "batchnorm": batchnorm_summaries(model),
            }
        history.append(epoch_record)
        should_stop = False
        if selection["macro_recall"] > best_recall:
            best_recall = selection["macro_recall"]
            stale_epochs = 0
            save_checkpoint(best_path, model, metadata, config, epoch, selection,
                            hardware, history, source_ideal_metrics, optimizer)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                should_stop = True
        if config.recording_enabled:
            save_checkpoint(run_dir / "latest.pt", model, metadata, config, epoch, selection,
                            hardware, history, source_ideal_metrics, optimizer)
            if epoch % config.checkpoint_interval == 0:
                save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", model, metadata, config,
                                epoch, selection, hardware, history,
                                source_ideal_metrics, optimizer)
        if should_stop:
            break

    # ``history`` is the complete run history. Checkpoints intentionally keep
    # only the prefix that existed when they became best, but the experiment
    # artifact must retain later stale epochs as well so early stopping can be
    # audited and plotted truthfully.
    executed_history = list(history)
    if config.recording_enabled:
        save_checkpoint(run_dir / "final.pt", model, metadata, config,
                        executed_history[-1]["epoch"], executed_history[-1]["selection"],
                        hardware, executed_history, source_ideal_metrics, optimizer)
    best = load_checkpoint(best_path, device)
    model.load_state_dict(best["model_state"])
    if hardware and best["hardware_state"]:
        hardware.load_state_dict(best["hardware_state"])
        hardware.sync_to_model(model)
    best_selection = evaluate(
        model, data.selection, class_names, device, weight, "selection",
        config.membrane_loss_weight,
        records_dir / "selection_predictions.csv" if config.recording_enabled and config.record_sample_predictions else None,
    )
    final_metrics = None
    if data.final_test is not None:
        final_metrics = evaluate(
            model, data.final_test, class_names, device, weight, "final_test",
            config.membrane_loss_weight,
            records_dir / "final_predictions.csv" if config.recording_enabled and config.record_sample_predictions else None,
        )

    penalty = None
    if source_ideal_metrics:
        penalty = {
            "selection_top1_accuracy": source_ideal_metrics["top1_accuracy"] - best_selection["top1_accuracy"],
            "selection_macro_recall": source_ideal_metrics["macro_recall"] - best_selection["macro_recall"],
        }
    best_pulse_totals = {
        name: {"pulses": 0, "updated_cells": 0, "updated_synapses": 0}
        for name in mapped_names
    }
    for epoch_record in best["training_history"]:
        for name, values in epoch_record["pulse_updates"].items():
            best_pulse_totals[name]["pulses"] += values["pulses"]
            best_pulse_totals[name]["updated_cells"] += values["updated_cells"]
            best_pulse_totals[name]["updated_synapses"] += values.get("updated_synapses", 0)
    device_sensitivity = None
    if device_model is not None and config.device_preprocess == "raw":
        robust_device = DeviceModel(
            config.device_csv, config.device_v_bias,
            config.device_g_bins, config.device_max_pulses, "robust",
        )
        raw_parameters = device_model.parameters_dict()
        robust_parameters = robust_device.parameters_dict()
        device_sensitivity = {
            "official_preprocess": "raw",
            "alternate_preprocess": "robust_savgol_window31_order3",
            "raw": raw_parameters,
            "robust": robust_parameters,
            "relative_mid_step_change": {
                direction: (
                    robust_parameters[f"dg_{direction}_mid"]
                    / max(raw_parameters[f"dg_{direction}_mid"], 1e-30) - 1.0
                )
                for direction in ("ltp", "ltd")
            },
            "interpretation_limit": (
                "No optical stimulus timestamps are present; this is a preprocessing "
                "sensitivity analysis, not a calibrated optical-pulse model."
            ),
        }
    artifact = {
        "format_version": 1,
        "run": {"model": config.model, "stage": config.stage, "phase": config.phase, "seed": config.seed},
        "config": config.to_dict(),
        "metadata": metadata,
        "checkpoint": str(best_path.resolve()),
        "best_epoch": best["epoch"],
        "sample_counts": data.sample_counts,
        "selection_metrics": best_selection,
        "final_test_metrics": final_metrics,
        "pulse_statistics": {
            "per_layer": best_pulse_totals,
            "total_pulses": sum(v["pulses"] for v in best_pulse_totals.values()),
            "total_updated_cells": sum(v["updated_cells"] for v in best_pulse_totals.values()),
            "total_updated_synapses": sum(v["updated_synapses"] for v in best_pulse_totals.values()),
        } if hardware else None,
        "conductance_statistics": hardware.statistics() if hardware else None,
        "device_curve_sensitivity": device_sensitivity,
        "model_cost": model.synapse_cost(),
        "teacher": ({
            "checkpoint": str(Path(config.teacher_checkpoint).resolve()),
            "sha256": hashlib.sha256(Path(config.teacher_checkpoint).read_bytes()).hexdigest(),
            "architecture": "conv_snn",
        } if teacher is not None else None),
        "ideal_to_hardware_penalty": penalty,
        "history": best["training_history"],
        "executed_history": executed_history,
        "executed_epochs": len(executed_history),
    }
    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, jsonable(artifact))
    if config.recording_enabled:
        export_metrics_tables(records_dir, executed_history, best_selection, final_metrics)
        write_json(records_dir / "best_internal_summaries.json", {
            "schema_version": SCHEMA_VERSION,
            "parameters": parameter_summaries(model),
            "batchnorm": batchnorm_summaries(model),
        })
        write_json(records_dir / "recording_manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "run": artifact["run"], "best_epoch": artifact["best_epoch"],
            "checkpoint_interval": config.checkpoint_interval,
            "raw_activations_saved": False, "raw_gradients_saved": False,
            "prediction_spike_columns": len(class_names) if config.record_sample_predictions else 0,
        })
        if config.dataset_hash_mode == "sha256":
            inventory_rows, aggregate_sha256 = dataset_inventory(config.data_root)
            write_csv(records_dir / "dataset_files.csv", inventory_rows,
                      ["path", "size_bytes", "sha256"])
            manifest = read_json(records_dir / "recording_manifest.json")
            manifest["dataset"] = {
                "hash_mode": "sha256", "file_count": len(inventory_rows),
                "aggregate_sha256": aggregate_sha256,
            }
            write_json(records_dir / "recording_manifest.json", manifest)
    artifact["metrics_path"] = str(metrics_path.resolve())
    return artifact


def evaluate_saved_run(metrics_path: str | Path) -> dict[str, Any]:
    """Evaluate a frozen selected checkpoint on the untouched final-test split."""
    path = Path(metrics_path).resolve()
    artifact = read_json(path)
    config = ExperimentConfig(**artifact["config"]).with_overrides({
        "evaluate_final": True,
        "max_final_samples": None,
    })
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    data = build_dataloaders(config)
    if data.final_test is None:
        raise RuntimeError("final-test loader was not created")
    class_names = ordered_class_names(data.class_to_index)
    model = _build_from_config(config, len(class_names)).to(device)
    checkpoint = load_checkpoint(artifact["checkpoint"], device)
    checkpoint_metadata = checkpoint.get("metadata", {})
    if (
        checkpoint_metadata.get("architecture") != config.model
        or checkpoint_metadata.get("stage") != config.stage
        or checkpoint_metadata.get("training_phase") != config.phase
        or checkpoint_metadata.get("class_mapping") != data.class_to_index
    ):
        raise ValueError("frozen checkpoint metadata does not match its metrics/config")
    model.load_state_dict(checkpoint["model_state"])
    if config.phase == "hw_finetune":
        device_model = DeviceModel(
            config.device_csv, config.device_v_bias,
            config.device_g_bins, config.device_max_pulses, config.device_preprocess,
        )
        hardware = HardwareState(model, device_model, config.hw_scale_margin)
        if not checkpoint.get("hardware_state"):
            raise ValueError("hardware checkpoint has no differential conductance state")
        hardware.load_state_dict(checkpoint["hardware_state"])
        hardware.sync_to_model(model)
    weight = (
        class_weights(data.sample_counts["train"], device)
        if config.class_balance == "class_weights" else None
    )
    final_metrics = evaluate(
        model, data.final_test, class_names, device, weight, "final_test",
        config.membrane_loss_weight,
        ((Path(config.recording_dir) if config.recording_dir else path.parent / "records") / "final_predictions.csv")
        if config.recording_enabled and config.record_sample_predictions else None,
    )
    artifact["final_test_metrics"] = final_metrics
    artifact["final_evaluation"] = {
        "performed_after_selection": True,
        "checkpoint_retrained": False,
        "sample_limits_cleared": True,
    }
    write_json(path, jsonable(artifact))
    artifact["metrics_path"] = str(path)
    return artifact
