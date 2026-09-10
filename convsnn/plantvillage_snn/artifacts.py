from __future__ import annotations

import hashlib
from pathlib import Path

from .utils import read_json, write_json


def select_top10(metrics_path: str | Path, output_path: str | Path) -> dict:
    path = Path(metrics_path)
    artifact = read_json(path)
    run = artifact.get("run", {})
    if (
        run.get("model") not in {"conv_snn", "compact_snn"}
        or run.get("stage") != "all38"
        or run.get("phase") != "hw_finetune"
    ):
        raise ValueError("top-10 must come from an all38 hardware-aware pure-SNN run")
    metrics = artifact.get("selection_metrics")
    if not metrics or metrics.get("split") != "selection":
        raise ValueError("top-10 ranking requires selection-validation metrics")
    per_class = metrics["per_class"]
    original_mapping = artifact["metadata"]["class_to_original_index"]
    if len(per_class) != 38 or set(per_class) != set(original_mapping):
        raise ValueError("official all38 metrics must contain exactly the mapped 38 classes")
    ranked = sorted(
        per_class,
        key=lambda name: (-per_class[name]["recall"], -per_class[name]["precision"], name),
    )[:10]
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    selected = {
        "version": 1,
        "source_stage": "all38",
        "source_model": run["model"],
        "source_phase": "hw_finetune",
        "ranking_metric": "selection_validation_per_class_recall",
        "tie_breakers": ["precision_descending", "class_name_ascending"],
        "source_metrics": str(path.resolve()),
        "source_metrics_sha256": content_hash,
        "classes": [
            {
                "name": name,
                "original_index": original_mapping[name],
                "new_index": new_index,
                "selection_recall": per_class[name]["recall"],
                "selection_precision": per_class[name]["precision"],
            }
            for new_index, name in enumerate(ranked)
        ],
    }
    write_json(output_path, selected)
    return selected


def build_report(metrics_paths: list[str | Path], output_dir: str | Path) -> dict:
    records = []
    seen = set()
    for path in metrics_paths:
        artifact = read_json(path)
        run = artifact["run"]
        channels = tuple((artifact.get("config") or {}).get("channels") or ())
        key = (run["model"], run["stage"], run["phase"], run["seed"], channels)
        if key in seen:
            raise ValueError(f"duplicate run in report: {key}")
        seen.add(key)
        selection = artifact["selection_metrics"]
        final = artifact.get("final_test_metrics")
        pulse_statistics = artifact.get("pulse_statistics") or {}
        conductance = artifact.get("conductance_statistics") or {}
        records.append({
            **run,
            "metrics_path": str(Path(path).resolve()),
            "selection_top1_accuracy": selection["top1_accuracy"],
            "selection_macro_recall": selection["macro_recall"],
            "final_top1_accuracy": final["top1_accuracy"] if final else None,
            "final_macro_recall": final["macro_recall"] if final else None,
            "ideal_to_hardware_penalty": artifact.get("ideal_to_hardware_penalty"),
            "total_pulses": pulse_statistics.get("total_pulses"),
            "updated_physical_cells": pulse_statistics.get("total_updated_cells"),
            "updated_logical_synapses": pulse_statistics.get("total_updated_synapses"),
            "logical_synapses": (artifact.get("model_cost") or {}).get("logical_synapses"),
            "physical_cells": (artifact.get("model_cost") or {}).get("differential_physical_cells"),
            "synaptic_events": selection.get("inference_cost_per_sample", {}).get(
                "spike_driven_synaptic_events_upper_bound"
            ),
            "time_steps": artifact.get("config", {}).get("time_steps"),
            "channels": list(channels) if channels else None,
            "saturation_low_rate": conductance.get("saturation_low_rate"),
            "saturation_high_rate": conductance.get("saturation_high_rate"),
            "common_mode_std": conductance.get("common_mode_std"),
        })
    records.sort(key=lambda r: (r["stage"], r["model"], r["phase"], r["seed"]))
    baseline_cells = {
        (record["stage"], record["phase"]): record["physical_cells"]
        for record in records
        if record["model"] == "conv_snn" and record["physical_cells"]
    }
    for record in records:
        baseline = baseline_cells.get((record["stage"], record["phase"]))
        record["compression_vs_conv_snn"] = (
            baseline / record["physical_cells"]
            if baseline and record["physical_cells"] else None
        )
    report = {"format_version": 1, "runs": records}
    out = Path(output_dir)
    write_json(out / "comparison.json", report)
    lines = [
        "# PlantVillage pure-SNN hardware comparison",
        "",
        "Final-test values are shown only for runs explicitly evaluated after selection. Device energy/area is not estimated without measurements.",
        "",
        "| Stage | Architecture | Phase | Selection accuracy | Selection macro recall | Final accuracy | Final macro recall | Logical synapses | Differential cells | Compression | T | Events/sample | Write pulses | Updated cells | Saturation low/high |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def percent(value):
        return "—" if value is None else f"{100 * value:.2f}%"
    def multiple(value):
        return "—" if value is None else f"{value:.2f}×"
    def architecture(row):
        suffix = "-".join(str(value) for value in (row.get("channels") or []))
        return row["model"] if not suffix else f"{row['model']}[{suffix}]"
    for row in records:
        lines.append(
            f"| {row['stage']} | {architecture(row)} | {row['phase']} | "
            f"{percent(row['selection_top1_accuracy'])} | {percent(row['selection_macro_recall'])} | "
            f"{percent(row['final_top1_accuracy'])} | {percent(row['final_macro_recall'])} | "
            f"{row['logical_synapses'] or '—'} | {row['physical_cells'] or '—'} | "
            f"{multiple(row['compression_vs_conv_snn'])} | "
            f"{row['time_steps'] or '—'} | "
            f"{round(row['synaptic_events']) if row['synaptic_events'] is not None else '—'} | "
            f"{row['total_pulses'] if row['total_pulses'] is not None else '—'} | "
            f"{row['updated_physical_cells'] if row['updated_physical_cells'] is not None else '—'} | "
            f"{percent(row['saturation_low_rate'])}/{percent(row['saturation_high_rate'])} |"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    zh_lines = [
        "# PlantVillage 纯 SNN 硬件感知对比",
        "",
        "仅展示在模型选择完成后显式运行的最终测试结果。缺少实测能耗和阵列尺寸，因此不估算芯片能耗或面积。",
        "",
        "| 阶段 | 架构 | 训练相位 | 选择集准确率 | 选择集宏召回率 | 最终准确率 | 最终宏召回率 | 逻辑突触 | 差分器件单元 | 压缩倍数 | T | 每样本突触事件 | 写入脉冲 | 更新物理单元 | 低/高饱和率 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        zh_lines.append(
            f"| {row['stage']} | {architecture(row)} | {row['phase']} | "
            f"{percent(row['selection_top1_accuracy'])} | {percent(row['selection_macro_recall'])} | "
            f"{percent(row['final_top1_accuracy'])} | {percent(row['final_macro_recall'])} | "
            f"{row['logical_synapses'] or '—'} | {row['physical_cells'] or '—'} | "
            f"{multiple(row['compression_vs_conv_snn'])} | "
            f"{row['time_steps'] or '—'} | "
            f"{round(row['synaptic_events']) if row['synaptic_events'] is not None else '—'} | "
            f"{row['total_pulses'] if row['total_pulses'] is not None else '—'} | "
            f"{row['updated_physical_cells'] if row['updated_physical_cells'] is not None else '—'} | "
            f"{percent(row['saturation_low_rate'])}/{percent(row['saturation_high_rate'])} |"
        )
    (out / "comparison.zh-CN.md").write_text("\n".join(zh_lines) + "\n")
    return report
