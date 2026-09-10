#!/usr/bin/env python3
"""Build reproducible figures and sample-level evidence for the stage report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plantvillage-matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
import torch

from plantvillage_snn.config import ExperimentConfig
from plantvillage_snn.data import build_dataloaders
from plantvillage_snn.device import DeviceModel, HardwareState
from plantvillage_snn.models import build_model
from plantvillage_snn.train import load_checkpoint, ordered_class_names
from plantvillage_snn.utils import read_json, resolve_device, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "convsnn_reproduction"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
DATA_ROOT = ROOT / "data" / "1" / "PlantVillage"

RUNS = {
    "all38_ideal": ARTIFACT_ROOT / "final/baseline/conv_snn_all38_ideal_seed42/metrics.json",
    "all38_hw": ARTIFACT_ROOT / "final/baseline/conv_snn_all38_hw_finetune_seed42/metrics.json",
    "top10_ideal": ARTIFACT_ROOT / "final/baseline/conv_snn_top10_ideal_seed42/metrics.json",
    "top10_hw": ARTIFACT_ROOT / "final/baseline/conv_snn_top10_hw_finetune_seed42/metrics.json",
}

ZH_NAMES = {
    "Apple___Apple_scab": "苹果黑星病",
    "Apple___Black_rot": "苹果黑腐病",
    "Apple___Cedar_apple_rust": "苹果雪松锈病",
    "Apple___healthy": "健康苹果叶",
    "Blueberry___healthy": "健康蓝莓叶",
    "Cherry_(including_sour)___Powdery_mildew": "樱桃白粉病",
    "Cherry_(including_sour)___healthy": "健康樱桃叶",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "玉米灰斑病",
    "Corn_(maize)___Common_rust_": "玉米普通锈病",
    "Corn_(maize)___Northern_Leaf_Blight": "玉米北方叶枯病",
    "Corn_(maize)___healthy": "健康玉米叶",
    "Grape___Black_rot": "葡萄黑腐病",
    "Grape___Esca_(Black_Measles)": "葡萄黑麻疹",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "葡萄叶枯病",
    "Grape___healthy": "健康葡萄叶",
    "Orange___Haunglongbing_(Citrus_greening)": "柑橘黄龙病",
    "Peach___Bacterial_spot": "桃细菌性斑点病",
    "Peach___healthy": "健康桃叶",
    "Pepper,_bell___Bacterial_spot": "甜椒细菌性斑点病",
    "Pepper,_bell___healthy": "健康甜椒叶",
    "Potato___Early_blight": "马铃薯早疫病",
    "Potato___Late_blight": "马铃薯晚疫病",
    "Potato___healthy": "健康马铃薯叶",
    "Raspberry___healthy": "健康树莓叶",
    "Soybean___healthy": "健康大豆叶",
    "Squash___Powdery_mildew": "南瓜白粉病",
    "Strawberry___Leaf_scorch": "草莓叶焦病",
    "Strawberry___healthy": "健康草莓叶",
    "Tomato___Bacterial_spot": "番茄细菌性斑点病",
    "Tomato___Early_blight": "番茄早疫病",
    "Tomato___Late_blight": "番茄晚疫病",
    "Tomato___Leaf_Mold": "番茄叶霉病",
    "Tomato___Septoria_leaf_spot": "番茄斑枯病",
    "Tomato___Spider_mites Two-spotted_spider_mite": "番茄二斑叶螨",
    "Tomato___Target_Spot": "番茄靶斑病",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "番茄黄化曲叶病毒",
    "Tomato___Tomato_mosaic_virus": "番茄花叶病毒",
    "Tomato___healthy": "健康番茄叶",
}


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
    })


def save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, facecolor="white")
    plt.close(fig)


def load_runs() -> dict[str, dict[str, Any]]:
    return {name: read_json(path) for name, path in RUNS.items()}


def ordered_per_class(metrics: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(metrics["per_class"].items(), key=lambda item: item[1]["index"])


def class_names() -> list[str]:
    return sorted(path.name for path in (DATA_ROOT / "train").iterdir() if path.is_dir())


def count_images(directory: Path) -> int:
    return sum(1 for path in directory.iterdir() if path.is_file())


def plot_dataset_overview(names: list[str]) -> None:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(8, 5, figsize=(15, 21))
    for ax, name in zip(axes.flat, names):
        paths = sorted((DATA_ROOT / "train" / name).glob("*"))
        path = paths[int(rng.integers(0, len(paths)))]
        with Image.open(path) as source:
            image = source.convert("RGB")
        ax.imshow(image)
        ax.set_title(ZH_NAMES[name], fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("PlantVillage 38 类叶片图像示例（每类随机抽取 1 张，随机种子 42）", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "01_dataset_examples_38.png")


def plot_class_distribution(names: list[str]) -> None:
    train_counts = [count_images(DATA_ROOT / "train" / name) for name in names]
    final_counts = [count_images(DATA_ROOT / "val" / name) for name in names]
    y = np.arange(len(names))
    labels = [ZH_NAMES[name] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(15, 13), sharey=True)
    axes[0].barh(y, train_counts, color="#3A78B8")
    axes[0].set_title(f"原训练目录：{sum(train_counts):,} 张")
    axes[0].set_xlabel("图像数量")
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[1].barh(y, final_counts, color="#E38B3D")
    axes[1].set_title(f"独立最终测试目录：{sum(final_counts):,} 张")
    axes[1].set_xlabel("图像数量")
    fig.suptitle("PlantVillage 各类别样本数量并不均衡", fontsize=16)
    fig.tight_layout()
    save(fig, "02_class_distribution.png")


def plot_model_architecture() -> None:
    stages = [
        ("RGB 图像\n3×64×64", "#E8F1FA"),
        ("Conv 3×3\n16 通道", "#B9D8F4"),
        ("LIF + 池化\n产生脉冲", "#CBE8D3"),
        ("Conv 3×3\n32 通道", "#B9D8F4"),
        ("LIF + 池化\n产生脉冲", "#CBE8D3"),
        ("Conv 3×3\n64 通道", "#B9D8F4"),
        ("LIF + 池化\n产生脉冲", "#CBE8D3"),
        ("FC 128\n+ LIF", "#F3D49B"),
        ("输出层\n38 或 10 + LIF", "#EBA7A7"),
        ("8 步脉冲累加\n取最大类别", "#D8C7EC"),
    ]
    fig, ax = plt.subplots(figsize=(17, 4.3))
    ax.set_xlim(0, len(stages) * 1.7)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    for index, (label, color) in enumerate(stages):
        x = index * 1.7 + 0.15
        box = FancyBboxPatch((x, 1.25), 1.35, 1.1, boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor="#35556F", linewidth=1.3)
        ax.add_patch(box)
        ax.text(x + 0.675, 1.8, label, ha="center", va="center", fontsize=9.5)
        if index < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.36, 1.8), (x + 1.68, 1.8),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color="#4A5964", linewidth=1.2))
    ax.text(8.45, 3.1, "同一结构在每个时间步重复运行，膜电位在 8 步内持续积累",
            ha="center", va="center", fontsize=14, weight="bold")
    ax.text(8.45, 0.45, "蓝色：可学习卷积　绿色：脉冲神经元　黄色/红色：分类层",
            ha="center", va="center", fontsize=10, color="#4A5964")
    save(fig, "03_convsnn_architecture.png")


def plot_lif_and_pipeline() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    t = np.arange(1, 9)
    input_current = np.array([0.36, 0.28, 0.45, 0.22, 0.51, 0.31, 0.46, 0.35])
    beta, threshold = 0.85, 1.0
    membrane, spikes = [], []
    value = 0.0
    for current in input_current:
        value = beta * value + current
        spike = value >= threshold
        if spike:
            value -= threshold
        membrane.append(value)
        spikes.append(int(spike))
    axes[0].plot(t, membrane, "o-", color="#3A78B8", label="膜电位")
    axes[0].axhline(threshold, color="#D14B4B", linestyle="--", label="发放阈值")
    for step, spike in zip(t, spikes):
        if spike:
            axes[0].vlines(step, 0, threshold, color="#222222", linewidth=2)
    axes[0].set_title("LIF 神经元：积累、发放、复位")
    axes[0].set_xlabel("时间步")
    axes[0].set_ylabel("膜电位（示意值）")
    axes[0].set_xticks(t)
    axes[0].legend()

    axes[1].axis("off")
    steps = ["训练集\n参数更新", "选择验证集\n调参与早停", "38 类硬件模型\n按类别召回率排序",
             "全新 Top-10\n重新训练", "独立测试集\n只做最终评价"]
    colors = ["#B9D8F4", "#CBE8D3", "#F3D49B", "#D8C7EC", "#EBA7A7"]
    for i, (label, color) in enumerate(zip(steps, colors)):
        x = 0.02 + i * 0.195
        box = FancyBboxPatch((x, 0.36), 0.155, 0.30, transform=axes[1].transAxes,
                             boxstyle="round,pad=0.02", facecolor=color, edgecolor="#52616B")
        axes[1].add_patch(box)
        axes[1].text(x + 0.0775, 0.51, label, transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            axes[1].annotate("", xy=(x + 0.19, 0.51), xytext=(x + 0.16, 0.51),
                             xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#52616B"))
    axes[1].set_title("实验流程：最终测试集不参与模型选择", pad=18)
    fig.tight_layout()
    save(fig, "04_lif_and_experiment_pipeline.png")


def plot_training_curves(runs: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, (key, title) in enumerate((("all38_ideal", "38 类理想训练"),
                                        ("top10_ideal", "Top-10 理想训练"))):
        history = runs[key]["history"]
        epochs = [item["epoch"] for item in history]
        train_acc = [item["train_accuracy"] * 100 for item in history]
        selection_acc = [item["selection"]["top1_accuracy"] * 100 for item in history]
        macro = [item["selection"]["macro_recall"] * 100 for item in history]
        train_loss = [item["train_loss"] for item in history]
        selection_loss = [item["selection"]["loss"] for item in history]
        axes[row, 0].plot(epochs, train_acc, "o-", label="训练准确率")
        axes[row, 0].plot(epochs, selection_acc, "o-", label="选择集准确率")
        axes[row, 0].plot(epochs, macro, "o--", label="选择集宏召回率")
        axes[row, 0].set_title(f"{title}：准确率与宏召回率")
        axes[row, 0].set_ylabel("百分比（%）")
        axes[row, 0].set_ylim(0, 102)
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].plot(epochs, train_loss, "o-", label="训练损失")
        axes[row, 1].plot(epochs, selection_loss, "o-", label="选择集损失")
        axes[row, 1].set_title(f"{title}：损失")
        axes[row, 1].set_ylabel("损失值")
        axes[row, 1].legend(fontsize=8)
        for col in range(2):
            axes[row, col].set_xlabel("Epoch")
            axes[row, col].set_xticks(epochs)
    fig.suptitle("理想模型训练过程（最佳检查点保存时的完整历史）", fontsize=16)
    fig.tight_layout()
    save(fig, "05_epoch_accuracy_loss.png")


def plot_result_summary(runs: dict[str, dict[str, Any]]) -> None:
    labels = ["38类\n理想", "38类\n硬件感知", "Top-10\n理想", "Top-10\n硬件感知"]
    keys = ["all38_ideal", "all38_hw", "top10_ideal", "top10_hw"]
    accuracy = [runs[key]["final_test_metrics"]["top1_accuracy"] * 100 for key in keys]
    macro = [runs[key]["final_test_metrics"]["macro_recall"] * 100 for key in keys]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.36
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Top-1 准确率", color="#3A78B8")
    bars2 = ax.bar(x + width / 2, macro, width, label="宏召回率", color="#E38B3D")
    ax.bar_label(bars1, fmt="%.2f%%", padding=3)
    ax.bar_label(bars2, fmt="%.2f%%", padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("最终测试结果（%）")
    ax.set_title("理想权重与硬件感知权重的最终测试结果")
    ax.legend()
    save(fig, "06_final_result_summary.png")


def plot_top10_recall(runs: dict[str, dict[str, Any]]) -> None:
    ideal = ordered_per_class(runs["top10_ideal"]["final_test_metrics"])
    hardware = ordered_per_class(runs["top10_hw"]["final_test_metrics"])
    names = [name for name, _ in ideal]
    y = np.arange(len(names))
    height = 0.36
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(y - height / 2, [item["recall"] * 100 for _, item in ideal], height,
            label="理想", color="#3A78B8")
    ax.barh(y + height / 2, [item["recall"] * 100 for _, item in hardware], height,
            label="硬件感知", color="#E38B3D")
    ax.set_yticks(y, [ZH_NAMES[name] for name in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("类别召回率（%）")
    ax.set_title("Top-10 各类别最终测试召回率")
    ax.legend()
    save(fig, "07_top10_per_class_recall.png")


def plot_all38_recall(runs: dict[str, dict[str, Any]]) -> None:
    ideal = {name: item["recall"] * 100
             for name, item in ordered_per_class(runs["all38_ideal"]["final_test_metrics"])}
    hardware = {name: item["recall"] * 100
                for name, item in ordered_per_class(runs["all38_hw"]["final_test_metrics"])}
    names = sorted(hardware, key=hardware.get)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.scatter([ideal[name] for name in names], y, label="理想", s=33, color="#3A78B8")
    ax.scatter([hardware[name] for name in names], y, label="硬件感知", s=33, color="#E38B3D")
    for i, name in enumerate(names):
        ax.plot([ideal[name], hardware[name]], [i, i], color="#B9B9B9", linewidth=1)
    ax.set_yticks(y, [ZH_NAMES[name] for name in names], fontsize=8)
    ax.set_xlim(0, 103)
    ax.set_xlabel("类别召回率（%）")
    ax.set_title("38 类最终测试召回率：从硬件感知表现最差到最好排序")
    ax.legend()
    save(fig, "08_all38_per_class_recall.png")


def plot_top10_confusion(runs: dict[str, dict[str, Any]]) -> None:
    metrics = runs["top10_hw"]["final_test_metrics"]
    names = [name for name, _ in ordered_per_class(metrics)]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(normalized * 100, cmap="Blues", vmin=0, vmax=100)
    for row in range(len(names)):
        for col in range(len(names)):
            value = normalized[row, col] * 100
            if value >= 3:
                ax.text(col, row, f"{value:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if value > 55 else "black")
    labels = [ZH_NAMES[name] for name in names]
    ax.set_xticks(np.arange(len(names)), labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(names)), labels, fontsize=8)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("Top-10 硬件感知模型归一化混淆矩阵（%）")
    fig.colorbar(image, ax=ax, label="占该真实类别的比例（%）")
    fig.tight_layout()
    save(fig, "09_top10_confusion_matrix.png")


def plot_device_mapping(runs: dict[str, dict[str, Any]]) -> None:
    config = ExperimentConfig(**runs["all38_hw"]["config"])
    device = DeviceModel(config.device_csv, config.device_v_bias,
                         config.device_g_bins, config.device_max_pulses,
                         config.device_preprocess)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    pulse = np.arange(device.max_pulses + 1)
    starts = [0, device.n_g_bins // 2, device.n_g_bins - 1]
    for index in starts:
        axes[0].plot(pulse, device.ltp_curves[index] * 1e9,
                     label=f"LTP，初始 {device.g_centers[index] * 1e9:.1f} nS")
        axes[0].plot(pulse, device.ltd_curves[index] * 1e9, linestyle="--",
                     label=f"LTD，初始 {device.g_centers[index] * 1e9:.1f} nS")
    axes[0].set_title("由实测数据得到的脉冲—电导更新曲线")
    axes[0].set_xlabel("累计写入脉冲数")
    axes[0].set_ylabel("电导（nS）")
    axes[0].legend(fontsize=7)
    axes[1].axis("off")
    labels = ["软件梯度\n目标权重变化", "换算目标\nΔG+ / ΔG-", "LUT 选择\nLTP/LTD 脉冲", "更新实际电导\n并同步回权重"]
    colors = ["#B9D8F4", "#CBE8D3", "#F3D49B", "#EBA7A7"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = 0.03 + i * 0.245
        box = FancyBboxPatch((x, 0.37), 0.19, 0.27, transform=axes[1].transAxes,
                             boxstyle="round,pad=0.02", facecolor=color, edgecolor="#52616B")
        axes[1].add_patch(box)
        axes[1].text(x + 0.095, 0.505, label, transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=9)
        if i < len(labels) - 1:
            axes[1].annotate("", xy=(x + 0.24, 0.505), xytext=(x + 0.195, 0.505),
                             xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
    axes[1].set_title("硬件感知微调中的一次权重更新", pad=18)
    fig.tight_layout()
    save(fig, "10_device_mapping.png")


def _build_hw_model_and_loader(metrics: dict[str, Any]):
    config = ExperimentConfig(**metrics["config"]).with_overrides({
        "evaluate_final": True,
        "max_final_samples": None,
        "batch_size": 128,
        "num_workers": 4,
    })
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    data = build_dataloaders(config)
    names = ordered_class_names(data.class_to_index)
    model = build_model(config.model, len(names), config.time_steps, config.beta,
                        config.threshold, config.channels, config.layer_betas,
                        config.layer_thresholds).to(device)
    checkpoint = load_checkpoint(metrics["checkpoint"], device)
    model.load_state_dict(checkpoint["model_state"])
    empirical = DeviceModel(config.device_csv, config.device_v_bias,
                            config.device_g_bins, config.device_max_pulses,
                            config.device_preprocess)
    hardware = HardwareState(model, empirical, config.hw_scale_margin)
    hardware.load_state_dict(checkpoint["hardware_state"])
    hardware.sync_to_model(model)
    model.eval()
    return config, data, names, model, device


def infer_sample_extremes(metrics: dict[str, Any]) -> dict[str, Any]:
    config, data, names, model, device = _build_hw_model_and_loader(metrics)
    if data.final_test is None:
        raise RuntimeError("final-test loader missing")
    dataset = data.final_test.dataset
    records: list[dict[str, Any]] = []
    offset = 0
    with torch.inference_mode():
        for images, targets in data.final_test:
            model.reset_state()
            result = model(images.to(device))
            counts = result["spike_counts"].detach().cpu()
            predictions = counts.argmax(1)
            sorted_counts, _ = counts.sort(dim=1, descending=True)
            top_margin = sorted_counts[:, 0] - sorted_counts[:, 1]
            for i in range(len(targets)):
                target = int(targets[i])
                prediction = int(predictions[i])
                path = dataset.root / dataset.records[offset + i]["path"]
                records.append({
                    "path": str(path.resolve()),
                    "true_index": target,
                    "true_class": names[target],
                    "predicted_index": prediction,
                    "predicted_class": names[prediction],
                    "correct": target == prediction,
                    "predicted_spikes": float(counts[i, prediction]),
                    "true_spikes": float(counts[i, target]),
                    "top_margin": float(top_margin[i]),
                    "error_margin": float(counts[i, prediction] - counts[i, target]),
                })
            offset += len(targets)

    correct = sorted((r for r in records if r["correct"]),
                     key=lambda r: (-r["top_margin"], r["true_class"], r["path"]))
    wrong = sorted((r for r in records if not r["correct"]),
                   key=lambda r: (-r["error_margin"], -r["top_margin"], r["true_class"], r["path"]))
    observed_accuracy = len(correct) / len(records)
    expected_accuracy = float(metrics["final_test_metrics"]["top1_accuracy"])
    if not np.isclose(observed_accuracy, expected_accuracy, atol=1e-12):
        raise RuntimeError(
            f"reloaded checkpoint accuracy {observed_accuracy} differs from artifact {expected_accuracy}"
        )

    def diverse(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        chosen, seen = [], set()
        for record in candidates:
            if record["true_class"] in seen:
                continue
            chosen.append(record)
            seen.add(record["true_class"])
            if len(chosen) == count:
                break
        return chosen

    output = {
        "definition": {
            "best": "correct predictions with the largest top-1 minus top-2 output spike margin",
            "worst": "incorrect predictions with the largest predicted minus true-class spike margin",
            "diversity": "at most one displayed image per true class",
        },
        "model": "all38 hardware-aware ConvSNN",
        "time_steps": config.time_steps,
        "sample_count": len(records),
        "correct_count": len(correct),
        "observed_top1_accuracy": observed_accuracy,
        "expected_top1_accuracy": expected_accuracy,
        "best": diverse(correct, 6),
        "worst": diverse(wrong, 6),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "sample_extremes.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def plot_sample_extremes(extremes: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(18, 8))
    for row, key in enumerate(("best", "worst")):
        for col, record in enumerate(extremes[key]):
            ax = axes[row, col]
            with Image.open(record["path"]) as source:
                ax.imshow(source.convert("RGB"))
            if key == "best":
                title = (f"真/预测：{ZH_NAMES[record['true_class']]}\n"
                         f"领先：{record['top_margin']:.0f} 个脉冲")
                color = "#176B3A"
            else:
                title = (f"真实：{ZH_NAMES[record['true_class']]}\n"
                         f"误判：{ZH_NAMES[record['predicted_class']]}\n"
                         f"错误领先：{record['error_margin']:.0f} 个脉冲")
                color = "#A12828"
            ax.set_title(title, fontsize=8.5, color=color)
            ax.axis("off")
    axes[0, 0].set_ylabel("高把握度正确样本", fontsize=12, color="#176B3A")
    axes[1, 0].set_ylabel("高把握度错误样本", fontsize=12, color="#A12828")
    fig.suptitle("38 类硬件感知模型的代表性最好/最坏测试样本\n"
                 "“最好/最坏”指模型输出脉冲差值，不代表图像本身质量", fontsize=15)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.04, top=0.84, hspace=0.34, wspace=0.12)
    save(fig, "11_best_worst_samples.png")


def write_figure_manifest() -> None:
    manifest = {
        "version": 1,
        "source_artifacts": {key: str(path.resolve()) for key, path in RUNS.items()},
        "dataset_root": str(DATA_ROOT.resolve()),
        "sample_grid_seed": 42,
        "figures": sorted(path.name for path in FIGURE_DIR.glob("*.png")),
    }
    (REPORT_DIR / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-inference", action="store_true",
                        help="reuse reports/sample_extremes.json")
    args = parser.parse_args()
    configure_plotting()
    runs = load_runs()
    names = class_names()
    plot_dataset_overview(names)
    plot_class_distribution(names)
    plot_model_architecture()
    plot_lif_and_pipeline()
    plot_training_curves(runs)
    plot_result_summary(runs)
    plot_top10_recall(runs)
    plot_all38_recall(runs)
    plot_top10_confusion(runs)
    plot_device_mapping(runs)
    sample_path = REPORT_DIR / "sample_extremes.json"
    extremes = read_json(sample_path) if args.skip_inference and sample_path.is_file() else infer_sample_extremes(runs["all38_hw"])
    plot_sample_extremes(extremes)
    write_figure_manifest()
    print(f"generated {len(list(FIGURE_DIR.glob('*.png')))} figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
