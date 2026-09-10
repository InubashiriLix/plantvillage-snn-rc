#!/usr/bin/env python3
"""Visualize the 38-class long-epoch accuracy investigation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plantvillage-matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.build_stage_report_assets import ZH_NAMES, configure_plotting


OLD_METRICS = ROOT / "artifacts/convsnn_reproduction/final/baseline/conv_snn_all38_ideal_seed42/metrics.json"
NEW_METRICS = ROOT / "artifacts/accuracy_investigation/long_epoch/conv_snn_all38_ideal_seed42/metrics.json"
OUTPUT = ROOT / "reports/accuracy_investigation"
FIGURES = OUTPUT / "figures"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def ordered_classes(metrics: dict) -> list[tuple[str, dict]]:
    return sorted(metrics["final_test_metrics"]["per_class"].items(), key=lambda item: item[1]["index"])


def training_curves(new: dict) -> None:
    history = new["executed_history"]
    epoch = np.array([x["epoch"] for x in history])
    best = int(new["best_epoch"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    train_acc = np.array([x["train_accuracy"] for x in history]) * 100
    val_acc = np.array([x["selection"]["top1_accuracy"] for x in history]) * 100
    macro = np.array([x["selection"]["macro_recall"] for x in history]) * 100
    train_loss = np.array([x["train_loss"] for x in history])
    val_loss = np.array([x["selection"]["loss"] for x in history])

    axes[0, 0].plot(epoch, train_acc, label="训练准确率", linewidth=1.5)
    axes[0, 0].plot(epoch, val_acc, label="选择集准确率", linewidth=1.5)
    axes[0, 0].plot(epoch, macro, label="选择集宏召回率", linewidth=1.5)
    axes[0, 0].set_title("准确率与宏召回率")
    axes[0, 0].set_ylabel("百分比（%）")
    axes[0, 0].set_ylim(0, 102)
    axes[0, 0].legend()

    axes[0, 1].plot(epoch, train_loss, label="训练损失")
    axes[0, 1].plot(epoch, val_loss, label="选择集损失")
    axes[0, 1].set_title("训练损失与选择集损失")
    axes[0, 1].set_ylabel("损失")
    axes[0, 1].legend()

    gap = train_acc - val_acc
    axes[1, 0].plot(epoch, gap, color="#8E5BA6")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("训练准确率与选择集准确率的差距")
    axes[1, 0].set_ylabel("百分点")

    smooth = np.convolve(macro, np.ones(5) / 5, mode="valid")
    axes[1, 1].plot(epoch, macro, alpha=0.35, label="每个 epoch")
    axes[1, 1].plot(epoch[4:], smooth, linewidth=2, label="5-epoch 移动平均")
    axes[1, 1].set_title("宏召回率的波动与长期趋势")
    axes[1, 1].set_ylabel("宏召回率（%）")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.axvline(best, color="#D14B4B", linestyle="--", linewidth=1.2)
        ax.text(best + 1, ax.get_ylim()[1] * 0.95, "最佳 Epoch 82", color="#D14B4B", fontsize=8, va="top")
        ax.set_xlabel("Epoch")
    fig.suptitle("38 类 ConvSNN：完整 100 Epoch 训练过程", fontsize=17)
    fig.tight_layout()
    save(fig, "01_training_curves_100_epochs.png")


def overall_comparison(old: dict, new: dict) -> None:
    labels = ["选择集准确率", "选择集宏召回率", "最终测试准确率", "最终测试宏召回率"]
    old_values = [
        old["selection_metrics"]["top1_accuracy"], old["selection_metrics"]["macro_recall"],
        old["final_test_metrics"]["top1_accuracy"], old["final_test_metrics"]["macro_recall"],
    ]
    new_values = [
        new["selection_metrics"]["top1_accuracy"], new["selection_metrics"]["macro_recall"],
        new["final_test_metrics"]["top1_accuracy"], new["final_test_metrics"]["macro_recall"],
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    width = 0.36
    old_bars = ax.bar(x - width / 2, np.array(old_values) * 100, width, label="旧模型：最佳 Epoch 8", color="#8497B0")
    new_bars = ax.bar(x + width / 2, np.array(new_values) * 100, width, label="长训练：最佳 Epoch 82", color="#3A78B8")
    ax.bar_label(old_bars, fmt="%.2f%%", padding=3)
    ax.bar_label(new_bars, fmt="%.2f%%", padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("百分比（%）")
    ax.set_title("延长训练前后的总体结果")
    ax.legend()
    save(fig, "02_old_vs_new_overall.png")


def recall_dumbbell(old: dict, new: dict) -> None:
    old_pc = old["final_test_metrics"]["per_class"]
    new_pc = new["final_test_metrics"]["per_class"]
    names = sorted(new_pc, key=lambda name: new_pc[name]["recall"])
    y = np.arange(len(names))
    old_values = np.array([old_pc[name]["recall"] * 100 for name in names])
    new_values = np.array([new_pc[name]["recall"] * 100 for name in names])
    fig, ax = plt.subplots(figsize=(14, 14))
    for index in range(len(names)):
        color = "#77B77A" if new_values[index] >= old_values[index] else "#D66A6A"
        ax.plot([old_values[index], new_values[index]], [index, index], color=color, linewidth=1.4, alpha=0.75)
    ax.scatter(old_values, y, color="#8497B0", s=30, label="旧模型")
    ax.scatter(new_values, y, color="#245B91", s=34, label="长训练模型")
    ax.set_yticks(y, [ZH_NAMES[name] for name in names], fontsize=8)
    ax.set_xlim(0, 103)
    ax.set_xlabel("最终测试类别召回率（%）")
    ax.set_title("38 个类别：旧模型与长训练模型的召回率")
    ax.legend()
    save(fig, "03_per_class_recall_old_new.png")


def recall_improvements(old: dict, new: dict) -> None:
    old_pc = old["final_test_metrics"]["per_class"]
    new_pc = new["final_test_metrics"]["per_class"]
    names = sorted(new_pc, key=lambda name: new_pc[name]["recall"] - old_pc[name]["recall"])
    delta = np.array([(new_pc[name]["recall"] - old_pc[name]["recall"]) * 100 for name in names])
    colors = ["#3A78B8" if value >= 0 else "#D14B4B" for value in delta]
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.barh(np.arange(len(names)), delta, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(names)), [ZH_NAMES[name] for name in names], fontsize=8)
    ax.set_xlabel("召回率变化（百分点）")
    ax.set_title("延长训练对每个类别的影响")
    save(fig, "04_per_class_recall_improvement.png")


def confusion_matrix(new: dict) -> None:
    ordered = ordered_classes(new)
    names = [name for name, _ in ordered]
    matrix = np.asarray(new["final_test_metrics"]["confusion_matrix"], dtype=float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(17, 15))
    image = ax.imshow(normalized * 100, cmap="Blues", vmin=0, vmax=100)
    short = [ZH_NAMES[name] for name in names]
    ax.set_xticks(np.arange(len(names)), short, rotation=72, ha="right", fontsize=6)
    ax.set_yticks(np.arange(len(names)), short, fontsize=6)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("长训练模型：38 类归一化混淆矩阵")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label="占真实类别的比例（%）")
    fig.tight_layout()
    save(fig, "05_new_confusion_matrix.png")


def top_confusions(new: dict) -> None:
    ordered = ordered_classes(new)
    names = [name for name, _ in ordered]
    matrix = np.asarray(new["final_test_metrics"]["confusion_matrix"], dtype=float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    candidates = []
    for true_index in range(len(names)):
        for pred_index in range(len(names)):
            if true_index == pred_index or matrix[true_index, pred_index] == 0:
                continue
            candidates.append((normalized[true_index, pred_index] * 100, int(matrix[true_index, pred_index]),
                               names[true_index], names[pred_index]))
    top = sorted(candidates, reverse=True)[:15][::-1]
    labels = [f"{ZH_NAMES[true]} → {ZH_NAMES[pred]}" for _, _, true, pred in top]
    values = [fraction for fraction, _, _, _ in top]
    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(np.arange(len(top)), values, color="#D8794A")
    ax.set_yticks(np.arange(len(top)), labels, fontsize=8)
    ax.set_xlabel("占该真实类别的比例（%）")
    ax.set_title("长训练模型最明显的 15 组误判方向")
    for bar, (_, count, _, _) in zip(bars, top):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f"{count} 张", va="center", fontsize=8)
    save(fig, "06_top_confusions.png")


def temporal_accuracy(old: dict, new: dict) -> None:
    steps = np.arange(1, len(new["final_test_metrics"]["temporal_accuracy"]) + 1)
    old_values = np.asarray(old["final_test_metrics"]["temporal_accuracy"]) * 100
    new_values = np.asarray(new["final_test_metrics"]["temporal_accuracy"]) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, old_values, "o-", label="旧模型")
    ax.plot(steps, new_values, "o-", label="长训练模型")
    for step, value in zip(steps, new_values):
        ax.text(step, value + 2, f"{value:.1f}", ha="center", fontsize=8)
    ax.set_xticks(steps)
    ax.set_ylim(0, 100)
    ax.set_xlabel("累计时间步")
    ax.set_ylabel("最终测试准确率（%）")
    ax.set_title("输出脉冲随时间积累后的分类准确率")
    ax.legend()
    save(fig, "07_temporal_accuracy.png")


def activity(old: dict, new: dict) -> None:
    layers = ["conv1", "conv2", "conv3", "hidden", "output"]
    labels = ["Conv1", "Conv2", "Conv3", "隐藏层", "输出层"]
    old_activity = old["final_test_metrics"]["snn_activity"]
    new_activity = new["final_test_metrics"]["snn_activity"]
    x = np.arange(len(layers))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    width = 0.36
    axes[0].bar(x - width / 2, [old_activity[x]["firing_rate"] * 100 for x in layers], width, label="旧模型")
    axes[0].bar(x + width / 2, [new_activity[x]["firing_rate"] * 100 for x in layers], width, label="长训练模型")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("平均发放率（%）")
    axes[0].set_title("各层脉冲发放率")
    axes[0].legend()
    axes[1].bar(x - width / 2, [old_activity[x]["silent_neuron_rate"] * 100 for x in layers], width, label="旧模型")
    axes[1].bar(x + width / 2, [new_activity[x]["silent_neuron_rate"] * 100 for x in layers], width, label="长训练模型")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("静默神经元比例（%）")
    axes[1].set_title("各层静默神经元")
    axes[1].legend()
    fig.suptitle("训练时间对 SNN 内部活动的影响", fontsize=16)
    fig.tight_layout()
    save(fig, "08_snn_activity.png")


def samples_vs_recall(new: dict) -> None:
    per_class = new["final_test_metrics"]["per_class"]
    names = list(per_class)
    train_counts = np.asarray(new["sample_counts"]["train"])
    ordered = sorted(per_class.items(), key=lambda item: item[1]["index"])
    recall = np.asarray([item["recall"] * 100 for _, item in ordered])
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(train_counts, recall, s=45, alpha=0.75, color="#3A78B8")
    for index, (name, _) in enumerate(ordered):
        if recall[index] < 70 or train_counts[index] < 200:
            ax.annotate(ZH_NAMES[name], (train_counts[index], recall[index]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
    correlation = float(np.corrcoef(np.log1p(train_counts), recall)[0, 1])
    ax.set_xscale("log")
    ax.set_xlabel("训练样本数量（对数坐标）")
    ax.set_ylabel("最终测试召回率（%）")
    ax.set_title(f"训练样本数量与类别召回率（log-count Pearson r = {correlation:.2f}）")
    save(fig, "09_samples_vs_recall.png")


def write_overview(old: dict, new: dict) -> None:
    old_pc = old["final_test_metrics"]["per_class"]
    new_pc = new["final_test_metrics"]["per_class"]
    by_new = sorted(new_pc, key=lambda name: new_pc[name]["recall"])
    by_delta = sorted(new_pc, key=lambda name: new_pc[name]["recall"] - old_pc[name]["recall"], reverse=True)

    def class_rows(names: list[str], mode: str) -> list[str]:
        rows = []
        for name in names:
            old_recall = old_pc[name]["recall"]
            new_recall = new_pc[name]["recall"]
            delta = (new_recall - old_recall) * 100
            rows.append(f"| {ZH_NAMES[name]} | {old_recall * 100:.2f}% | {new_recall * 100:.2f}% | {delta:+.2f} |")
        return rows

    final_old = old["final_test_metrics"]
    final_new = new["final_test_metrics"]
    selection_old = old["selection_metrics"]
    selection_new = new["selection_metrics"]
    lines = [
        "---",
        "title: 38 类 ConvSNN 长训练结果可视化总览",
        "author: Xinrong Li · 2363123",
        "date: 2026年8月6日",
        "---", "",
        "## 核心结论", "",
        f"训练实际运行 {new['executed_epochs']} 个 epoch，最佳 checkpoint 位于 Epoch {new['best_epoch']}。",
        f"最终测试准确率从 {percent(final_old['top1_accuracy'])} 提升到 **{percent(final_new['top1_accuracy'])}**，",
        f"最终测试宏召回率从 {percent(final_old['macro_recall'])} 提升到 **{percent(final_new['macro_recall'])}**。",
        "Epoch 100 的选择集结果已经低于 Epoch 82，因此最终结果使用 Epoch 82 checkpoint。", "",
        "| 指标 | 旧模型 | 长训练模型 | 提升（百分点） |", "|---|---:|---:|---:|",
        f"| 选择集准确率 | {percent(selection_old['top1_accuracy'])} | {percent(selection_new['top1_accuracy'])} | {(selection_new['top1_accuracy'] - selection_old['top1_accuracy']) * 100:+.2f} |",
        f"| 选择集宏召回率 | {percent(selection_old['macro_recall'])} | {percent(selection_new['macro_recall'])} | {(selection_new['macro_recall'] - selection_old['macro_recall']) * 100:+.2f} |",
        f"| 最终测试准确率 | {percent(final_old['top1_accuracy'])} | {percent(final_new['top1_accuracy'])} | {(final_new['top1_accuracy'] - final_old['top1_accuracy']) * 100:+.2f} |",
        f"| 最终测试宏召回率 | {percent(final_old['macro_recall'])} | {percent(final_new['macro_recall'])} | {(final_new['macro_recall'] - final_old['macro_recall']) * 100:+.2f} |", "",
        "![完整训练曲线](figures/01_training_curves_100_epochs.png)", "",
        "![总体结果对比](figures/02_old_vs_new_overall.png)", "",
        "## 逐类别结果", "",
        "![逐类召回率](figures/03_per_class_recall_old_new.png)", "",
        "![逐类提升](figures/04_per_class_recall_improvement.png)", "",
        "### 提升最大的 10 个类别", "",
        "| 类别 | 旧召回率 | 新召回率 | 提升（百分点） |", "|---|---:|---:|---:|",
        *class_rows(by_delta[:10], "delta"), "",
        "### 新模型仍然最弱的 10 个类别", "",
        "| 类别 | 旧召回率 | 新召回率 | 提升（百分点） |", "|---|---:|---:|---:|",
        *class_rows(by_new[:10], "weak"), "",
        "## 混淆与误判", "",
        "![38类混淆矩阵](figures/05_new_confusion_matrix.png)", "",
        "![主要误判方向](figures/06_top_confusions.png)", "",
        "## SNN 时间行为与内部活动", "",
        "![时序准确率](figures/07_temporal_accuracy.png)", "",
        "![SNN活动](figures/08_snn_activity.png)", "",
        "## 样本数量与召回率", "",
        "![样本数量与召回率](figures/09_samples_vs_recall.png)", "",
        "## 数据来源", "",
        f"- 旧模型指标：`{OLD_METRICS.relative_to(ROOT)}`",
        f"- 长训练指标：`{NEW_METRICS.relative_to(ROOT)}`",
        f"- 长训练 checkpoint：`{Path(new['checkpoint']).relative_to(ROOT)}`",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "overview.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_plotting()
    old, new = load(OLD_METRICS), load(NEW_METRICS)
    if new.get("executed_epochs") != 100 or not new.get("final_test_metrics"):
        raise RuntimeError("long-epoch run is incomplete or has no frozen final evaluation")
    training_curves(new)
    overall_comparison(old, new)
    recall_dumbbell(old, new)
    recall_improvements(old, new)
    confusion_matrix(new)
    top_confusions(new)
    temporal_accuracy(old, new)
    activity(old, new)
    samples_vs_recall(new)
    write_overview(old, new)
    manifest = {
        "version": 1,
        "old_metrics": str(OLD_METRICS.resolve()),
        "new_metrics": str(NEW_METRICS.resolve()),
        "figures": sorted(path.name for path in FIGURES.glob("*.png")),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"generated {len(manifest['figures'])} figures and {OUTPUT / 'overview.zh-CN.md'}")


if __name__ == "__main__":
    main()
