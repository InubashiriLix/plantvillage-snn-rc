#!/usr/bin/env python3
"""Build the single current Chinese ConvSNN report and its PDF."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plantvillage-final-report-matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import tools.build_stage_report_assets as legacy  # noqa: E402


ARTIFACT = ROOT / "artifacts/hardware_refresh_2026-08-06"
REPORT = ROOT / "reports/final"
FIGURES = REPORT / "figures"
RUN_PATHS = {
    "all38_ideal": ARTIFACT / "retained/all38/metrics.json",
    "all38_hw": ARTIFACT / "final/conv_snn_all38_hw_finetune_seed42/metrics.json",
    "top10_ideal": ARTIFACT / "final/conv_snn_top10_ideal_seed42/metrics.json",
    "top10_hw": ARTIFACT / "final/conv_snn_top10_hw_finetune_seed42/metrics.json",
}


def load_runs():
    return {key: json.loads(path.read_text()) for key, path in RUN_PATHS.items()}


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def configure_legacy() -> None:
    legacy.REPORT_DIR = REPORT
    legacy.FIGURE_DIR = FIGURES
    legacy.ARTIFACT_ROOT = ARTIFACT
    legacy.RUNS = RUN_PATHS
    legacy.configure_plotting()


def plot_temporal_activity(runs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = {"all38_ideal": "#3A78B8", "all38_hw": "#E38B3D",
              "top10_ideal": "#25805A", "top10_hw": "#A64B8C"}
    labels = {"all38_ideal": "38 类理想", "all38_hw": "38 类硬件感知",
              "top10_ideal": "Top-10 理想", "top10_hw": "Top-10 硬件感知"}
    for key in colors:
        values = runs[key]["final_test_metrics"]["temporal_accuracy"]
        axes[0].plot(range(1, len(values) + 1), np.asarray(values) * 100, "o-",
                     label=labels[key], color=colors[key])
    axes[0].set_title("输出脉冲随时间累积后，准确率逐步提高")
    axes[0].set_xlabel("时间步")
    axes[0].set_ylabel("累计 Top-1 准确率（%）")
    axes[0].set_xticks(range(1, 9))
    axes[0].legend(fontsize=8)

    layers = list(runs["all38_hw"]["final_test_metrics"]["snn_activity"])
    x = np.arange(len(layers))
    width = 0.36
    for offset, key in ((-width / 2, "all38_hw"), (width / 2, "top10_hw")):
        rates = [runs[key]["final_test_metrics"]["snn_activity"][name]["firing_rate"] * 100
                 for name in layers]
        axes[1].bar(x + offset, rates, width, label=labels[key], color=colors[key])
    axes[1].set_xticks(x, layers)
    axes[1].set_ylabel("平均放电率（%）")
    axes[1].set_title("硬件感知模型的逐层脉冲活动")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    legacy.save(fig, "11_temporal_and_activity.png")


def plot_sample_extremes(runs) -> None:
    path = Path(runs["top10_hw"]["config"]["data_root"])
    csv_path = ARTIFACT / "final/conv_snn_top10_hw_finetune_seed42/records/final_predictions.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    correct = sorted((row for row in rows if row["correct"] == "1"),
                     key=lambda row: (-float(row["top1_margin"]), row["path"]))[:4]
    wrong = sorted((row for row in rows if row["correct"] == "0"),
                   key=lambda row: (-float(row["top1_margin"]), row["path"]))[:4]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    for row_index, selected in enumerate((correct, wrong)):
        for col, record in enumerate(selected):
            ax = axes[row_index, col]
            with Image.open(path / record["path"]) as image:
                ax.imshow(image.convert("RGB"))
            true_name = legacy.ZH_NAMES[record["true_class"]]
            predicted_name = legacy.ZH_NAMES[record["predicted_class"]]
            if row_index == 0:
                title = f"正确：{true_name}\n领先 {float(record['top1_margin']):.0f} 个脉冲"
                color = "#176B3A"
            else:
                title = f"真实：{true_name}\n误判：{predicted_name}\n领先 {float(record['top1_margin']):.0f} 个脉冲"
                color = "#A12828"
            ax.set_title(title, fontsize=8.5, color=color)
            ax.axis("off")
    fig.suptitle("Top-10 硬件感知模型的高置信正确与高置信错误样本", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    legacy.save(fig, "12_best_worst_samples.png")


def compress_dataset_grid() -> None:
    source = FIGURES / "01_dataset_examples_38.png"
    target = FIGURES / "01_dataset_examples_38.jpg"
    with Image.open(source) as image:
        image.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
        image.convert("RGB").save(target, quality=84, optimize=True, progressive=True)
    source.unlink()


def table_rows(runs) -> str:
    rows = []
    labels = (("all38_ideal", "38 类", "理想"), ("all38_hw", "38 类", "硬件感知"),
              ("top10_ideal", "Top-10", "理想"), ("top10_hw", "Top-10", "硬件感知"))
    for key, stage, phase in labels:
        run = runs[key]
        final = run["final_test_metrics"]
        pulse = (run.get("pulse_statistics") or {}).get("total_pulses", "—")
        rows.append(f"| {stage} | {phase} | {final['sample_count']:,} | {percent(final['top1_accuracy'])} | {percent(final['macro_recall'])} | {run['best_epoch']} | {pulse} |")
    return "\n".join(rows)


def weakest_rows(run, count=6) -> str:
    ranked = sorted(run["final_test_metrics"]["per_class"].items(),
                    key=lambda item: (item[1]["recall"], item[0]))[:count]
    return "\n".join(
        f"| {legacy.ZH_NAMES[name]} | {value['samples']} | {percent(value['recall'])} | {percent(value['precision'])} |"
        for name, value in ranked
    )


def write_report(runs) -> Path:
    ideal38 = runs["all38_ideal"]["final_test_metrics"]
    hw38 = runs["all38_hw"]["final_test_metrics"]
    idealt = runs["top10_ideal"]["final_test_metrics"]
    hwt = runs["top10_hw"]["final_test_metrics"]
    top10 = json.loads((ARTIFACT / "top10_classes.json").read_text())
    class_lines = "\n".join(
        f"{item['new_index'] + 1}. {legacy.ZH_NAMES[item['name']]}（原始索引 {item['original_index']}）"
        for item in sorted(top10["classes"], key=lambda item: item["new_index"])
    )
    markdown = f"""---
title: 基于实测器件电导数据的 ConvSNN 植物叶片分类
author: Xinrong Li · 2363123
date: 2026年8月6日
---

## 核心结论

端到端 ConvSNN 在 PlantVillage 38 类独立测试集上取得 {percent(ideal38['top1_accuracy'])} 准确率；引入实测电导曲线、差分电导映射和离散脉冲写入后，硬件感知准确率为 {percent(hw38['top1_accuracy'])}。依据 38 类硬件感知模型在选择验证集上的类别召回率重新选出 Top-10，并从头训练新的 10 输出模型。Top-10 理想与硬件感知准确率分别为 {percent(idealt['top1_accuracy'])} 和 {percent(hwt['top1_accuracy'])}。

硬件感知结果是使用实测电流—电导数据约束权重更新的仿真结果，不等同于芯片实测部署结果，也不用于推算缺少测量依据的能耗或面积。

![四组最终结果](figures/06_final_result_summary.png)

## 1. 数据集与评估协议

PlantVillage 包含健康叶片和多种植物病害。原训练目录共有 43,444 张图像，按固定随机种子分为 36,927 张训练图像和 6,517 张选择验证图像；原 `val` 目录的 10,861 张图像完整保留为 38 类最终测试集。Top-10 最终测试集包含 4,894 张图像。

选择验证集负责调参、早停、最佳 checkpoint 和 Top-10 排名。最终测试集只在全部选择冻结后使用。类别数量不均衡，因此同时报告总体准确率和对每个类别等权的宏召回率。

![38 类真实图像示例](figures/01_dataset_examples_38.jpg)

![类别分布](figures/02_class_distribution.png)

## 2. ConvSNN 与 LIF 时间计算

当前模型不是普通 CNN 编码器后接一个 SNN 分类头。三层卷积和两层全连接的每个可学习阶段均接入 LIF 脉冲神经元。RGB 像素作为第一层的直接输入电流，在 8 个时间步内重复输入；第一层 LIF 之后传递脉冲。输出类别由 8 步累计输出脉冲数最大者确定，每个批次开始前清空膜电位。

![ConvSNN 结构](figures/03_convsnn_architecture.png)

![LIF 与实验流程](figures/04_lif_and_experiment_pipeline.png)

## 3. 实测器件数据怎样进入训练

实测 CSV 在固定偏压下记录电流。电流除以偏压绝对值得到电导，相邻电导的上升与下降分别形成 LTP 和 LTD 经验更新曲线。LTP 可理解为器件电导逐步增强，LTD 可理解为电导逐步减弱。

单个器件只能表达正电导，因此一个有符号软件权重由 `G+` 和 `G-` 两个器件组成，权重正比于两者差值。反向传播给出目标变化后，查找表选择有限个 LTP/LTD 脉冲，更新实际电导，再同步回网络权重。卷积和全连接权重进入映射；偏置、批归一化状态及固定 LIF 参数仍保持理想数值。

![器件曲线与硬件更新](figures/10_device_mapping.png)

## 4. 训练与模型选择

38 类理想训练最多运行 100 epochs，实际运行 100 epochs，最佳 checkpoint 为 Epoch 82。新 Top-10 名单与历史名单相比替换了 3 个类别，因此没有复用旧模型，而是训练全新的 10 输出网络；实际运行 56 epochs，最佳 checkpoint 为 Epoch 36。

硬件搜索只读取训练集和选择验证集。两个任务最终均选择学习率 0.02、每次更新最多 4 个脉冲。38 类硬件模型最佳 Epoch 为 1，Top-10 硬件模型最佳 Epoch 为 3。

![训练曲线](figures/05_epoch_accuracy_loss.png)

## 5. 最终测试结果

| 任务 | 权重形式 | 测试样本 | Top-1 准确率 | 宏召回率 | 最佳 Epoch | 写入脉冲 |
|---|---|---:|---:|---:|---:|---:|
{table_rows(runs)}

38 类从理想到硬件感知下降 {(ideal38['top1_accuracy'] - hw38['top1_accuracy']) * 100:.2f} 个准确率百分点和 {(ideal38['macro_recall'] - hw38['macro_recall']) * 100:.2f} 个宏召回率百分点。Top-10 对应下降 {(idealt['top1_accuracy'] - hwt['top1_accuracy']) * 100:.2f} 和 {(idealt['macro_recall'] - hwt['macro_recall']) * 100:.2f} 个百分点。两组正式硬件模型的电导饱和率均接近零，性能下降主要来自初始电导映射、离散写入和非线性更新，而不是大量器件卡在上下限。

## 6. 38 类逐类表现

![38 类理想与硬件召回率](figures/08_all38_per_class_recall.png)

硬件模型表现最弱的类别如下。总体准确率明显高于宏召回率，说明少数困难类别仍会被总体数字掩盖。

| 类别 | 测试样本 | 召回率 | 精确率 |
|---|---:|---:|---:|
{weakest_rows(runs['all38_hw'])}

## 7. 新 Top-10 任务

新 Top-10 完全由 38 类硬件感知 selection recall 确定，并按 precision 和类别名称处理并列；final-test 不参与排序。

{class_lines}

![Top-10 逐类召回率](figures/07_top10_per_class_recall.png)

![Top-10 硬件混淆矩阵](figures/09_top10_confusion_matrix.png)

## 8. 时间行为、内部活动与样本

8 个时间步的累计准确率持续上升，说明最终预测利用了真实的时间累积过程。逐层放电率用于检查网络是否完全沉默或异常饱和，不保存全部原始激活张量。

![时间步准确率与放电活动](figures/11_temporal_and_activity.png)

下图按照输出脉冲领先差值选择高置信正确与高置信错误样本。“最好”和“最坏”描述模型置信行为，不代表图像质量。

![代表性样本](figures/12_best_worst_samples.png)

## 9. 限制与结论

当前输入为重复的 RGB 直接电流，不是事件相机异步脉冲。实测器件 CSV 缺少逐次光刺激同步日志，因此经验 LUT 保留了电导非线性和方向差异，但仍需后续器件实验确认真实脉冲条件。批归一化、偏置和 LIF 参数尚未映射到器件。

结果表明，延长训练显著改善了 38 类理想 ConvSNN；器件约束会造成可测量的性能损失，但新 Top-10 硬件模型仍保持 {percent(hwt['top1_accuracy'])} 最终准确率。后续重点应放在减小初始电导映射损失、改善困难类别和验证真实光感器件接口。
"""
    target = REPORT / "PlantVillage_ConvSNN_Final.zh-CN.md"
    target.write_text(markdown, encoding="utf-8")
    return target


def write_manifest(runs) -> None:
    manifest = {
        "version": 1,
        "current_artifact_root": str(ARTIFACT.resolve()),
        "source_metrics": {key: str(path.resolve()) for key, path in RUN_PATHS.items()},
        "figures": sorted(path.name for path in FIGURES.glob("*")),
        "report": "PlantVillage_ConvSNN_Final.zh-CN.md",
        "pdf": "PlantVillage_ConvSNN_Final.zh-CN.pdf",
    }
    (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def build_pdf(markdown: Path) -> Path:
    target = REPORT / "PlantVillage_ConvSNN_Final.zh-CN.pdf"
    subprocess.run([
        "pandoc", str(markdown.name), "--from", "markdown", "--pdf-engine=xelatex",
        "-V", "mainfont=Noto Sans CJK SC", "-V", "CJKmainfont=Noto Sans CJK SC",
        "-V", "papersize:a4", "-V", "geometry:margin=1.8cm", "-V", "fontsize=10pt",
        "--include-in-header=pdf-layout.tex", "-o", str(target.name),
    ], cwd=REPORT, check=True)
    return target


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_legacy()
    runs = load_runs()
    # Plot the full executed ideal histories, not only the prefix stored in best.pt.
    for run in (runs["all38_ideal"], runs["top10_ideal"]):
        if run.get("executed_history"):
            run["history"] = run["executed_history"]
    names = legacy.class_names()
    legacy.plot_dataset_overview(names)
    compress_dataset_grid()
    legacy.plot_class_distribution(names)
    legacy.plot_model_architecture()
    legacy.plot_lif_and_pipeline()
    legacy.plot_training_curves(runs)
    legacy.plot_result_summary(runs)
    legacy.plot_top10_recall(runs)
    legacy.plot_all38_recall(runs)
    legacy.plot_top10_confusion(runs)
    legacy.plot_device_mapping(runs)
    plot_temporal_activity(runs)
    plot_sample_extremes(runs)
    markdown = write_report(runs)
    write_manifest(runs)
    pdf = build_pdf(markdown)
    print(json.dumps({"markdown": str(markdown), "pdf": str(pdf),
                      "figures": len(list(FIGURES.glob("*")))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
