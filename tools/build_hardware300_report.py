#!/usr/bin/env python3
"""Publish aggregate-only figures and tables from a complete hardware300 run."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rc"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plantvillage_rc.hardware_data import read_responses, write_csv, write_json


def build_report(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    summary = json.loads((run_dir / "evaluation_summary.json").read_text())
    audit = json.loads((run_dir / "audit.json").read_text())
    definition = json.loads((run_dir / "frozen_run.json").read_text())
    units = json.loads((run_dir / "unit_diagnostic.json").read_text())
    selection = json.loads((run_dir / "final_selection.json").read_text())
    if summary["status"] != "complete" or len(summary["repeats"]) != 3 or summary["prediction_count"] != 900:
        raise ValueError("publish only a complete 300-sample, three-repeat evaluation")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    # Raw input matrices, per-image identifiers/predictions and models stay local.
    public_summary = {key: value for key, value in summary.items() if key != "paired_predictions"}
    write_json(output / "summary.json", public_summary)
    write_json(output / "audit.json", {key: value for key, value in audit.items() if key != "historical_always_wrong_indices"})
    write_json(output / "search_protocol.json", definition)
    write_json(output / "delivery_model.json", selection)
    dependencies = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "matplotlib", "Pillow")
    (output / "requirements-reproduce.txt").write_text("\n".join(f"{name}=={version(name)}" for name in dependencies) + "\n")
    write_csv(output / "unit_diagnostic.csv", units)
    comparison = []
    matrices = {"historical": np.zeros((10, 10), dtype=int), "nested": np.zeros((10, 10), dtype=int)}
    paired = []
    for repeat in summary["repeats"]:
        for variant in ("historical", "nested"):
            values = repeat["historical"] if variant == "historical" else repeat
            comparison.append({"seed": repeat["seed"], "variant": variant,
                               **{key: values[key] for key in ("accuracy", "macro_f1", "macro_recall", "correct", "sample_count")}})
            key = "historical_confusion_matrix" if variant == "historical" else "confusion_matrix"
            matrices[variant] += np.asarray(repeat[key])
        paired.append({"seed": repeat["seed"], "corrected": repeat["corrected"], "regressed": repeat["regressed"],
                       "net_correct": repeat["corrected"] - repeat["regressed"]})
    write_csv(output / "repeat_metrics.csv", comparison)
    write_csv(output / "paired_changes.csv", paired)
    per_class, long = [], []
    for variant, matrix in matrices.items():
        for i, name in enumerate(audit["classes"]):
            support, predicted = int(matrix[i].sum()), int(matrix[:, i].sum())
            per_class.append({"variant": variant, "label": i, "class_name": name,
                              "unique_samples": 30, "prediction_count": support, "correct": int(matrix[i, i]),
                              "recall": float(matrix[i, i] / support),
                              "precision": float(matrix[i, i] / predicted) if predicted else 0.0})
            for j in range(10):
                long.append({"variant": variant, "true_label": i, "predicted_label": j,
                             "count": int(matrix[i, j]), "row_fraction": float(matrix[i, j] / support)})
    write_csv(output / "per_class_metrics.csv", per_class)
    write_csv(output / "confusion_matrix_long.csv", long)
    fold_selections = []
    for path in sorted((run_dir / "evaluation").glob("*/selection.json")):
        choice = json.loads(path.read_text())["selected"]
        fold_selections.append({"fold": path.parent.name, "members": "+".join(choice["members"]),
                                "weights": json.dumps(choice["weights"]), "inner_accuracy": choice["scores"]["accuracy"],
                                "inner_macro_f1": choice["scores"]["macro_f1"]})
    if len(fold_selections) != 15:
        raise ValueError("missing outer selection records")
    write_csv(output / "fold_selections.csv", fold_selections)
    # Error list with identifiers remains in the ignored local experiment directory.
    write_csv(run_dir / "paired_predictions.csv", summary["paired_predictions"])
    stable = {}
    for row in summary["paired_predictions"]:
        entry = stable.setdefault(row["sample_index"], {"sample_index": row["sample_index"], "sample_id": row["sample_id"],
                                                       "class_name": row["class_name"], "old_errors": 0, "new_errors": 0})
        entry["old_errors"] += row["historical_prediction"] != row["label"]
        entry["new_errors"] += row["prediction"] != row["label"]
    write_csv(run_dir / "stable_errors.csv", sorted(stable.values(), key=lambda row: (-row["new_errors"], row["sample_index"])))
    measured, metadata = read_responses(run_dir / "sources/responses.csv")
    mean_traces = measured[:, :, :, :64].mean(axis=2)
    row_dispersion = measured[:, :, :, :64].std(axis=2).mean(axis=(1, 2))
    amplitude = np.abs(mean_traces).mean(axis=(1, 2))
    quality_rows = []
    for i, meta in enumerate(metadata):
        entry = stable[int(meta["subset_sample_index"])]
        quality_rows.append({**entry, "mean_abs_current_uA": float(amplitude[i] * 1e6),
                             "mean_row_std_uA": float(row_dispersion[i] * 1e6),
                             "row_dispersion_to_amplitude": float(row_dispersion[i] / max(amplitude[i], 1e-15))})
    write_csv(run_dir / "response_quality.csv", quality_rows)
    diagnostic_groups = []
    for name, group in (("historical_always_wrong", [row for row in quality_rows if row["old_errors"] == 3]),
                        ("other_samples", [row for row in quality_rows if row["old_errors"] != 3])):
        diagnostic_groups.append({"group": name, "sample_count": len(group),
                                  "mean_current_uA": float(np.mean([row["mean_abs_current_uA"] for row in group])),
                                  "mean_row_dispersion_ratio": float(np.mean([row["row_dispersion_to_amplitude"] for row in group])),
                                  "new_always_wrong_count": sum(row["new_errors"] == 3 for row in group),
                                  "new_always_correct_count": sum(row["new_errors"] == 0 for row in group)})
    write_json(output / "error_diagnostics.json", {"groups": diagnostic_groups,
                                                 "note": "Descriptive association only; this is not evidence of a sensor defect or a mislabeled image."})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "hardware300"})
    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        svg = figures / f"{name}.svg"
        fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)

    positions = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for delta, variant, color, label in ((-0.19, "historical", "#a3aab4", "Historical exploratory ensemble"),
                                        (0.19, "nested", "#247d70", "New full nested selection")):
        values = [row["accuracy"] * 100 for row in comparison if row["variant"] == variant]
        bars = ax.bar(positions + delta, values, width=0.36, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(positions, [str(row["seed"]) for row in summary["repeats"]])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Fixed outer split seed (300 samples per repeat)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Hardware-only classification: complete OOF predictions")
    save(fig, "accuracy_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (variant, matrix) in zip(axes, matrices.items()):
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(10):
            for j in range(10):
                if matrix[i, j]:
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if normalized[i, j] > 0.55 else "black")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        ax.set_title(variant + " (counts across 3 repeats)")
    fig.colorbar(im, ax=axes, label="Row-normalized recall", shrink=0.85)
    fig.suptitle("900 correlated predictions from 300 unique samples")
    save(fig, "confusion_comparison")

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    y = np.arange(10)
    for offset, variant, color in ((-0.18, "historical", "#a3aab4"), (0.18, "nested", "#247d70")):
        values = [row["recall"] * 100 for row in per_class if row["variant"] == variant]
        ax.barh(y + offset, values, height=0.34, color=color, label=variant)
    ax.set_yticks(y, [f"{i}: {name}" for i, name in enumerate(audit["classes"])], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 105)
    ax.set_xlabel("Mean class recall over 3 repeats (%)")
    ax.set_title("30 unique samples per class"); ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "per_class_recall")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for offset, unit, color in ((-0.17, "A", "#a3aab4"), (0.17, "uA", "#247d70")):
        rows = [row for row in units if row["unit"] == unit]
        bars = axes[0].bar(np.arange(2) + offset, [row["accuracy"] * 100 for row in rows], 0.32, color=color, label=unit)
        axes[0].bar_label(bars, fmt="%.1f", padding=2)
    axes[0].set_xticks([0, 1], ["RandomForest", "ExtraTrees"])
    axes[0].set_ylim(0, 105); axes[0].set_ylabel("5-fold accuracy (%)")
    axes[0].set_title("Units only: fixed configuration, seed 20260909"); axes[0].legend()
    correlations = audit["mean_inter_row_correlations"]
    axes[1].bar(range(9), list(correlations.values()), color="#4d7198")
    axes[1].set_xticks(range(9), list(correlations), rotation=45, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("Mean pairwise Pearson correlation")
    axes[1].set_title("12 device rows: redundancy diagnostic")
    save(fig, "hardware_diagnostics")

    achieved = "观察到交叉验证均值提升" if summary["improved_mean"] else "未确认超过历史 90.44% 平均准确率"
    selected_text = " + ".join(selection["selected"]["members"])
    lines = [
        "# 九路真实硬件十分类：300 样本优化结果", "",
        f"**结论：{achieved}。** 新流程三次完整嵌套验证平均准确率 **{summary['mean_accuracy']:.2%}**，重复间样本标准差 **{summary['std_accuracy'] * 100:.2f} 个百分点**；历史探索结果为 **{summary['historical_mean_accuracy']:.2%}**，差值 **{summary['mean_accuracy_delta'] * 100:+.2f} 个百分点**。", "",
        "这里只使用九路实测电流响应，不使用原始 RGB/事件输入作为分类特征。每张为 9 通道 × 12 器件行 × (64 响应点 + tail)，共 7020 维；300 张、每类 30 张。", "",
        "## 完整验证结果", "", "| 外层划分种子 | 历史准确率 | 新嵌套准确率 | 新宏 F1 | 修正/退步样本数 |", "|---|---:|---:|---:|---:|",
    ]
    for row in summary["repeats"]:
        lines.append(f"| {row['seed']} | {row['historical']['accuracy']:.2%} | {row['accuracy']:.2%} | {row['macro_f1']:.2%} | {row['corrected']} / {row['regressed']} |")
    lines.extend([
        "", "历史模型已经在同一批数据上筛过 201 组配置；其嵌套复核仅涉及融合权重。新流程把特征、模型、参数与融合权重都纳入外层训练集内的三折选择。两者共享外层划分，但选择流程严格程度不同；历史成绩不能视为未参与选择的独立基线。", "",
        "每次评估 300 张，三次共 900 次相关预测，不是 900 个独立样本。标准差反映三次划分差异，不是独立测试置信区间。数据已参与此前探索，新的嵌套验证也不能将它变为独立测试集。未提供采集批次/时间信息，当前分层划分不能证明跨批次泛化。", "",
        "![准确率](figures/accuracy_comparison.png)", "![逐类召回](figures/per_class_recall.png)", "![混淆矩阵](figures/confusion_comparison.png)", "",
        "## 诊断与方法", "",
        "- 输入、硬件响应和历史预测的样本编号、标签与路径逐一核对；没有完全重复的响应样本或非有限电流值。",
        "- 正式训练统一用微安；单位对照只改变单位，保持特征、参数和划分不变。该对照不是模型选择后的正式成绩，且不能证明历史实现存在同一单位问题。",
        "- 12 行响应高度相关，因此比较完整响应、均值/标准差、中位数、状态补偿和蛇形空间池化。tail 独立保留，不跨 tail 做等间隔差分。",
        "- 六个模型族各 6 个固定候选，共 36 个；种子 20260922。任何标准化、PCA、模型拟合和融合权重选择均不使用外层留出标签。",
        "- 硬件输入仍为 RGB+ON+OFF 九路。新增统计或空间特征是电流响应的软件后处理，不是额外传感通道。", "",
        "![硬件诊断](figures/hardware_diagnostics.png)", "",
        "## 文件与使用", "",
        "- [每次重复的指标](repeat_metrics.csv)、[逐类指标](per_class_metrics.csv)、[混淆矩阵长表](confusion_matrix_long.csv)、[配对改善/退步计数](paired_changes.csv)。",
        "- [单位对照](unit_diagnostic.csv)、[十五个外层折的选择](fold_selections.csv)、[冻结搜索协议](search_protocol.json)、[完整汇总](summary.json)。",
        "- [数据审计和原文件 SHA-256](audit.json)、[全数据交付模型的配置](delivery_model.json)。所有准确率是 0–1 比例；混淆矩阵行为真实类、列为预测类。",
        "- [本次运行依赖版本](requirements-reproduce.txt)，Python 版本见冻结协议。重算结果时使用同一环境；新版本依赖的 smoke 通过不等于数值完全一致。",
        "- [持续误判样本的响应质量分组统计](error_diagnostics.json)。逐样本的配对变化和质量诊断仅保存在本地，统计差异不等于器件故障或标签错误。",
        "- `figures/` 同时提供 PNG 和可编辑 SVG。逐类表中的 `prediction_count=90` 表示每类 30 张重复评估三次。", "",
        f"交付模型在全部 300 张上重新做内层选择并训练：`{selected_text}`，权重 `{selection['selected']['weights']}`。这一全数据模型没有独立测试成绩，其内层分数不替代上面的外层结果。模型、原文件、完整逐样本预测和稳定误判清单保存在本地 `artifacts_hardware300/`，不推送公开仓库。", "",
        "运行方法见 [硬件训练说明](../../rc/HARDWARE300.md)。", "",
        "## 下一步", "",
        "优先安排新的、混排采集且记录批次的独立实测样本。固定交付模型后再测试；不要反复更换划分种子、删除难样本或挑最高单次成绩。若需要进一步开发特征或扩大搜索，应作为新一轮探索另行记录，不能覆盖本轮冻结协议。", "",
    ])
    if not summary["improved_mean"]:
        lines.extend(["本轮没有确认提升，不建议用该新交付模型替换已有历史融合方案。保留新模型用于复现和后续研究，同时保留历史 90.44% 的原始证据与其探索性限制。", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*")) if path.is_file() and path.name != "files_sha256.json"}
    write_json(output / "files_sha256.json", manifest)
    print(f"Published aggregate results: {output}; mean accuracy {summary['mean_accuracy']:.6%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts_hardware300")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rc10_hardware300")
    arguments = parser.parse_args()
    build_report(arguments.run_dir, arguments.output_dir)
