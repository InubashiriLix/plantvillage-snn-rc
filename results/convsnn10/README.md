# SNN 十分类结果：论文绘图数据

这里是此前完成的 **ConvSNN Top10** 正式实验结果，含理想模型和硬件感知微调模型。所有表格从原始记录导出，未重新训练；两套模型使用相同的 10 类及 4,894 张冻结测试图片。

| 模型 | 测试准确率 | 测试宏召回率 | 最佳 epoch | 实际训练 epoch 数 |
|---|---:|---:|---:|---:|
| 理想 ConvSNN (`ideal`) | **97.6706%** | **96.0440%** | 36 | 56 |
| 硬件感知 ConvSNN (`hw_finetune`) | **93.8496%** | **91.3550%** | 3 | 6 |

硬件感知指实测电导曲线约束的仿真。Top10 根据完整 38 类硬件感知模型的 selection recall 排名筛选，不是自然随机抽取的十类，也不是 RC10 的类别集合；不能直接与 RC10 或完整 38 分类比较。这里只包含 seed=42 的已有实验，没有多次运行均值、标准差或误差条。

## 画图用哪些文件

| 文件 | 内容与适合的图 |
|---|---|
| [summary.csv](summary.csv) | 两模型 × 验证/测试，共 4 行；总体准确率和宏召回率柱状图 |
| [per_class_metrics.csv](per_class_metrics.csv) | 两模型 × 两 split × 十类，共 40 行；逐类 recall/precision 对比 |
| [confusion_matrix_long.csv](confusion_matrix_long.csv) | 两模型 × 两 split × 10×10，共 400 行；计数或按真实类别归一化的混淆矩阵 |
| [epoch_metrics.csv](epoch_metrics.csv) | 全部已执行 epoch，共 62 行；训练/验证准确率及 loss 曲线，`is_best_epoch` 标记最终采用的模型 |
| [temporal_accuracy.csv](temporal_accuracy.csv) | 两模型 × 两 split × 8 步，共 32 行；累计脉冲随时间步的分类准确率 |
| [class_mapping.csv](class_mapping.csv) | 类别名、十分类索引与原 38 类索引；所有矩阵按 `new_index=0..9` 排列 |
| [metrics.json](metrics.json) | 精确指标、层放电活动、模型开销、电导及脉冲统计 |
| [provenance.json](provenance.json) | 原始记录路径、SHA-256、导出文件校验值及核对方法 |

CSV 为 UTF-8，准确率、recall、precision 和 `row_fraction` 都是 **0–1 比例**，画百分比时乘 100。`split=selection` 是验证集（2,938 张），`split=final_test` 是冻结测试集（4,894 张）。论文测试结果图请筛选 `final_test`。

混淆矩阵中 **行是真实标签、列是预测标签**，`count` 是样本数，`row_fraction=count/该真实类别样本总数`。`per_class_metrics.csv` 中的 `samples` 是该类真实样本数，`correct` 是对角线计数。逐类“准确率”在本研究中表示 recall。

`epoch_metrics.csv` 的 `selection_accuracy`、`selection_macro_recall` 和 `selection_loss` 都来自验证集，没有逐 epoch 测试集曲线。硬件微调 epoch 从 1 重新计数，不应直接与理想训练拼成同一次连续训练。最佳模型之后的早停观察 epoch 也保留在表中；最终成绩来自 `is_best_epoch=True` 的模型。

`class_mapping.csv` 中的 `selection_recall/selection_precision` 是 **38 类源模型用于选类时**的分数，不能用作十分类模型的最终成绩。十分类成绩请用 `per_class_metrics.csv`。

## 读取示例

从仓库根目录，在已安装项目依赖的环境中运行：

```python
import pandas as pd

base = "results/convsnn10/"
summary = pd.read_csv(base + "summary.csv")
print(summary[summary["split"] == "final_test"])

long = pd.read_csv(base + "confusion_matrix_long.csv")
test = long[(long["phase"] == "hw_finetune") & (long["split"] == "final_test")]
matrix = test.pivot(index="true_index", columns="predicted_index", values="row_fraction")
labels = pd.read_csv(base + "class_mapping.csv").sort_values("new_index")["name"].tolist()
assert matrix.shape == (10, 10)
```

已有图可直接参考：[逐类召回率](../../convsnn/reports/final/figures/07_top10_per_class_recall.png)、[测试混淆矩阵](../../convsnn/reports/final/figures/09_top10_confusion_matrix.png)、[原终版报告](../../convsnn/reports/final/PlantVillage_ConvSNN_Final.zh-CN.pdf)。

## 来源与复核

源实验位于原 PlantVillage 工程的 `artifacts/hardware_refresh_2026-08-06/final/conv_snn_top10_{ideal,hw_finetune}_seed42/`。原始 `records/per_class_metrics.csv` 和 `records/confusion_matrix_long.csv` 仅含 selection，因此本目录的两 split 表格统一从冻结后的 `metrics.json` 重建，并与原始 `selection_predictions.csv` / `final_predictions.csv` 逐样本重算的矩阵核对。

四套矩阵的总样本数、对角线准确率、逐类 recall/precision、宏召回率均已核对。源记录 SHA-256 保存在来源清单，避免混用其他搜索候选模型。完整权重、原图、逐样本输出及训练中间状态不在此交付包中。

如需从已有原始实验重新导出，将 `ORIGINAL_PROJECT` 替换为原 PlantVillage 工程目录：

```bash
python tools/export_convsnn10_results.py --source-root ORIGINAL_PROJECT
python -m pytest -q convsnn/tests/test_published_top10.py
```

接收者只需下载本目录即可绘图，无需拿到训练数据或重跑模型。仓库为公开，可直接查看 GitHub 链接并下载绘图数据。
