# 九路真实硬件：300 样本训练与验证

该流程与模拟 RC10 分开：输入为已经测得的电流响应，学习器只接触硬件电流。原始 RGB＋ON＋OFF CSV 用于核对样本、标签和扫描顺序，不作为分类特征。固定十类、每类 30 张，不重新筛选类别或删除难样本。

## 安装与数据接口

从仓库根目录安装 `python -m pip install -e './convsnn[dev]' -e ./rc`。本流程使用现有 NumPy、SciPy、scikit-learn、joblib 和 Matplotlib，不依赖 Excel 应用或 GPU。工作簿只解析静态单元格值，不执行公式。

- `classification_features_7020.csv`：300 行，5 个元数据列和 7020 个实测响应列。
- `train_10class_30each_rgb_onoff_q080_graded.csv`：19200 行，300 张 × 64 帧，每帧九路输入。
- `多模型优化与验证结果.xlsx`：`推荐模型预测` 工作表保存三次五折的历史样本预测和外层折号。

硬件列按名称解析，例如 `R_row01_step01_A`、`OFF-blue_row12_tail_A`，列的物理存放顺序可变化。内部形状固定为 `[sample, channel, device_row, response_point] = [N,9,12,65]`，通道次序为 R、G、B、ON-red、ON-green、ON-blue、OFF-red、OFF-green、OFF-blue。

单位为安培，模型特征统一转换为微安并保留符号。前 64 点是刺激步骤，末点是单独的 tail；文件未定义 tail 的采样间隔，因此不把它纳入等间隔差分。没有独立基线点，首响应不能被当作零基态。

元数据列为 `subset_sample_index,label_index,class_name,source_sample_index,sample_id`。训练时与原始输入和历史预测逐一核对；元数据不进入特征。预测时只要求样本编号和 sample_id，其余标签信息可省略。

## 可复现命令

将 `HARDWARE_DATA_DIR` 改成收到文件的目录。以下默认使用同一个被 Git 忽略的 `artifacts_hardware300/` 输出目录：

```bash
export HARDWARE_DATA_DIR=/path/to/received/files
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 审计与单位对照；原文件复制到本地 sources/，并记录 SHA-256
python rc/scripts/10_optimize_hardware.py audit \
  --responses "$HARDWARE_DATA_DIR/classification_features_7020.csv" \
  --inputs "$HARDWARE_DATA_DIR/train_10class_30each_rgb_onoff_q080_graded.csv" \
  --workbook "$HARDWARE_DATA_DIR/多模型优化与验证结果.xlsx" --unit-diagnostic

# 三次外层五折；每个外层训练集再进行三折选择
python -u rc/scripts/10_optimize_hardware.py evaluate \
  --responses artifacts_hardware300/sources/responses.csv \
  --inputs artifacts_hardware300/sources/inputs.csv \
  --workbook artifacts_hardware300/sources/workbook.xlsx --workers 6 --hours 7

# 外层评估完成后，全数据内层选择并训练交付模型
python rc/scripts/10_optimize_hardware.py fit \
  --responses artifacts_hardware300/sources/responses.csv \
  --inputs artifacts_hardware300/sources/inputs.csv \
  --workbook artifacts_hardware300/sources/workbook.xlsx --workers 6 --hours 1

# 使用自己训练的可信本地模型预测新的硬件响应
python rc/scripts/10_optimize_hardware.py predict \
  --model artifacts_hardware300/model.joblib \
  --responses artifacts_hardware300/sources/responses.csv \
  --output artifacts_hardware300/delivery_predictions.csv

# 导出不含原始电流或逐样本标识的汇总与 PNG/SVG 图
python tools/build_hardware300_report.py
python -m pytest -q rc/tests/test_hardware_learning.py
```

预测示例使用训练数据仅检查接口，不是独立测试。`model.joblib` 包含预处理、类别顺序和最终读出，使用 joblib 反序列化时只加载自己生成的可信文件。

## 特征、模型与选择

特征操作均确定且逐样本完成；需要学习的标准化和 PCA 在每个训练折内部拟合。

| 特征 | 维数 | 定义 |
|---|---:|---|
| full | 7020 | 九通道、十二行、64 点和 tail 完整展开 |
| mean_std | 1170 | 十二行的均值与总体标准差，各 9×65 |
| median | 585 | 十二行中位数，9×65 |
| compensated_090/098/100 | 1170 | 首点不变，后续 63 点做 I[t]−ρI[t−1]，tail 不变，再取行均值/标准差 |
| spatial | 1530 | 均值/标准差的 64 步还原蛇形 8×8，取 2×2、4×4、8×8 池化，再追加 18 个 tail 值 |

固定种子 `20260922`，六个模型族各抽取六个候选，共 36 个。完整名单在运行前写入 `frozen_run.json`；不会看到外层结果后继续调参。

- Ridge：标准化，alpha 在 0.001–100 的离散网格中选择。
- RBF-SVM：标准化，可选 32/64 维 PCA；C 为 0.1/1/10/100，gamma 为 scale/0.001/0.01/0.1，概率估计仅用训练折。
- RandomForest / ExtraTrees：400 棵树，叶节点最小样本数 1/2/4，max_features 为 sqrt/0.3/1.0。
- HistGradientBoosting：最多 200 次迭代、32 bins；学习率 0.03/0.1，叶数 7/15/31，L2 0.1/1/10，训练折内部早停。
- MLP：标准化，可选 32/64 维 PCA；隐层 64 或 128→32，alpha 0.01/0.1/1，最多 300 次迭代，训练折内部 15% 早停集、耐心值 25。

内层按准确率、宏 F1 排序；候选并列时优先较低输入维度，再按固定候选编号确定。最佳树模型和 MLP 的概率按 MLP 权重 0/0.2/0.4/0.6/0.8/1 比较；与最佳单模型成绩完全并列时保留单模型。Ridge 的决策分数经 softmax 形成归一化输出，不将其解释为校准概率。

## 验证边界与运行恢复

外层严格复用工作簿中 `20260909/20260910/20260911` 的折号，每折 60 张、每类 6 张。每个外层训练集的内层采用三折分层划分，随机种子为外层重复种子加零基折号。最终交付模型只在全部 300 张上重新做三折选择，种子为 `20260922`。

完整报告必须包含 15 个外层折，每张图片每次重复只预测一次。三次评估不能当作 900 个独立样本。当前数据已经参与旧模型家族筛选，因此新结果仍然是同一批数据的探索，独立泛化需新增实测数据。没有采集批次信息，不假定 CSV 中按类别排序就是物理采集顺序。

每个候选完成全部内层折后缓存，外层选择先落盘，再计算外层预测。数据哈希、代码哈希、依赖版本、完整候选及训练数组哈希用于校验恢复；任何不一致都拒绝复用旧缓存，需指定新的 `--output-dir`。

`--hours` 是本次命令的计算预算，按拟合之间的边界检查，正在进行的一次拟合会结束后才停止。超时退出码为 2，进度标记 partial，不输出完整平均成绩；重复相同命令可以恢复，新的调用预算应计入总工作时间。`progress.json` 给出已完成外层折数，日志包含候选运行时间和选择；降低 `--workers` 可减少资源占用而不改变候选集合。

## 数据与公开交付

原始电流、输入、工作簿副本、完整 OOF 预测、稳定误判列表、缓存和模型都位于 `artifacts_hardware300/`，不进入公开 Git。公开的结果是聚合指标、冻结搜索参数、哈希和绘图，不包含逐图路径。历史原始优化代码没有提供，所以这里只核验它的预测记录，不将新基线称为历史模型的精确重训。

已有完整结果见 [真实硬件 300 样本汇总](../results/rc10_hardware300/README.md)。其成绩不能直接对比 [模拟 RC10](../results/rc10/v5_results.json) 的 97.02%，也不能当作 [SNN Top10](../results/convsnn10/README.md) 的同一实验。
