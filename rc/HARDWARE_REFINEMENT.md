# 九路实测硬件第二轮优化

在第一轮 83.44% 的基础上补充响应分布特征、小样本神经网络和多模型融合。第一轮代码、缓存、指标均保留。本轮仍只使用 `N×9×12×65` 实测电流；原始光学输入、路径、类别名和样本编号不进入分类器。

## 方法

每条曲线前 64 点是规则刺激响应，最后的 tail 单独处理。状态补偿为首点不变，之后 `I[t]−rho×I[t−1]`，不把首点当作零基态，也不跨 tail 差分。单位统一为微安。

每条曲线的 18 项统计按顺序为：0%、10%、…、100% 的 11 个分位数，均值、总体标准差、相邻差分绝对值的均值、相邻差分总体标准差、首点、末规则点、tail。统计仅在单张图片内部计算。

| 表示 | 维数 | 内容 |
|---|---:|---|
| sequence | 576 | 补偿后取十二行均值，九路×64 点，不含 tail |
| distribution | 1944 | 原响应，九路×十二行×18 项统计 |
| compensated_distribution | 1944 | 补偿响应的同样统计，tail 保留原值 |
| mixed | 2520 | 原响应分布＋补偿后的均值序列 |
| multiscale | 4113 | 原响应分布 1944＋补偿分布 1944＋相对响应分位数 81＋蛇形 4×4 池化 144 |

相对响应按 RGB、ON、OFF 三组计算：每组各通道响应除以该组三通道绝对值之和再加 `0.05 μA`，取九个分位数。这是实测电流的相对表示，不等同于重建原始 RGB。蛇形池化先将扫描还原为 8×8，再平均相邻 2×2 区块。

神经网络：StandardScaler → PCA → 单隐层 64 单元 MLP，L-BFGS、最多 400 次迭代，比较 alpha 0.1/1/10、PCA 32/64、白化与否、ReLU/tanh。400 次上限可能产生未完全收敛警告，警告保存在本地候选日志中，不自动忽略失败或改参数。L-BFGS 是适合尝试的小样本求解器，参见 [scikit-learn 文档](https://scikit-learn.org/1.8/modules/neural_networks_supervised.html)。

树模型为 400 棵 RandomForest/ExtraTrees，最小叶节点 1，max_features 为 sqrt 或 0.3；SVM 使用标准化及 `gamma=scale`。完整 30 个配置以 [配置文件](configs/hardware_refinement_v2.json) 为准。

每个外层训练集内重新做三折预测：按准确率、宏 F1、固定编号排序，分别比较神经网络和树模型前 1/3/5 名的平均概率，再比较树权重 0.2/0.4/0.6/0.8 的融合与纯模型。按内层准确率、宏 F1、较少成员、编号打破并列。元数据和外层留出标签均不进入该折拟合或融合权重选择。

## 评估边界

本轮是**自适应探索**。先根据历史工作簿和第一轮结果设计特征，在原 `20260909` 第一外层折之外的 240 张上做了 51＋38 个配置的开发比较，再冻结 30 个候选。该开发子集包含其它外层折的验证样本，所以后续三次五折不能被称为完全嵌套、无偏或独立测试结果。每折内部的标准化、PCA、模型训练与融合选择仍只用该折训练数据。

固定复用三个历史种子的五折划分，全体 300 张、每类 30 张均保留。900 次相关预测不等于 900 张独立图片，三次准确率的样本标准差也不是独立测试置信区间。第一轮及历史探索的选择流程不同，比较只描述已有数据上的观察。

## 复现

先按 [基础安装与数据说明](HARDWARE300.md) 安装依赖。以下从仓库根目录执行，将目录改为收到原文件的位置。精确版本见结果中的 `search_protocol.json`。

```bash
export HARDWARE_DATA_DIR=/path/to/received/files
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python rc/scripts/11_refine_hardware.py evaluate \
  --responses "$HARDWARE_DATA_DIR/classification_features_7020.csv" \
  --inputs "$HARDWARE_DATA_DIR/train_10class_30each_rgb_onoff_q080_graded.csv" \
  --workbook "$HARDWARE_DATA_DIR/多模型优化与验证结果.xlsx" --workers 6

python rc/scripts/11_refine_hardware.py fit \
  --responses "$HARDWARE_DATA_DIR/classification_features_7020.csv" \
  --inputs "$HARDWARE_DATA_DIR/train_10class_30each_rgb_onoff_q080_graded.csv" \
  --workbook "$HARDWARE_DATA_DIR/多模型优化与验证结果.xlsx" --workers 6

python rc/scripts/10_optimize_hardware.py predict \
  --model artifacts_hardware300_v2/run/model.joblib \
  --responses "$HARDWARE_DATA_DIR/classification_features_7020.csv" \
  --output artifacts_hardware300_v2/run/delivery_predictions.csv
```

最后一条仅验证训练数据上的推理接口，不给出测试成绩。模型用全部 300 张训练；只加载自己生成的可信 joblib。预测文件支持相同列定义、不同样本数的新测量，标签列可省略。

开发过程可另行重算；它不是外层验证成绩：

```bash
for stage in 1 2; do
  python rc/scripts/11_refine_hardware.py screen \
    --responses "$HARDWARE_DATA_DIR/classification_features_7020.csv" \
    --inputs "$HARDWARE_DATA_DIR/train_10class_30each_rgb_onoff_q080_graded.csv" \
    --workbook "$HARDWARE_DATA_DIR/多模型优化与验证结果.xlsx" \
    --configs "rc/configs/hardware_development_stage${stage}.json" \
    --output-dir "artifacts_hardware300_v2/stage${stage}"
done

# 对比报告需要先按 HARDWARE300.md 重现第一轮，以取得逐样本配对结果及审计。
python tools/build_hardware_refinement_report.py
python -m pytest -q rc/tests/test_hardware_learning.py
```

`--hours` 按命令调用计时，在拟合之间检查；超时保留已完成缓存，不报告部分折均值。相同命令可恢复；数据、代码、依赖版本、候选或训练数组变更时拒绝复用。原文件、逐样本预测、缓存和模型均保存在被忽略的 `artifacts_hardware300_v2/`。

最终聚合结果、完整开发记录和 PNG/SVG 图见 [第二轮结果](../results/rc10_hardware300_v2/README.md)。确认泛化提升需要新增、记录采集批次的实测数据。

## 初始化扩展实验

第二轮结果出来后，另对六个神经网络配置各增加四个初始化种子（20260924–20260927），先在同样 240 张上做开发检查，再用总计 54 个候选完成相同的三次五折。它包含额外的自适应选择，不能当作独立复核。两套结果同时保留。

```bash
python rc/scripts/11_refine_hardware.py screen \
  --responses "$HARDWARE_DATA_DIR/classification_features_7020.csv" \
  --inputs "$HARDWARE_DATA_DIR/train_10class_30each_rgb_onoff_q080_graded.csv" \
  --workbook "$HARDWARE_DATA_DIR/多模型优化与验证结果.xlsx" \
  --configs rc/configs/hardware_development_seeds.json \
  --output-dir artifacts_hardware300_v2/seed_screen

for action in evaluate fit; do
  python rc/scripts/11_refine_hardware.py "$action" \
    --responses "$HARDWARE_DATA_DIR/classification_features_7020.csv" \
    --inputs "$HARDWARE_DATA_DIR/train_10class_30each_rgb_onoff_q080_graded.csv" \
    --workbook "$HARDWARE_DATA_DIR/多模型优化与验证结果.xlsx" \
    --configs rc/configs/hardware_refinement_seeds.json \
    --output-dir artifacts_hardware300_v2/seed_run
done
python tools/build_hardware_refinement_report.py --seed-run artifacts_hardware300_v2/seed_run
```
