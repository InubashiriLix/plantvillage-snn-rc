# ConvSNN / CNN 与硬件感知实验

输入为缩放后的 `64×64 RGB` 图像。ConvSNN 将相同 RGB 张量作为直接电流输入重复 8 步，经过 `Conv(3→16) → Conv(16→32) → Conv(32→64) → FC(1024→128) → FC(128→类别数)`；三个卷积后有 BatchNorm，每个学习层后有 LIF，池化保留空间结构，累计输出脉冲决定类别。训练使用替代梯度、膜电位辅助损失和可选放电正则化。`compact_snn` 为可蒸馏的紧凑版本。

硬件阶段将有符号权重映射为差分电导 `G+−G−`，使用 [实测器件数据](cnnRef/source/data.csv) 拟合 LTP/LTD，通过 LUT 将更新转换为离散脉冲，再同步电导和网络权重。[cnnRef](cnnRef/README.md) 保留传统 CNN 与器件更新参考。

38 类最终测试：理想准确率 **90.61%**、宏召回率 **87.48%**；硬件感知准确率 **81.19%**、宏召回率 **75.94%**。见 [精确指标](../results/convsnn38/metrics.json) 与 [中文报告](reports/final/PlantVillage_ConvSNN_Final.zh-CN.md)。

所有命令从仓库根目录执行。环境安装见 [根 README](../README.md)。数据布局为 `data/{train,val}/<类别>/图片`；训练集按 seed=42 分出 15% selection，原 val 作为 final_test。

```bash
python -m plantvillage_snn --help
# 1. 理想训练，显式 100 epoch 上限；早停由 selection 决定
python -m plantvillage_snn train --data-root data --output-dir artifacts/convsnn \
  --model conv_snn --stage all38 --phase ideal --epochs 100 --time-steps 8 \
  --device auto --evaluate-final
# 2. 以第一步生成的模型进行硬件微调
python -m plantvillage_snn train --data-root data --output-dir artifacts/convsnn \
  --stage all38 --phase hw_finetune --device auto \
  --checkpoint artifacts/convsnn/conv_snn_all38_ideal_seed42/best.pt \
  --device-csv convsnn/cnnRef/source/data.csv --evaluate-final
# 3. 冻结模型评估，无需重新训练
python -m plantvillage_snn evaluate \
  --metrics artifacts/convsnn/conv_snn_all38_hw_finetune_seed42/metrics.json
# 含所有模型、阶段与硬件路径的 CPU smoke（测试生成临时图片）
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q convsnn/tests
```

上面的通用训练命令验证流程；报告中的准确率来自历史超参选择与硬件微调，不能保证用默认参数重训得到相同数字。历史胜出配置保存在 [configs](configs/README.md)，可通过 `train --config ...` 加载；其中硬件 checkpoint 需先用理想训练生成。

`--device auto` 有 CUDA 时选择 CUDA，否则 CPU；显式 `--device cuda` 在不可用时报错。CPU 最小试验可增加 `--epochs 1 --time-steps 2 --image-size 16 --max-train-samples 76 --max-selection-samples 38 --max-final-samples 38 --device cpu`，需要本地 38 类数据。

`tools/` 保留实验编排、CSV 导出和报告生成脚本；历史编排器默认引用旧实验目录，使用前须生成所依赖的 metrics/checkpoint。`cnnRef/main.py` 是历史 CNN 演示，在 `(cd convsnn/cnnRef && python main.py)` 下执行。它会下载 CIFAR-10 演示数据，不是 PlantVillage 成绩复现入口。

报告中的 `artifacts/` 路径表示历史本地产物，不随仓库上传。PDF 为原实验快照；Markdown、图片与来源清单均保留。只有整理后的指标进入 `results/`，模型、激活、梯度与完整 CSV 均在本地生成。
