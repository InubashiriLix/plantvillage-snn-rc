# PlantVillage：ConvSNN 与 Reservoir Computing

本私有仓库将两条植物叶片分类路线整合到一个 `main` 分支：端到端训练的卷积脉冲网络，以及固定器件动力学、只训练线性读出的 RC。原工程保留在本地；不随仓库分发数据集和大型训练产物，不使用 Git LFS 或 Release。

| 路线与评估集 | 准确率 | 宏召回率 | 说明 |
|---|---:|---:|---|
| ConvSNN 38 类，理想，最终测试 | 90.61% | 87.48% | 8 步 LIF 网络 |
| ConvSNN 38 类，硬件感知，最终测试 | 81.19% | 75.94% | 实测电导曲线约束的仿真 |
| RC38 v5，验证集 | 79.91% | 77.23% | 未达 80% 宏召回门槛，尚无独立测试集结果 |
| RC10 v5，冻结测试集 | 97.02% | 96.80% | 从 38 类验证结果筛出的 Top10 |

RC10 是筛选后的类别，不能与完整 38 分类直接横向比较；两条路线的验证划分与选择流程也不同。硬件感知结果均不代表真实芯片部署测量。精确指标及来源见 [results](results/README.md)。

- [convsnn/](convsnn/README.md)：ConvSNN、紧凑 SNN、CNN 参考、器件映射与报告。
- [rc/](rc/README.md)：RC v1–v5、38 类演进、Top10、训练和导出脚本。
- [硬件交接示例](rc/hardware_example/README.md)：十类各一张的 640 帧 CSV、标签表及读取规范。
- [ConvSNN 报告](convsnn/reports/final/PlantVillage_ConvSNN_Final.zh-CN.pdf)、[RC 报告源文件](rc/reports/main.typ)、[RC 公式说明](rc/note.md)。

以下命令均从仓库根目录执行，Python 3.10+（CI 使用 3.11）。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './convsnn[dev]' -e ./rc
python -m pytest -q
python rc/scripts/09_read_hardware_preview.py
```

CUDA 版 PyTorch 需与机器驱动匹配。ConvSNN 的 `--device auto` 自动选择 CUDA，否则回退 CPU；RC v4/v5 默认 `--device cuda`，不可用时明确报错，可显式传 `--device cpu`。纯 CPU 安装可先执行 `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`。

将原始图像放到 `data/train/<类别>/` 和 `data/val/<类别>/`。`data/val` 保留为冻结测试集，验证集从 `data/train` 分层划分。最小训练与正式复现参数见各子项目 README。

```bash
# ConvSNN 38 类训练（产物含 metrics.json 和 checkpoint）
python -m plantvillage_snn train --data-root data --output-dir artifacts/convsnn \
  --device-csv convsnn/cnnRef/source/data.csv --device auto --evaluate-final
# RC38：8×8 patch，64 帧，12 通道
python rc/scripts/01_preprocess_dataset.py --patch-grid 8 8 --active-columns 12 \
  --output-dir artifacts/preprocessed_v4
python rc/scripts/07_train_dense_v5.py --device cuda
# RC10 固定类别、训练、完整 CSV/NPY 导出
python rc/scripts/01_preprocess_dataset.py --classes-file rc/configs/top10_classes.json \
  --patch-grid 8 8 --active-columns 12 --output-dir artifacts_best10/preprocessed_v4
python rc/scripts/07_train_dense_v5.py --device cuda \
  --input-dir artifacts_best10/preprocessed_v4 --output-dir artifacts_best10/experiments_v5
python rc/scripts/08_export_hardware_inputs.py
```

完整导出在 `artifacts_best10/hardware_export_v1/` 生成每个 split 的 9 路连续、9 路二值、12 路连续 CSV/NPY、标签、样本索引和校验清单。单个训练 CSV 约 213–241 MiB，全部受忽略规则保护。历史成绩作为已有实验记录保存，本次整合验证不重新训练完整数据集。

只维护并上传 `main`。提交文件使用明确路径；数据、权重、状态矩阵、缓存和归档文件均不提交。发布前运行 `python tools/check_repository.py`。
