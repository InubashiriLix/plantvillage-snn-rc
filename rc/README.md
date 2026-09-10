# RGB-Tonic + ON/OFF Reservoir Computing

RC 用固定光电器件动力学把图像转换为状态，只训练线性读出。早期 v1/v2 将 `64×64 RGB` 切成 3×4 patch，经 12 步蛇形扫描形成 RGB、ON、OFF 共 9 路输入，驱动 12×9 节点；v3 加入继承状态的纹理扫描。原始设计与公式见 [note.md](note.md)，逐代成绩见 [实验历史](EXPERIMENT_HISTORY.md)。

v4/v5 将图像切成 8×8 patch 网格，每个 patch 内对 RGB 求均值，每张得到 **8×8×3=192 个 RGB 输入值**。按蛇形顺序展开为 64 帧；追加 RGB 增强、减弱各 3 路，再加对比度、Sobel 和熵 3 路，共 12 列。12 行各自有时间常数和固定掩码，构成 12×12 储备池。v4 在 8 个检查点读取，使用平衡 Ridge；v5 用连续 ON/OFF 幅值，每帧读取 144 个状态，形成 `64×144=9216` 维特征，由线性 Softmax 分类。

RC38 v5 最佳验证准确率 **79.91%**、宏召回率 **77.23%**。预设 80% 宏召回门槛未达到，未评估冻结测试集。RC10 类别由先前 RC38 验证报告筛选，固定列表见 [top10_classes.json](configs/top10_classes.json)。RC10 测试准确率 **97.02%**、宏召回率 **96.80%**，不能与完整 38 类直接比较，也不等同于 ConvSNN 筛出的 Top10。

所有命令从仓库根目录执行；安装依赖见 [根 README](../README.md)。默认数据 `data/train` 分层拆出验证集，`data/val` 为冻结测试集。缩放、阈值、归一化只使用训练集统计，验证集用于选参。

```bash
# v2 基线
python rc/scripts/01_preprocess_dataset.py
python rc/scripts/02_run_reservoir.py --device-profile row-spectrum
python rc/scripts/03_train_readout.py
# v3 双扫描
python rc/scripts/04_train_dual_pass.py
# v4/v5 38 类
python rc/scripts/01_preprocess_dataset.py --patch-grid 8 8 --active-columns 12 \
  --output-dir artifacts/preprocessed_v4
python rc/scripts/06_train_spatial_v4.py --device cuda
python rc/scripts/07_train_dense_v5.py --device cuda
# 固定 RC10
python rc/scripts/01_preprocess_dataset.py --classes-file rc/configs/top10_classes.json \
  --patch-grid 8 8 --active-columns 12 --output-dir artifacts_best10/preprocessed_v4
python rc/scripts/07_train_dense_v5.py --device cuda \
  --input-dir artifacts_best10/preprocessed_v4 --output-dir artifacts_best10/experiments_v5
python rc/scripts/08_export_hardware_inputs.py
# 无原始数据也可以运行的合成图片 smoke 与硬件校验
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q rc/tests
python rc/scripts/09_read_hardware_preview.py
```

v4/v5 读出训练默认 CUDA；不可用时报错并提示 `--device cpu`，不会悄悄切换设备。状态提取本身使用 NumPy、分批 memmap。`--quick` 缩小搜索范围；v5 的 `--no-augmentation` 禁用翻转试验。`05_cnn_upper_bound.py` 为 ImageNet MobileNetV3 上界检查，会下载预训练权重，成绩不属于 RC。

`01` 保存预处理配置及 NPZ；`02` 保存状态；`03/04/06/07` 保存指标、读出模型和中间特征。v3–v5 仅达到门槛才评估冻结测试；不应为拿到测试分数而修改门槛。`08` 输出完整 CSV/NPY 和 SHA-256 清单，见 [硬件协议](hardware_example/README.md)。默认产物全部位于根目录 `artifacts*`，不提交 Git。显式相对 CLI 路径按当前工作目录解析，默认路径绑定仓库根目录。

[中文报告源文件](reports/main.typ) 与 [历史 PDF](reports/main.pdf) 保留已有研究内容。PDF 是迁移前快照，当前命令以本 README 为准。重新编译：`typst compile rc/reports/main.typ rc/reports/main.pdf`（需 Noto CJK 字体）。
