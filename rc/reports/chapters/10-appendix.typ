#import "../template.typ": *

#pagebreak()
= 附录

== A　主要程序入口

#academic-table(
  2,
  ([文件], [作用]),
  (
    [`01_preprocess_dataset.py`], [生成分层数据划分、空间分块、事件与纹理编码],
    [`02_run_reservoir.py`], [运行基础储备池状态仿真],
    [`03_train_readout.py`], [训练 v2 平衡 Ridge 读出],
    [`04_train_dual_pass.py`], [运行 v3 RGB—纹理双通路实验],
    [`06_train_spatial_v4.py`], [运行 v4 8×8 空间与检查点实验],
    [`07_train_dense_v5.py`], [运行 v5 连续事件与稠密状态实验],
    [`08_export_hardware_inputs.py`], [导出硬件 CSV/NPY、映射表和校验清单],
  ),
  widths: (58mm, 103mm),
)

== B　精选十分类映射

#academic-table(
  2,
  ([标签], [PlantVillage 类别目录名]),
  (
    [0], [`Blueberry___healthy`],
    [1], [`Cherry_(including_sour)___healthy`],
    [2], [`Corn_(maize)___Common_rust_`],
    [3], [`Corn_(maize)___healthy`],
    [4], [`Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`],
    [5], [`Orange___Haunglongbing_(Citrus_greening)`],
    [6], [`Peach___healthy`],
    [7], [`Soybean___healthy`],
    [8], [`Tomato___Tomato_Yellow_Leaf_Curl_Virus`],
    [9], [`Tomato___healthy`],
  ),
  widths: (25mm, 136mm),
)

== C　硬件通道映射

```text
列 1  R                 列 7  OFF_R
列 2  G                 列 8  OFF_G
列 3  B                 列 9  OFF_B
列 4  ON_R              列 10 luminance_std
列 5  ON_G              列 11 sobel_mean
列 6  ON_B              列 12 entropy_16bin
```

== D　报告数值溯源

#academic-table(
  2,
  ([内容], [本地事实来源]),
  (
    [38 类 v4 指标], [`artifacts/experiments_v4/v4_results.json`],
    [38 类 v5 指标], [`artifacts/experiments_v5/v5_results.json`],
    [十分类 v3 指标与消融], [`artifacts_best10/experiments_3x4_v3/v3_results.json`],
    [十分类 v5 指标], [`artifacts_best10/experiments_v5/v5_results.json`],
    [十分类逐类报告], [`artifacts_best10/experiments_v5/validation_classification_report.json`],
    [硬件文件与校验], [`artifacts_best10/hardware_export_v1/manifest.json`],
  ),
  widths: (56mm, 105mm),
)

#v(8pt)
#callout([复现注意], [
  报告中的版本指标与参数应从上述 JSON 重新提取。若未来实验覆盖这些文件，必须同步更新正文表格、摘要和结论，避免 README 与运行产物发生偏差。
], tone: "orange")
