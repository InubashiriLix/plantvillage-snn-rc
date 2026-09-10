#import "../template.typ": *
#import "../icons.typ": *

= 硬件接口与实验交接

== 数据交付内容

硬件数据包位于 `artifacts_best10/hardware_export_v1/`，同时提供便于人工检查的 CSV 和适合程序批量读取的 NPY。每个 NPY 的轴顺序固定为 `[sample, step, channel]`，CSV 每行对应一个硬件时间步。

#academic-table(
  5,
  ([输入版本], [列数], [事件形式], [主要用途], [复现十类最佳流程]),
  (
    [`inputs9_continuous`], [9], [连续幅值], [九列器件推荐接口], [否，缺少纹理],
    [`inputs9_binary_q080`], [9], [0/1], [只能输入开关脉冲的兼容接口], [否],
    [`inputs12_continuous`], [12], [连续幅值], [完整十二列器件接口], [是],
  ),
  widths: (42mm, 18mm, 27mm, 48mm, 27mm),
)

== 单样本播放协议

#figure(
  block(width: 100%, fill: colors.paper, stroke: .6pt + colors.line, radius: 6pt, inset: 10pt)[
    #set par(first-line-indent: 0em)
    #grid(
      columns: (1fr, 12pt, 1fr, 12pt, 1fr, 12pt, 1fr),
      align: center + horizon,
      block(fill: colors.blue-soft, radius: 4pt, inset: 8pt)[#align(center)[#chip() #linebreak() #text(size: 8pt, weight: 650)[① 恢复基线]]],
      text(fill: colors.gray)[→],
      block(fill: colors.green-soft, radius: 4pt, inset: 8pt)[#align(center)[#scan() #linebreak() #text(size: 8pt, weight: 650)[② 播放 1…64]]],
      text(fill: colors.gray)[→],
      block(fill: colors.orange-soft, radius: 4pt, inset: 8pt)[#align(center)[#reservoir() #linebreak() #text(size: 8pt, weight: 650)[③ 每帧读状态]]],
      text(fill: colors.gray)[→],
      block(fill: colors.blue-soft, radius: 4pt, inset: 8pt)[#align(center)[#readout() #linebreak() #text(size: 8pt, weight: 650)[④ 保存并分类]]],
    )
  ],
  caption: [单个硬件样本的标准执行时序],
)

每个样本必须从一致基线开始，64 帧内部保持连续动态。每帧的 9 路或 12 路信号同时施加至对应列，并在规定时刻读取全部阵列状态。完成一幅图像后保存状态轨迹，再复位并处理下一样本。

== 文件规模与对齐

#academic-table(
  4,
  ([Split], [样本数], [十二路 NPY shape], [光学帧总数]),
  (
    [train], [15,162], [[15162, 64, 12]], [970,368],
    [val], [3,791], [[3791, 64, 12]], [242,624],
    [test], [4,738], [[4738, 64, 12]], [303,232],
  ),
  widths: (30mm, 34mm, 58mm, 38mm),
)

标签数组第一维与输入第一维一一对应，`sample_index` 只在各 split 内有效；跨 split 唯一定位应使用 `(split, sample_index)` 或 `sample_id`。`manifest.json` 记录每个文件的 shape、字节数和 SHA-256，用于复制、传输和解压后的完整性验证。

== 硬件侧待标定参数

导出文件不规定真实器件的物理驱动量。以下参数需要通过实验确定：

- 归一化输入到电压、光功率、电流或脉宽的单调映射；
- 单帧持续时间和帧间隔；
- 每帧输入后状态读取的延迟；
- 样本间主动复位或自然衰减等待时间；
- 阵列通道增益、暗电流、漂移和坏点补偿；
- 状态量化位宽与采集噪声条件下的读出再训练方案。

#callout([首次联调建议], [
  先使用 `preview_10_samples.csv` 完成十个类别各一个样本的 64 帧时序检查；确认同步输入、列映射、采样时刻和样本间复位后，再切换到完整数据集。
])

== 从仿真到实测的闭环

真实器件实验应先采集训练、验证和测试样本的状态轨迹，再基于实测状态重新训练线性读出，而不是直接套用仿真读出权重。建议依次执行小样本连通性验证、重复性与漂移测试、输入范围标定、十分类训练、冻结测试评估，最后再扩展到 38 类。
