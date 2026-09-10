#import "../template.typ": *
#import "../icons.typ": *

= 系统总体设计

== 总体架构

系统由数据层、编码层、动态计算层、读出层和硬件交付层组成。敏感的类别划分、归一化尺度和模型选择均只由训练集或验证集确定，冻结测试集仅在模型达到预设门槛后使用。

#figure(
  block(width: 100%, fill: colors.paper, stroke: .6pt + colors.line, radius: 6pt, inset: 12pt)[
    #set par(first-line-indent: 0em)
    #grid(
      columns: (1fr, 12pt, 1fr, 12pt, 1fr, 12pt, 1fr, 12pt, 1fr),
      align: center + horizon,
      block(fill: colors.green-soft, stroke: .7pt + colors.green, radius: 4pt, inset: 7pt)[
        #align(center)[#leaf() #linebreak() #text(size: 8.5pt, weight: 650)[叶片图像] #linebreak() #text(size: 7pt, fill: colors.gray)[64×64 RGB]]
      ],
      text(fill: colors.gray)[→],
      block(fill: colors.blue-soft, stroke: .7pt + colors.blue, radius: 4pt, inset: 7pt)[
        #align(center)[#grid-icon() #linebreak() #text(size: 8.5pt, weight: 650)[空间编码] #linebreak() #text(size: 7pt, fill: colors.gray)[8×8 蛇形扫描]]
      ],
      text(fill: colors.gray)[→],
      block(fill: colors.orange-soft, stroke: .7pt + colors.orange, radius: 4pt, inset: 7pt)[
        #align(center)[#event() #linebreak() #text(size: 8.5pt, weight: 650)[12 路输入] #linebreak() #text(size: 7pt, fill: colors.gray)[RGB/事件/纹理]]
      ],
      text(fill: colors.gray)[→],
      block(fill: colors.green-soft, stroke: .7pt + colors.green, radius: 4pt, inset: 7pt)[
        #align(center)[#reservoir() #linebreak() #text(size: 8.5pt, weight: 650)[动态阵列] #linebreak() #text(size: 7pt, fill: colors.gray)[12×12 状态]]
      ],
      text(fill: colors.gray)[→],
      block(fill: colors.blue-soft, stroke: .7pt + colors.blue, radius: 4pt, inset: 7pt)[
        #align(center)[#readout() #linebreak() #text(size: 8.5pt, weight: 650)[线性读出] #linebreak() #text(size: 7pt, fill: colors.gray)[类别概率]]
      ],
    )
  ],
  caption: [植物叶片分类系统的端到端信息流],
) <fig-system>

== 分层职责与数据边界

#academic-table(
  4,
  ([层级], [主要输入], [核心操作], [输出]),
  (
    [数据层], [PlantVillage 图像与目录标签], [固定类别、分层划分、图像缩放], [训练/验证/测试样本],
    [编码层], [64×64 RGB 图像], [分块、扫描、相对差分与纹理统计], [64×9 或 64×12 序列],
    [动态层], [归一化通道序列], [掩码调制、非线性激励、衰减累积], [12×12 状态轨迹],
    [读出层], [终态、检查点或完整轨迹], [平衡 Ridge 或线性 Softmax], [类别分数与预测],
    [交付层], [固定编码与类别映射], [CSV/NPY 导出、校验和、协议描述], [硬件播放数据包],
  ),
  widths: (22mm, 38mm, 58mm, 37mm),
)

== 关键设计约束

系统设计受到物理阵列接口和实验可信度两方面约束。阵列具有 12 个输入列和 12 个响应行，因此每一帧最多同时施加 12 路信号；为了保留二维空间关系，8×8 patch 被转换为 64 个连续时间步，而不是 64 个并行端口。不同样本之间必须复位，但同一样本内部不能复位，否则储备池的短期记忆会被破坏。

另一方面，模型选择不能访问冻结测试集。项目将宏召回率 80% 设为阶段门槛：只有验证宏召回率达到门槛时才评估测试集。这一规则避免在多轮开发中通过测试集反复选择方案。

#callout([边界声明], [
  归一化数值 $[0,1]$ 只是逻辑输入。其到实际电压、光功率、电流或脉宽的映射，以及帧持续时间、采样时刻和复位时间，必须由真实器件标定确定。
], tone: "orange")

== 计算规模

8×8 路线每张图像使用 64 个光学帧，相比逐像素扫描 64×64 图像所需的 4096 帧，光学帧数减少 64 倍。不同状态读取方式对应不同的电读次数与特征维数：终态读取成本最低，但可能丢失早期状态；检查点读取在成本与信息量之间折中；逐帧读取保留完整动态轨迹，但形成 9216 维特征。

#academic-table(
  5,
  ([读取方式], [光学帧], [电读次数], [特征维数], [主要用途]),
  (
    [终态], [64], [1], [108/144], [最低读取开销基线],
    [八检查点], [64], [8], [864/1152], [v4 主路线],
    [逐帧稠密], [64], [64], [9216], [v5 最佳路线],
  ),
  widths: (30mm, 24mm, 25mm, 29mm, 45mm),
)
