#import "../template.typ": *
#import "../icons.typ": *

= 光电储备池模型与读出层

== 储备池计算原理

储备池计算将输入送入具有非线性和记忆的动态系统，收集其状态后仅训练读出层。与端到端训练全部循环连接不同，储备池内部参数可以保持固定，因而能够把物理器件固有的弛豫、非线性和响应差异直接转化为计算资源 @lukosevicius2009。已有工作表明，单个非线性延迟节点也能形成有效的虚拟储备池 @appeltant2011；更一般的物理实现覆盖光学、电子、自旋和机械等多类动态介质 @tanaka2019。物理 RC 的关键不是精确模拟某一种神经网络，而是保证输入可分离性、状态丰富度与适当的衰减记忆 @rifai2024。

== 12×12 动态阵列

每个时间步的 9 路或 12 路信号同时映射到阵列列方向，12 行承担不同动态响应。对第 $r$ 行、第 $j$ 列节点，可将一级仿真概括为

$ x_(r,j)(t) = alpha_r x_(r,j)(t-1) + (1-alpha_r) f(gamma m_(r,j) u_j(t)), $ <eq-state>

其中 $u_j(t)$ 为第 $j$ 个输入通道，$m_(r,j)$ 为固定掩码，$gamma$ 为输入增益，$f(·)$ 为非线性响应，$alpha_r$ 控制状态保持程度。较小 $alpha_r$ 强调当前输入，较大 $alpha_r$ 保留更长历史。项目使用一组行时间常数谱，而不是令全部节点具有完全相同的衰减。

#figure(
  block(width: 100%, fill: colors.green-soft, radius: 5pt, inset: 11pt)[
    #set par(first-line-indent: 0em)
    #grid(
      columns: (38mm, 1fr, 40mm),
      gutter: 12pt,
      align: center + horizon,
      block[
        #icon-label(event(size: 26pt), [12 路时序输入], [64 个连续时间步])
        #v(6pt)
        #text(size: 7.5pt, fill: colors.gray)[列方向：RGB / ON / OFF / 纹理]
      ],
      grid(
        columns: (5pt,) * 12,
        rows: (5pt,) * 12,
        gutter: 1.3pt,
        ..range(144).map(i => circle(radius: 2.5pt, fill: if calc.rem(i, 13) == 0 { colors.orange } else if calc.rem(i, 5) == 0 { colors.blue } else { colors.green.lighten(35%) }))
      ),
      block[
        #icon-label(readout(size: 26pt), [状态读出], [1、8 或 64 次采样])
        #v(6pt)
        #text(size: 7.5pt, fill: colors.gray)[行方向：异构时间常数谱]
      ],
    )
  ],
  caption: [12×12 动态阵列的输入、状态与读出关系],
)

== 状态采样策略

终态读取仅使用第 64 帧后的 12×9 或 12×12 状态，读取成本最低。八检查点策略在扫描过程中均匀采集八次完整状态，使读出层能够同时访问早期、中期和晚期响应。逐帧策略则收集 64×12×12 状态并展平为 9216 维向量，从而最大限度保留动态轨迹。

八检查点与逐帧读取并未增加光学输入帧数；其代价主要体现在电学读取次数、状态存储和读出层参数规模。真实器件部署时应结合采样带宽、噪声和能耗重新评估。

== 平衡 Ridge 读出

v2 至 v4 主要使用类别平衡 Ridge。设状态矩阵为 $X∈RR^(N×D)$、独热标签为 $Y∈RR^(N×K)$，样本权重矩阵为 $W$，则读出权重可写为

$ beta = (X^T W X + lambda I)^(-1) X^T W Y. $ <eq-ridge>

其中 $lambda$ 为正则化系数。Ridge 通过引入二次正则项改善非正交特征条件下估计的稳定性 @hoerl1970。类别权重用于减弱样本数量差异对决策边界的影响，验证阶段以宏召回率作为主要选择指标。

== 线性 Softmax 读出

v5 的 9216 维稠密状态使用线性 Softmax 分类器。对第 $k$ 类，预测概率为

$ p(y=k|x) = exp(w_k^T x+b_k) / sum_(j=1)^K exp(w_j^T x+b_j). $

模型仍保持单层线性读出，新增能力来自更完整的动态状态，而不是深层特征变换。训练采用类别平衡损失、权重衰减和早停；水平与垂直翻转仅在训练集上进行，并且只有验证宏召回率提升时才保留。

== 仿真与真实器件的对应关系

一级仿真提供可控的非线性、掩码、输入增益和指数衰减，用于验证系统级假设。真实光电阵列还会引入通道增益不一致、噪声、漂移、饱和、温度依赖以及读出误差。因此，仿真模型给出的准确率应被视为算法与接口可行性证据，而不是器件性能的直接测量。
