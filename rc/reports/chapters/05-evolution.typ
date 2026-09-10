#import "../template.typ": *
#import "../icons.typ": *

= 方法演进：从 v2 到 v5

== 演进逻辑

项目迭代遵循“识别信息瓶颈，再改变一个决定性变量”的原则。v2 解决动态多样性，v3 增加互补纹理，v4 提高空间分辨率并增加中间状态读出，v5 在相同光学帧数下保留连续事件幅值与全部状态轨迹。

#figure(
  block(width: 100%, fill: colors.paper, stroke: .6pt + colors.line, radius: 6pt, inset: 10pt)[
    #set par(first-line-indent: 0em)
    #grid(
      columns: (1fr, 10pt, 1fr, 10pt, 1fr, 10pt, 1fr),
      align: center + horizon,
      block(fill: colors.green-soft, radius: 4pt, inset: 8pt)[
        #align(center)[#reservoir() #linebreak() #text(weight: 700)[v2] #linebreak() #text(size: 7.5pt)[异构动态]]
      ], text(fill: colors.gray)[→],
      block(fill: colors.blue-soft, radius: 4pt, inset: 8pt)[
        #align(center)[#texture() #linebreak() #text(weight: 700)[v3] #linebreak() #text(size: 7.5pt)[双通路纹理]]
      ], text(fill: colors.gray)[→],
      block(fill: colors.orange-soft, radius: 4pt, inset: 8pt)[
        #align(center)[#grid-icon() #linebreak() #text(weight: 700)[v4] #linebreak() #text(size: 7.5pt)[8×8 空间扫描]]
      ], text(fill: colors.gray)[→],
      block(fill: colors.green-soft, radius: 4pt, inset: 8pt)[
        #align(center)[#readout() #linebreak() #text(weight: 700)[v5] #linebreak() #text(size: 7.5pt)[逐帧稠密读取]]
      ],
    )
  ],
  caption: [四个主要版本的能力演进],
) <fig-evolution>

== v2：异构行动态

v2 仍使用 3×4 分块和单次扫描，但将统一的储备池衰减替换为行时间常数谱，并引入类别平衡的 SVD Ridge 读出。其开发意义在于确认：当各行具有不同记忆长度时，相同输入序列可以形成更丰富的动态投影。完整 38 类运行选择 $q=0.60$、$gamma=0.5$ 和 Ridge $lambda=10^(-4)$，为后续版本提供基线。

== v3：RGB—纹理双通路

v3 在 RGB 扫描之后增加 12 步纹理扫描，将局部对比度、Sobel 强度和熵编码到额外通路。RGB 尾状态与纹理尾状态拼接为 216 维特征。在完整 38 类验证任务上，宏召回率达到 62.09%，较 v2 提升 12.83 个百分点，说明纹理信息能够弥补单纯颜色变化的不足。

对于一级线性状态方程，当两个 pass 的尾状态同时进入线性读出时，携带状态和复位状态可能形成线性等价表示。十分类消融结果中，`dual_reset` 与 `dual_carry` 指标相同，表明“状态继承”本身不能被单独解释为性能来源；主要增益来自纹理通路及状态拼接。

== v4：8×8 空间扫描

v4 将分块从 3×4 提升至 8×8。虽然输入时间步由 12 增至 64，但与 4096 帧逐像素扫描相比仍降低 64 倍。它把三路纹理与九路 RGB/事件同时映射到 12 列，并比较终态与八检查点读取。

完整 38 类搜索表明，十二路八检查点方案明显优于只读终态：其验证宏召回率为 72.17%，相比 v3 提升 10.08 个百分点。结果说明空间分辨率和中间状态可见性是两项关键因素。

== v5：连续事件与逐帧稠密读取

v5 保持 64 个光学帧不变，将二值事件替换为归一化连续 ON/OFF 幅值，并在每帧后读取完整 12×12 状态。特征维数达到 9216，但分类器仍为线性 Softmax。完整 38 类验证宏召回率提升至 77.23%；验证准确率为 79.91%。

训练集翻转增强在完整任务上未改善宏召回率，因此按预设规则被拒绝。这一结果体现了项目的数据治理原则：增强策略不是默认保留，而是由独立验证指标决定。

== 版本代价对比

#academic-table(
  7,
  ([版本], [分块], [光学帧], [输入通路], [电读], [特征维数], [关键变化]),
  (
    [v2], [3×4], [12], [RGB+事件], [末态], [108], [异构衰减],
    [v3], [3×4 双 pass], [24], [RGB+事件+纹理], [双尾态], [216], [互补纹理],
    [v4], [8×8], [64], [12 路联合], [8], [1152], [空间与检查点],
    [v5], [8×8], [64], [12 路连续], [64], [9216], [完整状态轨迹],
  ),
  widths: (13mm, 24mm, 20mm, 31mm, 18mm, 24mm, 33mm),
)

#callout([设计结论], [
  本项目的主要性能提升并非来自更复杂的非线性分类器，而是来自输入信息保留和状态可观察性的增强。这一结论与物理储备池“固定动态系统 + 简单读出”的设计目标一致。
], tone: "blue")
