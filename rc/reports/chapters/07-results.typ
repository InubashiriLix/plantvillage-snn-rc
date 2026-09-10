#import "../template.typ": *
#import "../icons.typ": *

= 实验结果与分析

== 完整 38 类任务

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  metric([v3 验证宏召回率], [62.09%], note: [RGB—纹理双通路]),
  metric([v4 验证宏召回率], [72.17%], note: [8×8 + 八检查点], tone: "blue"),
  metric([v5 验证宏召回率], [77.23%], note: [连续事件 + 稠密读取], tone: "orange"),
)

#v(10pt)
#figure(
  block(width: 100%, fill: colors.paper, stroke: .6pt + colors.line, radius: 5pt, inset: 11pt)[
    #set par(first-line-indent: 0em)
    #bar-row([v2 单通路], 0.4925878)
    #v(5pt)
    #bar-row([v3 双通路], 0.6209266, color: colors.blue)
    #v(5pt)
    #bar-row([v4 八检查点], 0.7216787, color: colors.orange)
    #v(5pt)
    #bar-row([v5 稠密读取], 0.7722612, color: colors.green)
    #v(7pt)
    #line(length: 100%, stroke: .4pt + colors.line)
    #v(5pt)
    #text(size: 7.8pt, fill: colors.gray)[主指标为验证集宏召回率；v3 至 v5 未访问冻结测试集。]
  ],
  caption: [完整 38 类任务的验证宏召回率演进],
) <fig-full-results>

#academic-table(
  6,
  ([模型], [验证准确率], [验证宏召回率], [预测类别], [相对前版增益], [测试状态]),
  (
    [v2], [49.19%], [49.26%], [38], [基线], [曾评估],
    [v3], [61.96%], [62.09%], [38], [+12.83 pp], [未评估],
    [v4], [72.02%], [72.17%], [38], [+10.08 pp], [未评估],
    [v5], [79.91%], [77.23%], [38], [+5.06 pp], [未评估],
  ),
  widths: (18mm, 29mm, 32mm, 24mm, 29mm, 27mm),
)
#source-note([`artifacts/experiments_v2/primary_selection.json`、`experiments_v3/v3_results.json`、`experiments_v4/v4_results.json` 与 `experiments_v5/v5_results.json`。])

v3、v4 和 v5 分别获得 12.83、10.08 和 5.06 个百分点的宏召回率增益，说明迭代方向有效，但边际提升逐渐缩小。v5 的准确率已接近 80%，宏召回率仍低 2.77 个百分点，表明剩余误差更集中在少数困难类别，而非所有类别同步下降。

== v4 状态读取消融

v4 同时比较九路/十二路和终态/八检查点四种组合。十二路八检查点方案达到 72.17% 宏召回率；九路终态仅为 50.65%。十二路终态与九路八检查点均在约 63% 附近，说明纹理信息和中间状态读取都能独立改善性能，而两者结合效果最好。

#academic-table(
  5,
  ([模式], [输入列], [读取次数], [特征维数], [验证宏召回率]),
  (
    [RGB/事件终态], [9], [1], [108], [50.65%],
    [RGB/事件检查点], [9], [8], [864], [63.25%],
    [十二路终态], [12], [1], [144], [63.27%],
    [十二路检查点], [12], [8], [1152], [72.17%],
  ),
  widths: (44mm, 25mm, 28mm, 30mm, 32mm),
)

== 验证集筛选十分类

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 6pt,
  metric([验证准确率], [97.36%]),
  metric([验证宏召回率], [97.98%], tone: "blue"),
  metric([测试准确率], [97.02%], tone: "orange"),
  metric([测试宏召回率], [96.80%], tone: "green"),
)

筛选十分类的 v5 主模型使用 `graded_q000` 连续事件、$tau_(max)=256$、$gamma=3.0$、权重衰减 $10^(-4)$，训练 18 个 epoch。验证集共 3,791 个样本，十类均有预测；验证宏 F1 为 96.60%。模型通过 80% 门槛后，才对 4,738 个冻结测试样本进行一次性评估。

#figure(
  image("../figures/validation_confusion_matrix_balanced_v3.png", width: 118mm),
  caption: [筛选十分类 v3 验证集归一化混淆矩阵（用于展示类别混淆结构）],
)

现有混淆矩阵来自十分类 v3，而最终 v5 目录保留了逐类分类报告但没有输出 PNG 混淆矩阵。因此本图只用于说明 v3 的类别混淆结构，不能冒充 v5 混淆矩阵。后续若保存 v5 预测向量，应重新生成同一版式的最终混淆矩阵。

== 十分类逐类表现

v5 验证集的最低召回率为 Tomato Yellow Leaf Curl Virus 的 95.45%，最高为 Peach healthy 的 100%。最低精确率为 Cherry healthy 的 88.00%，表明部分其他类别样本被误报为该类。尽管总体指标较高，类别支持数从 58 到 881 差异明显，宏指标仍是必要的补充。

#academic-table(
  4,
  ([类别（简写）], [Precision], [Recall], [Support]),
  (
    [Blueberry healthy], [93.57%], [97.08%], [240],
    [Cherry healthy], [88.00%], [96.35%], [137],
    [Corn common rust], [97.93%], [98.95%], [191],
    [Corn healthy], [97.87%], [98.92%], [186],
    [Grape leaf blight], [92.86%], [98.26%], [172],
    [Orange greening], [99.08%], [97.73%], [881],
    [Peach healthy], [89.23%], [100.00%], [58],
    [Soybean healthy], [98.51%], [97.42%], [814],
    [Tomato yellow curl], [98.44%], [95.45%], [857],
    [Tomato healthy], [98.07%], [99.61%], [255],
  ),
  widths: (72mm, 30mm, 30mm, 25mm),
)

== 结果解释边界

#callout([必须保持的学术表述], [
  97.02% 测试准确率属于“由先前验证结果筛选的十类 + 十二路连续输入 + 12×12 动态阵列仿真 + 逐帧状态读取 + 线性 Softmax 读出”这一完整流程。它既不是完整 38 类指标，也不是九路输入或真实硬件的实测准确率。
], tone: "orange")
