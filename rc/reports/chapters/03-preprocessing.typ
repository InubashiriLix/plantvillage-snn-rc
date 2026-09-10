#import "../template.typ": *
#import "../icons.typ": *

= 数据集与输入编码

== 数据集与任务设置

项目使用 PlantVillage 的 38 类叶片目录作为完整任务。原始 `data/train` 被分层拆分为训练集和验证集，已有 `data/val` 被视为冻结测试集。另构建两类十分类任务：一类为自然的番茄十分类，另一类依据早期 38 类验证报告中的单类表现选择，用于降低首次硬件联调难度。

筛选十分类包含 15,162 个训练样本、3,791 个验证样本和 4,738 个冻结测试样本。必须注意，这些类别由先前验证表现决定，因此十分类结果存在选择条件，不能与完整 38 类任务作无偏横向替换。

== 图像归一化与空间分块

输入图像先转换为 RGB，使用双线性插值缩放至 64×64，并除以 255 映射到 $[0,1]$。v2/v3 使用 3×4 分块以形成 12 个时间步；v4/v5 使用 8×8 分块，每个 patch 恰为 8×8 像素，共形成 64 个时间步。

对于第 $t$ 个 patch $Q_t$，颜色通道 $c∈{R,G,B}$ 的 tonic 值定义为

$ C_c(t) = 1 / |Q_t| sum_((x,y)∈Q_t) I(x,y,c). $ <eq-tonic>

该值表示当前空间区域的绝对颜色和亮度信息。

== 蛇形扫描

固定蛇形扫描保证连续时间步尽量对应相邻空间区域，从而使相对差分具有明确的局部意义。训练、验证、测试以及后续真实硬件必须使用同一顺序。

#figure(
  block(width: 100%, fill: colors.blue-soft, radius: 5pt, inset: 10pt)[
    #set par(first-line-indent: 0em)
    #grid(
      columns: (1fr,) * 8,
      rows: (auto,) * 8,
      gutter: 2pt,
      ..range(64).map(i => {
        let row = calc.floor(i / 8)
        let col = calc.rem(i, 8)
        let step = if calc.rem(row, 2) == 0 { row * 8 + col + 1 } else { row * 8 + (8 - col) }
        block(
          fill: if calc.rem(row, 2) == 0 { white } else { colors.green-soft },
          stroke: .45pt + colors.line,
          radius: 2pt,
          inset: 4pt,
          align(center, text(size: 7.5pt, weight: 600, str(step))),
        )
      })
    )
  ],
  caption: [8×8 patch 的固定蛇形扫描次序，数字对应硬件播放 step],
) <fig-scan>

== RGB Tonic 与 ON/OFF 事件

事件由相邻 patch 的颜色统计差分产生，而不是由器件响应反向决定。对颜色通道 $c$，采用对称相对差分

$ d_c(t) = (C_c(t)-C_c(t-1)) / (C_c(t)+C_c(t-1)+epsilon), quad epsilon=10^(-6). $ <eq-diff>

二值事件根据训练集绝对差分分位数阈值 $theta_c$ 生成：

$ "ON"_c(t) = cases(1 & d_c(t)>theta_c, 0 & "其他"), quad "OFF"_c(t) = cases(1 & d_c(t)<-theta_c, 0 & "其他"). $

v5 使用连续幅值事件。正负变化分别除以训练集得到的通道尺度并裁剪至 $[0,1]$：

$ tilde("ON")_c(t)="clip"(max(d_c(t),0)/s_(c,+),0,1), $
$ tilde("OFF")_c(t)="clip"(max(-d_c(t),0)/s_(c,-),0,1). $

这与真实事件相机的异步像素触发机制并不相同，但借鉴了用变化符号分离增强与减弱响应的思想。事件视觉的一般原理与性质可参见 @gallego2022。

== 纹理扩展通道

十二路版本在九路 RGB/事件输入之后增加局部亮度标准差、Sobel 梯度幅值均值和 16-bin 灰度熵。三路特征分别刻画局部对比度、边缘强度和纹理复杂度，并使用训练集 99% 分位数缩放后裁剪。

#academic-table(
  4,
  ([阵列列], [通道], [信息类型], [取值范围]),
  (
    [1–3], [R、G、B], [当前 patch 的绝对颜色], [$[0,1]$],
    [4–6], [ON_R、ON_G、ON_B], [相邻 patch 的正向变化], [$[0,1]$ 或 0/1],
    [7–9], [OFF_R、OFF_G、OFF_B], [相邻 patch 的负向变化], [$[0,1]$ 或 0/1],
    [10], [luminance_std], [局部亮度对比度], [$[0,1]$],
    [11], [sobel_mean], [局部边缘强度], [$[0,1]$],
    [12], [entropy_16bin], [局部纹理复杂度], [$[0,1]$],
  ),
  widths: (22mm, 43mm, 65mm, 27mm),
)

#figure(
  block(fill: colors.paper, stroke: .6pt + colors.line, radius: 5pt, inset: 10pt)[
    #set par(first-line-indent: 0em)
    #grid(
      columns: (1fr, 1fr, 1fr), gutter: 8pt,
      block(fill: colors.green-soft, radius: 4pt, inset: 8pt)[#icon-label(rgb-icon(), [Tonic 通路], [R/G/B 连续绝对强度])],
      block(fill: colors.orange-soft, radius: 4pt, inset: 8pt)[#icon-label(event(), [事件通路], [ON/OFF 分离的局部变化])],
      block(fill: colors.blue-soft, radius: 4pt, inset: 8pt)[#icon-label(texture(), [纹理通路], [对比度、边缘和熵])],
    )
  ],
  caption: [十二路联合编码的三类信息来源],
)

== 归一化的数据治理

所有阈值、事件幅值尺度和纹理尺度只使用训练集统计。训练、验证和测试采用完全相同的编码参数；第一帧没有前序 patch，因此六路事件固定为零。该约束不仅防止数据泄漏，也保证导出的硬件序列与软件训练流程一致。
