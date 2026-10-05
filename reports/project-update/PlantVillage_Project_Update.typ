#let ink = rgb("17352C")
#let muted = rgb("5D716A")
#let green = rgb("246B51")
#let pale = rgb("EDF5F0")
#let line-color = rgb("D6E3DC")
#let amber = rgb("985D1E")
#let repo = "https://github.com/InubashiriLix/plantvillage-snn-rc"
#let commit = "e179579f7a3b6be866fb6a0f64df71012e1a72e3"
#let base = repo + "/tree/" + commit
#let update = base + "/results/convsnn10_robustness"
#let audit = json("assets/handoff_verification.json")
#let data = csv("assets/summary.csv")

#set document(title: "PlantVillage 项目概览与电导扰动评估更新", author: "PlantVillage 项目", date: datetime(year: 2026, month: 10, day: 5))
#set page(paper: "a4", margin: (left: 19mm, right: 19mm, top: 20mm, bottom: 19mm),
  header: [#set text(size: 8pt, fill: muted)
    #grid(columns: (1fr, auto), [PLANTVILLAGE / RESEARCH UPDATE], [2026.10.05])
    #v(2pt)#line(length: 100%, stroke: .5pt + line-color)],
  footer: context [#set text(size: 8pt, fill: muted)
    #grid(columns: (1fr, auto), [公开交付核验 · #link(repo)[GitHub 项目仓库]], [#counter(page).display("1") / #counter(page).final().first()])])
#set text(font: ("Noto Sans CJK SC", "DejaVu Sans"), size: 10pt, fill: ink, lang: "zh")
#set par(justify: true, leading: .68em, spacing: .8em)
#set heading(numbering: none)
#show heading.where(level: 2): it => block(above: 12pt, below: 6pt)[#text(size: 13pt, weight: "bold", fill: green, it.body)]
#show link: set text(fill: green)
#show raw.where(block: true): it => block(width: 100%, fill: rgb("F3F6F4"), inset: 9pt, radius: 4pt)[
  #set text(font: "DejaVu Sans Mono", size: 7.7pt)
  #it
]
#show raw.where(block: false): set text(font: ("DejaVu Sans Mono", "Noto Sans CJK SC"), size: 8.3pt)

#let title(kicker, body, subtitle) = {
  text(size: 9pt, weight: "bold", fill: green, kicker)
  v(5pt)
  text(size: 25pt, weight: "bold", body)
  v(6pt)
  text(size: 10pt, fill: muted, subtitle)
  v(12pt)
}
#let callout(label, body, warning: false) = block(width: 100%, fill: if warning { rgb("FBF2E7") } else { pale },
  stroke: (left: 3pt + if warning { amber } else { green }), inset: 10pt, radius: (right: 4pt))[
  #text(weight: "bold", fill: if warning { amber } else { green }, label)
  #v(4pt)#body
]
#let metric(value, label, note) = block(width: 100%, fill: pale, inset: 11pt, radius: 4pt)[
  #text(size: 25pt, weight: "bold", fill: green, value)
  #v(3pt)#text(size: 10pt, weight: "bold", label)
  #v(4pt)#text(size: 8pt, fill: muted, note)
]
#let tab(headers, rows, widths: auto, size: 8.7pt) = {
  set text(size: size)
  set par(leading: .45em)
  table(columns: if widths == auto { headers.len() } else { widths }, inset: 6pt,
    stroke: .4pt + line-color,
    fill: (_, y) => if y == 0 { green } else if calc.rem(y, 2) == 0 { rgb("F4F8F5") } else { white },
    table.header(..headers.map(h => text(fill: white, weight: "bold", h))), ..rows.flatten())
}
#let source(path, label) = link(base + "/" + path, label)

#title("01 / 汇报摘要", [PlantVillage 项目概览\ 与电导扰动评估更新], [面向项目汇报与资料交接 · 核验版本 #commit.slice(0, 7)])

本项目研究植物叶片分类的两条器件相关计算路线：端到端卷积脉冲网络（ConvSNN）与固定器件动力学的储备池计算（RC）。本次新增内容聚焦 *ConvSNN Top10 原模型的电导扰动鲁棒性*，补齐此前仅有常规准确率、训练记录和时间步曲线的交付缺口。

#grid(columns: (1fr, 1fr, 1fr), gutter: 9pt,
  metric("93.8496%", "原模型基线复现", "4,894 张预测逐张一致"),
  metric("26 次", "全量扰动评估", "6 档强度；每次 4,894 张"),
  metric("5 seeds", "非零扰动重复", "公开均值、样本标准差与原始行"))

== 这次交付解决了什么

已发布原模型推理权重、差分电导状态、逐次扰动结果、汇总 CSV、PNG/SVG 曲线及评估脚本。结果来自冻结模型的实际推理，没有从 93.85% 单点拟合或补造曲线，也没有重新训练模型。

#callout("可以从 GitHub 完成绘图与审阅", [
  对于此前提出的“不同扰动强度 → 测试准确率，包含均值和标准差”，仓库现已提供完整输入。接收者可直接使用 #link(update + "/summary.csv")[summary.csv] 绘图，也可下载现成曲线。
])

== “资料完整”的适用范围

#tab(([接收者的工作], [GitHub 是否足够], [补充条件]), (
  ([查看结果、制作论文曲线], [是], [CSV 与图均已公开]),
  ([核对误差条与混淆矩阵], [是], [逐次结果和核验测试已公开]),
  ([取得原模型权重与电导状态], [是], [提供可移植推理权重]),
  ([重新执行完整模型推理], [有条件], [另需匹配的图片与运行环境]),
), widths: (1.6fr, 1fr, 2fr))

#v(8pt)
#callout("新增结果改变了结论边界", [
  原始 93.85% 说明无额外扰动时的分类性能；并不能证明电导鲁棒性。量程归一化 1% 静态高斯扰动下，准确率为 *73.12% ± 21.76 个百分点*，表明该模型对此扰动设定较敏感。
], warning: true)

#pagebreak()
#title("02 / 项目背景与既有证据", [两条计算路线，\ 三类评估边界], [理想仿真、器件约束仿真与真实电流分类需要分别解释。])

== ConvSNN：端到端学习脉冲网络

64 × 64 RGB 像素作为直接输入电流，在 8 个时间步内重复输入。三层卷积、两层全连接均接入 LIF 神经元，最终按累计输出脉冲数分类。每批推理前重置膜电位。硬件感知阶段将有符号突触映射为 G+/G− 差分单元，并用实测电导曲线约束离散脉冲写入；偏置、BatchNorm 与 LIF 参数仍为理想数值。

#block(fill: pale, inset: 9pt, width: 100%)[#align(center)[RGB 电流输入 → Conv + LIF × 3 → FC + LIF × 2 → 8 步累计脉冲]]

== RC：固定储备池，训练读出

模拟 RC v5 将图像转为 8 × 8 网格的 RGB、ON/OFF 及纹理序列，使用 12 × 12 储备池、64 帧状态形成 9,216 维特征，再训练线性读出。另一路实验直接分类九路真实硬件电流响应，使用 300 个样本；后续加入响应分布统计及神经网络／树模型融合。

== 已公开的主要结果

#tab(([任务与模型], [评估集], [准确率], [宏召回率]), (
  ([ConvSNN 38 类 · 理想], [冻结测试], [90.61%], [87.48%]),
  ([ConvSNN 38 类 · 硬件感知], [冻结测试], [81.19%], [75.94%]),
  ([ConvSNN Top10 · 理想], [冻结测试], [97.67%], [96.04%]),
  ([ConvSNN Top10 · 硬件感知], [冻结测试], [93.85%], [91.35%]),
  ([RC38 v5 · 模拟], [验证集], [79.91%], [77.23%]),
  ([RC10 v5 · 模拟], [冻结测试], [97.02%], [96.80%]),
  ([九路真实硬件 · 第一轮], [300 样本重复嵌套验证], [83.44%], [83.44%]),
  ([九路真实硬件 · 第二轮], [同批数据探索], [90.67%], [90.67%]),
  ([九路真实硬件 · 初始化扩展], [同批数据探索], [90.89%], [90.89%]),
), widths: (2.2fr, 1.7fr, .8fr, .9fr), size: 8.1pt)

#v(9pt)
#callout("不能把这些数字当成同一张排行榜", [
  ConvSNN 与 RC 的 Top10 类别集合不同。RC38 尚无独立测试结果；真实硬件 300 样本实验没有独立测试集，第二轮存在自适应开发，不能视为无偏泛化估计。ConvSNN 的“硬件感知”是电导约束仿真，不是芯片部署实测。
], warning: true)

#v(7pt)#text(size: 8pt, fill: muted)[来源：#source("results/summary.csv", [汇总指标])；#source("results/convsnn10/README.md", [ConvSNN Top10])；#source("results/rc10_hardware300_v2/README.md", [真实硬件第二轮说明])。本报告以公开 main 快照为准，不纳入其他分支未发布的结果。]

#pagebreak()
#title("03 / 本次新增实验", [电导扰动怎样施加], [同一冻结 checkpoint · 相同测试图片 · 每次从原始电导恢复])

== 实验对象与控制条件

#tab(([项目], [固定设置]), (
  ([模型], [ConvSNN Top10 硬件感知模型；训练 seed = 42；最佳 epoch = 3]),
  ([测试数据], [4,894 张冻结测试图片；逐图路径、标签和 SHA-256 已公开]),
  ([映射规模], [155,824 个逻辑突触；311,648 个差分物理单元]),
  ([扰动强度], [σ = 0、0.01、0.02、0.05、0.10、0.20]),
  ([随机重复], [每个非零强度用 100、101、102、103、104 五个噪声种子]),
  ([训练与选择], [不重训，不用本次结果重新选模型或调超参数]),
), widths: (1fr, 4.1fr))

== 噪声模型

#block(fill: pale, inset: 14pt, width: 100%)[
  #align(center)[
    $ G'_plus.minus = op("clip")(G_plus.minus + sigma (G_max - G_min) epsilon_plus.minus, G_min, G_max) $
    $ epsilon_+ , epsilon_- ~ cal(N)(0, 1) quad w' = s (G'_+ - G'_-) / (G_max - G_min) $
  ]
]

这里的 σ 是 *电导量程归一化的噪声标准差*。例如 1% 表示噪声标准差为整个量程的 1%，不是每个电导值的 1%，也不是权重的 1%。电导下限为 1.0014012 × 10⁻⁷ S，上限为 1.9276658 × 10⁻⁷ S；s 为原层权重缩放系数。

== 单次试验的步骤

+ 从原始 G+/G− 重新开始，避免跨试验累积噪声。
+ 为每个物理单元独立采样加性高斯误差，越界值裁剪至实测电导范围。
+ 用扰动后的差分电导重建突触权重，保留其余模型参数。
+ 对全部 4,894 张图片推理。该次电导误差在所有图片和时间步中固定。
+ 保存正确数、准确率、宏召回率、裁剪比例与完整混淆矩阵。

不同强度使用同一组种子配对标准正态样本。误差条为五个准确率的样本标准差（ddof = 1）；零扰动只运行一次，误差条记为 0。它衡量一个已训练模型面对不同噪声实现的波动，不是训练种子波动或置信区间。

#callout("明确的仿真边界", [静态单元差异与动态读噪声不同。本次没有模拟温漂、时间相关漂移或重新校准，也没有在真实器件上施加这些噪声。], warning: true)

#pagebreak()
#title("04 / 新增结果", [基线保持一致，\ 扰动下性能显著下降], [均值与标准差全部由 26 次全量推理结果计算，无插值数据或平滑拟合。])

#figure(image("assets/accuracy_vs_conductance_noise.svg", width: 100%), caption: [静态独立高斯电导扰动下的测试准确率；误差条为五次噪声重复的样本标准差。])

#v(4pt)
#tab(([电导扰动 σ], [准确率均值], [标准差／百分点], [宏召回率均值]), (
  ([0%], [93.85%], [0.00（单次基线）], [91.35%]),
  ([1%], [73.12%], [21.76], [69.92%]),
  ([2%], [51.10%], [21.81], [50.45%]),
  ([5%], [21.63%], [9.41], [27.88%]),
  ([10%], [12.12%], [7.60], [18.34%]),
  ([20%], [7.98%], [3.79], [9.32%]),
), widths: (.9fr, 1fr, 1.5fr, 1.3fr), size: 8.4pt)

== 如何在汇报中解释

无扰动时 4,593 / 4,894 张分类正确，准确率为 93.8496%，且所有逐图预测与原 GPU 记录一致。1% 与 2% 扰动下的种子间波动很大，应同时展示重复数据和误差条，不能只选表现最好的随机种子。

该结果支持“此模型对本次定义的电导扰动较敏感”，不支持“已经具备较强器件鲁棒性”。类别不均衡时，严重退化模型的总体准确率可能低于 10%；因此保留宏召回率与混淆矩阵供检查。

#text(size: 8pt, fill: muted)[数据：#link(update + "/summary.csv")[summary.csv] 与 #link(update + "/repeat_metrics.csv")[repeat_metrics.csv]。数值在 CSV 中均为 0–1 比例；绘制百分比时乘以 100。]

#pagebreak()
#title("05 / GitHub 交付核验", [从全新下载验证，\ 不只检查文件是否存在], [核验提交：e179579 · 远端文件树与通过测试的本地版本完全一致])

#tab(([检查项目], [已完成的证据]), (
  ([公开仓库重新下载], [从 GitHub main 创建全新浅克隆；确认没有 data/ 图片目录]),
  ([代码来源隔离], [明确将导入路径指向新克隆，确认实际执行新克隆中的评估模块]),
  ([无需数据集重画], [用下载的 summary.csv 与绘图脚本重新生成 PNG / SVG]),
  ([发布数据核验], [新克隆中的 4 项交付测试通过；权重哈希、矩阵与误差条均核对]),
  ([独立基线推理], [仅额外提供图片与已安装依赖；4,894 张预测全匹配，93.8496%]),
  ([重复性与 CI], [另起进程的 5 次抽取试验完全一致；80 项本地测试通过、1 项跳过；GitHub CI 通过]),
), widths: (1.2fr, 3.6fr), size: 8.5pt)

== 接收者应从哪里取文件

#tab(([需要的内容], [仓库内位置／文件]), (
  ([曲线数据], [results/convsnn10_robustness/ 下的 summary.csv、repeat_metrics.csv]),
  ([现成曲线], [同目录 accuracy_vs_conductance_noise.png / .svg]),
  ([原模型推理权重], [同目录 model/checkpoint.pt，约 4.2 MiB]),
  ([测试清单与器件输入], [同目录 model/test_manifest.json、model/device.csv]),
  ([来源与环境], [同目录 model/manifest.json、protocol.json、environment.json、requirements.txt]),
  ([运行脚本], [convsnn/plantvillage_snn/robustness.py；tools/plot_convsnn10_robustness.py]),
), widths: (1.1fr, 3.7fr), size: 8.2pt)

#v(8pt)
#callout("原模型权重的精确定义", [
  交付包保留原模型张量、G+/G− 电导与层缩放，已逐项精确核对，未重训或重新量化。为便于移植，移除原机路径、优化器及随机数状态，将 NumPy 数组转为张量。因此它是原模型的推理导出，不是原训练 checkpoint 的逐字节副本。
])

#v(8pt)
#callout("不能承诺“一份 GitHub 下载即可完成所有重训”", [
  图片数据、其他路线的全部训练权重、完整中间状态未全部公开。本次可以证明的是：本次需求所列的鲁棒性绘图资料和目标模型已齐备；重新推理另需匹配的图片、Python 依赖及兼容 GPU 环境。第三方重新下载的数据版本仍需通过逐图哈希校验。
], warning: true)

#pagebreak()
#title("06 / 接收者操作与后续工作", [如何使用这份交付], [只绘图无需 PyTorch 或图片；完整复现需额外配置运行环境。])

== 直接取图或重新绘图

#link(update)[打开 GitHub 电导扰动交付目录]，下载现成 PNG/SVG 或 `summary.csv`。重新绘图可在仓库根目录运行：

```bash
python -m pip install matplotlib
python tools/plot_convsnn10_robustness.py \
  --results results/convsnn10_robustness
```

== 重新推理

使用 Python 3.11，安装与验证环境匹配的依赖；把匹配图片放在 `data/val/<类别>/`。本次验证使用 RTX 4060 Laptop GPU、PyTorch 2.11.0+cu130、NumPy 2.4.3、snnTorch 0.9.4。

```bash
python -m pip install \
  -r results/convsnn10_robustness/requirements.txt
python -m pip install -e ./convsnn
python -m plantvillage_snn.robustness \
  --bundle results/convsnn10_robustness/model \
  --data-root data --device cuda \
  --output artifacts/robustness_replay \
  --sigmas 0 0.01 0.02 0.05 0.10 0.20 \
  --seeds 100 101 102 103 104
```

输出目录必须不存在。脚本先校验权重与图片 SHA-256，并要求零扰动逐图复现；不一致则停止。本次 CPU 核查曾有 5 张预测差异、准确率 93.7883%，已被拒绝，未用于发布曲线。项目提供 #source("convsnn/tools/download.py", [数据下载辅助脚本])，但本轮没有重新下载并验证第三方数据源的当前版本。

== 后续研究建议

+ 预先定义不同噪声模型：区分量程归一化加性误差、相对电导误差与动态读噪声。
+ 用训练／验证数据研究噪声感知训练与校准，再冻结方案并进行独立测试；不要用当前测试结果反复选参。
+ 增加噪声种子与独立训练重复，分别报告器件变化和训练变化；补充真实器件噪声测量。

== 可追溯来源

#set par(spacing: .5em)
#text(size: 8.2pt)[
  #source("README.md", [① 项目 README 与评估边界]) · #source("results/summary.csv", [② 历史结果汇总]) \
  #source("convsnn/reports/final/PlantVillage_ConvSNN_Final.zh-CN.md", [③ 原 ConvSNN 终版报告]) · #source("rc/README.md", [④ RC 路线说明]) \
  #link(update)[⑤ 新增鲁棒性数据与协议] · #link(update + "/model/manifest.json")[⑥ 原权重来源与 SHA-256] \
  #link(audit.github_actions_run)[⑦ GitHub 自动测试记录] · 本报告随附 assets/handoff_verification.json 保存全新克隆核验摘要。
]

#v(8pt)
#text(size: 8pt, fill: muted)[本报告基于 2026-10-05 核验的公开提交 e179579。上述链接固定到该实验版本；后续 main 变化不改变本文所引用的数据。Typst 源文件与 assets/ 放在同一目录，执行 `typst compile PlantVillage_Project_Update.typ PlantVillage_Project_Update.pdf` 可重新编译。]
