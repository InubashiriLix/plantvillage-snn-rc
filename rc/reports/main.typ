#import "template.typ": *
#import "icons.typ": *

#show: report-init

#set page(numbering: none, header: none, footer: none)
#align(center)[
  #v(22mm)
  #leaf(size: 48pt)
  #v(11mm)
  #text(font: ("Noto Sans CJK SC", "FandolHei"), size: 24pt, weight: 750, fill: colors.green)[
    基于 RGB-Tonic 与 ON/OFF 事件编码的
  ]
  #v(3mm)
  #text(font: ("Noto Sans CJK SC", "FandolHei"), size: 25pt, weight: 750, fill: colors.green)[
    光电储备池植物叶片分类系统
  ]
  #v(9mm)
  #line(length: 48mm, stroke: 1.2pt + colors.orange)
  #v(8mm)
  #text(size: 14pt, fill: colors.blue)[项目设计与实验分析报告]
  #v(20mm)
  #grid(
    columns: (34mm, 72mm),
    gutter: 10pt,
    align: (right, left),
    text(fill: colors.gray)[项目名称], [PlantVillage RGB-Tonic ON/OFF Reservoir Computing],
    text(fill: colors.gray)[报告类型], [算法仿真、性能评估与硬件接口设计],
    text(fill: colors.gray)[文档版本], [1.0],
    text(fill: colors.gray)[编制日期], [2026 年 9 月],
  )
  #v(1fr)
  #block(width: 150mm, fill: colors.green-soft, radius: 5pt, inset: 10pt)[
    #set par(first-line-indent: 0em)
    #text(size: 9pt, fill: colors.green)[
      本报告中的性能指标来自软件仿真与冻结实验产物；除硬件接口协议外，不将仿真结果表述为真实器件实测结果。
    ]
  ]
]

#pagebreak()
#set page(numbering: "i", header: auto, footer: auto)
#counter(page).update(1)

#include "chapters/00-abstract.typ"
#pagebreak()
#counter(heading).update(0)
#outline(title: [目录], depth: 3, indent: auto)
#pagebreak()
#set page(numbering: "1", header: auto, footer: auto)
#counter(page).update(1)
#include "chapters/01-introduction.typ"
#include "chapters/02-system.typ"
#include "chapters/03-preprocessing.typ"
#include "chapters/04-reservoir.typ"
#include "chapters/05-evolution.typ"
#include "chapters/06-experiments.typ"
#include "chapters/07-results.typ"
#include "chapters/08-hardware.typ"
#include "chapters/09-conclusion.typ"

#pagebreak()
#heading(level: 1, outlined: true)[参考文献]
#bibliography("references.bib", style: "gb-7714-2015-numeric", title: none)

#include "chapters/10-appendix.typ"
