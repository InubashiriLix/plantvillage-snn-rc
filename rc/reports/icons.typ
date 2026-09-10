#import "template.typ": colors

// 统一的学术线性图标。所有图标由 Typst 原生矢量图元构成，适合 PDF 缩放与黑白打印。
#let icon-frame(body, size: 22pt, tone: colors.green) = box(
  width: size,
  height: size,
  inset: 0pt,
  baseline: 25%,
  rect(width: size, height: size, radius: 4pt, fill: tone.lighten(88%), stroke: .8pt + tone)[
    #align(center + horizon, body)
  ],
)

#let leaf(size: 22pt) = icon-frame(size: size)[
  #rotate(-35deg, ellipse(width: 10pt, height: 15pt, fill: colors.green.lighten(55%), stroke: .8pt + colors.green))
  #place(dx: -1pt, dy: 2pt, line(length: 12pt, angle: -48deg, stroke: .8pt + colors.green))
]

#let grid-icon(size: 22pt) = icon-frame(size: size, tone: colors.blue)[
  #grid(
    columns: (4pt,) * 3,
    rows: (4pt,) * 3,
    gutter: 1.5pt,
    ..range(9).map(i => rect(width: 4pt, height: 4pt, fill: if calc.rem(i, 2) == 0 { colors.blue } else { colors.blue.lighten(55%) }))
  )
]

#let scan(size: 22pt) = icon-frame(size: size, tone: colors.blue)[
  #set text(font: "Noto Sans CJK SC", size: 14pt, weight: 700, fill: colors.blue)
  ↝
]

#let rgb-icon(size: 22pt) = icon-frame(size: size)[
  #grid(columns: (4pt,) * 3, gutter: 1.5pt,
    circle(radius: 2pt, fill: rgb("#C95C54")),
    circle(radius: 2pt, fill: rgb("#4D8A73")),
    circle(radius: 2pt, fill: rgb("#4E6E81")),
  )
]

#let event(size: 22pt) = icon-frame(size: size, tone: colors.orange)[
  #set text(font: "Noto Sans CJK SC", size: 11pt, weight: 700, fill: colors.orange)
  ↑↓
]

#let texture(size: 22pt) = icon-frame(size: size, tone: colors.blue)[
  #set text(size: 14pt, weight: 700, fill: colors.blue)
  ≋
]

#let reservoir(size: 22pt) = icon-frame(size: size)[
  #grid(columns: (3pt,) * 4, gutter: 1.5pt,
    ..range(12).map(i => circle(radius: 1.5pt, fill: if calc.rem(i, 3) == 0 { colors.orange } else { colors.green }))
  )
]

#let readout(size: 22pt) = icon-frame(size: size, tone: colors.blue)[
  #set text(size: 13pt, weight: 700, fill: colors.blue)
  Σ
]

#let chip(size: 22pt) = icon-frame(size: size, tone: colors.orange)[
  #rect(width: 10pt, height: 10pt, fill: none, stroke: 1pt + colors.orange)[
    #align(center + horizon, text(size: 6.5pt, weight: 700, fill: colors.orange)[I/O])
  ]
]

#let icon-label(icon, title, subtitle) = grid(
  columns: (26pt, 1fr),
  gutter: 7pt,
  align: horizon,
  icon,
  block[
    #set par(first-line-indent: 0em)
    #text(font: ("Noto Sans CJK SC", "FandolHei"), size: 9.2pt, weight: 650)[#title]
    #linebreak()
    #text(size: 7.6pt, fill: colors.gray, subtitle)
  ],
)
