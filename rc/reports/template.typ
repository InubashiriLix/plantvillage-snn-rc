#let colors = (
  ink: rgb("#22312D"),
  green: rgb("#245C4A"),
  green-2: rgb("#4D8A73"),
  green-soft: rgb("#EAF2EE"),
  blue: rgb("#4E6E81"),
  blue-soft: rgb("#ECF2F5"),
  orange: rgb("#D9822B"),
  orange-soft: rgb("#FBF0E4"),
  gray: rgb("#697570"),
  line: rgb("#CBD6D1"),
  paper: rgb("#FBFCFB"),
)

#let report-init(body) = {
  set document(title: "基于 RGB-Tonic 与 ON/OFF 事件编码的光电储备池植物叶片分类系统")
  set page(
    paper: "a4",
    margin: (top: 24mm, bottom: 22mm, left: 25mm, right: 22mm),
    numbering: "1",
    number-align: center,
    header: context {
      if counter(page).get().first() > 1 {
        set text(size: 8.5pt, fill: colors.gray)
        grid(
          columns: (1fr, auto),
          align: (left, right),
          [PlantVillage 光电储备池分类系统],
          [中文技术报告],
        )
        line(length: 100%, stroke: .45pt + colors.line)
      }
    },
  )
  set text(
    font: ("Noto Serif CJK SC", "FandolSong"),
    lang: "zh",
    region: "CN",
    size: 10.5pt,
    fill: colors.ink,
  )
  set par(justify: true, leading: 0.78em, first-line-indent: 2em)
  set heading(numbering: "1.1")
  set figure(numbering: "1")
  set table(stroke: .45pt + colors.line, inset: 5pt)
  set math.equation(numbering: "(1)")
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(6pt)
    block(
      width: 100%,
      below: 14pt,
      stroke: (bottom: 1.2pt + colors.green),
      inset: (bottom: 7pt),
    )[
      #set text(font: ("Noto Sans CJK SC", "FandolHei"), weight: 700, size: 19pt, fill: colors.green)
      #it
    ]
  }
  show heading.where(level: 2): it => {
    v(7pt)
    block(above: 3pt, below: 7pt)[
      #set text(font: ("Noto Sans CJK SC", "FandolHei"), weight: 650, size: 14pt, fill: colors.ink)
      #it
    ]
  }
  show heading.where(level: 3): it => {
    block(above: 5pt, below: 4pt)[
      #set text(font: ("Noto Sans CJK SC", "FandolHei"), weight: 600, size: 11.5pt, fill: colors.blue)
      #it
    ]
  }
  show raw: it => box(
    fill: rgb("#F1F4F2"),
    stroke: .45pt + colors.line,
    radius: 3pt,
    inset: 5pt,
    text(font: "Noto Sans Mono CJK SC", size: 8.6pt, it),
  )
  show link: set text(fill: colors.blue)
  body
}

#let callout(title, body, tone: "green") = {
  let palette = if tone == "orange" {
    (fill: colors.orange-soft, accent: colors.orange)
  } else if tone == "blue" {
    (fill: colors.blue-soft, accent: colors.blue)
  } else {
    (fill: colors.green-soft, accent: colors.green)
  }
  block(
    width: 100%,
    fill: palette.fill,
    stroke: (left: 3pt + palette.accent),
    inset: (x: 10pt, y: 8pt),
    radius: (right: 4pt),
  )[
    #set par(first-line-indent: 0em)
    #text(font: ("Noto Sans CJK SC", "FandolHei"), weight: 700, fill: palette.accent)[#title]
    #v(3pt)
    #body
  ]
}

#let metric(label, value, note: none, tone: "green") = {
  let accent = if tone == "orange" { colors.orange } else if tone == "blue" { colors.blue } else { colors.green }
  block(
    width: 100%,
    fill: white,
    stroke: .7pt + colors.line,
    radius: 5pt,
    inset: 9pt,
  )[
    #set par(first-line-indent: 0em)
    #text(size: 8.5pt, fill: colors.gray)[#label]
    #v(2pt)
    #text(font: ("Noto Sans CJK SC", "FandolHei"), size: 18pt, weight: 700, fill: accent)[#value]
    #if note != none { v(2pt); text(size: 7.8pt, fill: colors.gray, note) }
  ]
}

#let academic-table(columns, header, rows, widths: auto) = {
  figure(
    table(
      columns: if widths == auto { columns } else { widths },
      align: (left,) + (center,) * (columns - 1),
      fill: (_, row) => if row == 0 { colors.green-soft } else if calc.rem(row, 2) == 0 { rgb("#F7F9F8") } else { white },
      stroke: (x, y) => if y == 0 or y == 1 { .8pt + colors.green } else { .35pt + colors.line },
      table.header(..header.map(cell => table.cell(text(weight: 650, cell)))),
      ..rows.flatten(),
    )
  )
}

#let bar-row(label, value, max-value: 1.0, color: colors.green, value-text: none) = {
  grid(
    columns: (32mm, 1fr, 17mm),
    gutter: 6pt,
    align: (right + horizon, left + horizon, right + horizon),
    text(size: 8.4pt, label),
    block(width: 100%, height: 9pt, fill: rgb("#E7ECE9"), radius: 2pt)[
      #block(width: value / max-value * 100%, height: 100%, fill: color, radius: 2pt)
    ],
    text(size: 8.4pt, weight: 650, if value-text == none { str(calc.round(value * 100, digits: 2)) + "%" } else { value-text }),
  )
}

#let source-note(body) = block(above: 2pt)[
  #set par(first-line-indent: 0em)
  #text(size: 7.8pt, fill: colors.gray)[数据来源：#body]
]
