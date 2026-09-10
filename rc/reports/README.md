# 中文 Typst 报告

主文件为 `main.typ`，章节位于 `chapters/`，统一版式与学术图标分别定义在
`template.typ` 和 `icons.typ`。

## 编译

从仓库根目录执行：

```bash
cd rc/reports
typst compile main.typ
```

持续预览（从仓库根目录）：

```bash
typst watch rc/reports/main.typ
```

若希望将 PDF 写入仓库统一输出目录，可在仓库根目录运行：

```bash
typst compile rc/reports/main.typ output/pdf/plantvillage_reservoir_report_zh.pdf
```

## 数据更新原则

- 38 类 v4/v5 指标分别以 `artifacts/experiments_v4/v4_results.json` 和
  `artifacts/experiments_v5/v5_results.json` 为准。
- 十分类指标以 `artifacts_best10/experiments_v5/v5_results.json` 为准。
- 硬件文件、shape 与校验信息以
  `artifacts_best10/hardware_export_v1/manifest.json` 为准。
- 修改实验结果后，需要同步检查摘要、第 7 章、总结和附录 D。

## 视觉规范

- 正文：Noto Serif CJK SC
- 标题：Noto Sans CJK SC
- 代码：Noto Sans Mono CJK SC
- 主色：`#245C4A`
- 辅色：`#4E6E81`
- 强调色：`#D9822B`

图标均由 Typst 原生矢量图元构成，避免使用网页图标字体和低分辨率位图。
