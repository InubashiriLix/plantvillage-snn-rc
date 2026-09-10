# PlantVillage ConvSNN 文档索引

## 正式报告

- [终版 PDF](../reports/final/PlantVillage_ConvSNN_Final.zh-CN.pdf)
- [终版 Markdown](../reports/final/PlantVillage_ConvSNN_Final.zh-CN.md)
- [报告来源清单](../reports/final/manifest.json)

报告作者信息：Xinrong Li，2363123。

## 当前实验数据

- 严格硬件链：`artifacts/hardware_refresh_2026-08-06/`
- 四模型汇总：`artifacts/hardware_refresh_2026-08-06/report/comparison.json`
- 新 Top-10 清单：`artifacts/hardware_refresh_2026-08-06/top10_classes.json`
- 38 类理想源：`artifacts/current_sources/all38_ideal_100ep/`
- 固定数据划分：`artifacts/current_sources/splits/seed42_val0p15.json`

严格链状态为 complete，共完成 19 个任务。最终测试在全部 selection 决策冻结后执行。

## 结果定义

- “类别准确率”按该类别召回率解释。
- Top-10 由 38 类硬件感知 selection recall 排名确定，并按 precision、类别名称处理并列。
- 硬件感知结果采用实测电导曲线和离散写入脉冲仿真，不代表真实芯片部署测量。
- 完整激活和梯度张量不进入归档，只保留必要的参数、梯度、放电、电导和脉冲摘要。

## 历史资料

旧报告、旧实验和旧 Top-10 RGB 导出已从 live 文档区移除，并存入：

`artifacts/archive/legacy_workspace_2026-08-06/`

该目录包含分项 `.tar.zst`、内容清单、SHA256 和恢复说明。
