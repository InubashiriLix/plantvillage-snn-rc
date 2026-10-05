# PlantVillage 项目概览与电导扰动评估更新

[下载六页 PDF](PlantVillage_Project_Update.pdf) · [Typst 源文件](PlantVillage_Project_Update.typ)

中文汇报涵盖项目两条路线、原有成绩、新增电导扰动方案与结果、GitHub 交付核验和复现命令。引用的实验数据固定到 `e179579`。

已从 GitHub 全新克隆核验：无需数据集即可重画曲线；4 项交付测试与仓库审计通过；使用新克隆的代码、权重以及另行提供的测试图片，重新得到 93.8496%，4,894 张逐图预测全部匹配。

**适用范围：** GitHub 已包含本次需求的曲线数据、原模型推理权重和脚本。重新推理另需图片与依赖；不能声称所有路线的数据、训练中间状态和权重都已完整分发。详细核验摘要见 [handoff_verification.json](assets/handoff_verification.json)。

在本目录重新编译（需 Typst 与 Noto Sans CJK SC 字体）：

```bash
typst compile PlantVillage_Project_Update.typ PlantVillage_Project_Update.pdf
```

请保留 `assets/` 与源文件的相对位置。PDF 已用 Poppler 逐页渲染检查。
