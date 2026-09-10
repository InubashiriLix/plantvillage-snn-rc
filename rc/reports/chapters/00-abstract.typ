#import "../template.typ": *

#heading(level: 1, outlined: false, numbering: none)[摘　要]

植物叶片病害识别通常依赖卷积神经网络等高计算量模型，而面向低功耗边缘端与新型光电器件的分类系统还需要兼顾输入编码、动态记忆、状态读取和硬件接口。本项目构建了一套面向 PlantVillage 数据集的光电储备池计算流程：首先将静态 RGB 图像划分为空间网格，并通过蛇形扫描转换为时间序列；随后联合编码 RGB 绝对强度、相邻区域 ON/OFF 变化事件以及局部纹理；最后将输入映射至 12×12 动态阵列，通过异构时间常数形成储备池状态，并训练线性读出层完成分类。

项目依次验证了单次扫描、RGB—纹理双通路、8×8 空间扫描和逐帧稠密读取四条技术路线。在完整 38 类验证任务上，v4 模型取得 72.02% 准确率和 72.17% 宏召回率；v5 模型进一步达到 79.91% 准确率和 77.23% 宏召回率。两者均未达到预设的 80% 宏召回率门槛，因此未访问冻结测试集。在由先前验证结果筛选的十分类任务上，v5 模型取得 97.36% 验证准确率、97.98% 验证宏召回率，以及 97.02% 测试准确率和 96.80% 测试宏召回率。该十分类结果用于硬件联调，但不作为无偏的 38 类结果替代品。

在系统交付方面，项目形成了九路和十二路连续输入、九路二值兼容输入、固定类别映射、样本索引、校验清单及完整硬件播放协议。结果表明，在保持每幅图像 64 个光学帧的条件下，空间结构、纹理通道和稠密状态读取能够显著增强线性读出层的判别能力，并为后续真实器件标定与闭环实验提供可复现的数据基础。

#v(6pt)
#set par(first-line-indent: 0em)
*关键词：* 植物病害识别；储备池计算；事件编码；光电阵列；PlantVillage；边缘计算

#v(16pt)
#heading(level: 1, outlined: false, numbering: none)[Abstract]
#set text(lang: "en")
This project develops an optoelectronic reservoir-computing pipeline for PlantVillage leaf classification. Static RGB images are partitioned into spatial patches and unfolded into temporal sequences by a fixed serpentine scan. RGB tonic intensity, signed ON/OFF changes, and local texture descriptors are mapped to a 12×12 dynamic array with heterogeneous decay constants. A linear readout is then trained from the resulting reservoir states. On the full 38-class validation task, the v5 dense-read model achieves 79.91% accuracy and 77.23% macro recall. On a validation-selected ten-class subset, it achieves 97.02% test accuracy and 96.80% test macro recall. The latter is explicitly treated as a hardware-integration subset rather than an unbiased replacement for the full task. The project additionally delivers reproducible 9-channel and 12-channel hardware input formats, checksums, class mappings, and a 64-frame playback protocol.

#v(6pt)
#set par(first-line-indent: 0em)
*Keywords:* plant disease recognition; reservoir computing; event encoding; optoelectronic array; PlantVillage
#set text(lang: "zh")
