# 历史胜出参数

来自终版报告引用的 38 类理想及硬件感知实验配置，仅将数据、输出、checkpoint、split manifest 与记录路径改为当前仓库约定。首次训练生成划分，seed 和比例保留。

从仓库根目录依次执行：

```bash
python -m plantvillage_snn train --config convsnn/configs/all38_ideal.json --device auto --evaluate-final
python -m plantvillage_snn train --config convsnn/configs/all38_hw.json --device auto --evaluate-final
```

完整数据不随仓库提供。训练随机性、设备和依赖版本可影响结果；本次整合仅验证最小训练路径。
