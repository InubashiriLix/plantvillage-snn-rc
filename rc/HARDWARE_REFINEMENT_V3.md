# Hardware300 v3：纯电流第三轮

本轮只使用九路实测电流（R/G/B、ON-red/green/blue、OFF-red/green/blue）。训练和预测都不读取光学输入；`inputs.csv` 仅在数据审计阶段用于核对样本与扫描顺序。结果来自同一批 300 张样本的三组固定五折重复，仍然没有独立测试集。

## 运行

在仓库根目录执行：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python rc/scripts/12_optimize_hardware_v3.py audit \
  --responses artifacts_hardware300/sources/responses.csv \
  --inputs artifacts_hardware300/sources/inputs.csv \
  --workbook artifacts_hardware300/sources/workbook.xlsx \
  --output-dir artifacts_hardware300_v3/audit

python rc/scripts/12_optimize_hardware_v3.py search \
  --responses artifacts_hardware300/sources/responses.csv \
  --inputs artifacts_hardware300/sources/inputs.csv \
  --workbook artifacts_hardware300/sources/workbook.xlsx \
  --configs rc/configs/hardware_refinement_v3.json \
  --output-dir artifacts_hardware300_v3/screen --workers 6 --hours 2

python rc/scripts/12_optimize_hardware_v3.py evaluate \
  --responses artifacts_hardware300/sources/responses.csv \
  --inputs artifacts_hardware300/sources/inputs.csv \
  --workbook artifacts_hardware300/sources/workbook.xlsx \
  --configs rc/configs/hardware_refinement_v3.json \
  --output-dir artifacts_hardware300_v3/run --workers 6 --hours 8

python rc/scripts/12_optimize_hardware_v3.py fit \
  --responses artifacts_hardware300/sources/responses.csv \
  --inputs artifacts_hardware300/sources/inputs.csv \
  --workbook artifacts_hardware300/sources/workbook.xlsx \
  --configs rc/configs/hardware_refinement_v3.json \
  --output-dir artifacts_hardware300_v3/run --workers 6 --hours 1
python rc/scripts/12_optimize_hardware_v3.py report \
  --responses artifacts_hardware300/sources/responses.csv \
  --inputs artifacts_hardware300/sources/inputs.csv \
  --workbook artifacts_hardware300/sources/workbook.xlsx \
  --output-dir artifacts_hardware300_v3/run
```

`evaluate` 的缓存按代码、数据、配置和训练折指纹保存，可以中断后用同一命令恢复。只有完成 15 个外层折后才会写完整均值；部分运行只报告进度。`fit` 生成的模型可以只用硬件 CSV 做预测：

```bash
python rc/scripts/12_optimize_hardware_v3.py predict \
  --responses path/to/measured_responses.csv \
  --model artifacts_hardware300_v3/run/model.joblib \
  --output predictions.csv
```

模型文件和逐样本预测属于本地产物，不应提交公开仓库。提交时只保留代码、配置、聚合指标和简短报告。
