# 已有实验结果

[summary.csv](summary.csv) 使用 0–1 比例保存精确指标，未重新训练或四舍五入替代原始值。

- [ConvSNN38](convsnn38/metrics.json)：终版报告引用的 selection 与 final_test 指标，含原 metrics 文件相对路径和 SHA-256；理想与硬件感知分开列出。
- [RC38 v5](rc38/v5_results.json)：最佳验证结果 0.7990562781 / 0.7722612423，`test_evaluated=false`，尚无该模型的独立测试结果。保留 v3/v4 结果用于演进比较。
- [RC10 v5](rc10/v5_results.json)：冻结测试 0.9702406079 / 0.9679804757，训练/验证/测试样本数 15162/3791/4738。类别基于此前 RC38 验证成绩筛选，不能与完整 38 类直接比较。

RC 结果复制自本地 `artifacts/experiments_v5` 与 `artifacts_best10/experiments_v5`；分类报告为验证集，编码参数仅由训练集确定。训练中间状态和模型未提交。本次整合的测试仅验证实现与交接格式，不构成完整数据重训。
