# 基于 RGB-Tonic + ON/OFF 事件编码的 12×12 光电阵列 RC 数据预处理与工作流程

## 0. 目的

本文档用于后续 **Reservoir Computing（RC）仿真**的数据预处理与阵列映射。

总体思路：

> 将静态 RGB 图像划分为多个空间区域，通过蛇形扫描将二维空间信息转换为时间序列；同时提取 RGB 绝对强度（tonic pathway）以及相邻区域之间的 ON/OFF 变化事件。随后将这些输入映射到 12×12 光电阵列的不同列，并利用器件的非线性响应、脉冲累积和自然衰减形成储备池状态，最后使用线性读出层完成分类。

当前推荐的有效计算区域为：

- **12 行 × 9 列 = 108 个物理储备节点**
- 第 1–3 列：RGB tonic 输入
- 第 4–6 列：RGB ON 事件
- 第 7–9 列：RGB OFF 事件
- 第 10–12 列：当前版本暂不参与计算，可保留为参考/漂移监测通道

---

## 1. 数据集输入

### 1.1 原始图像

每张输入图像统一处理为：

- 尺寸：`64 × 64`
- 通道：RGB
- 像素值建议归一化到 `[0, 1]`

记原始图像为：

$$
I(x,y,c), \qquad c\in\{R,G,B\}
$$

---

## 2. 空间区域划分

### 2.1 3×4 patch 划分

将每张 `64×64 RGB` 图像划分为 **3×4 = 12 个空间区域**：

$$
Q_1,Q_2,\ldots,Q_{12}
$$

由于 64 不能被 3 整除，实际程序中建议采用固定边界划分：

- 高度方向分成 3 组，尺寸尽可能接近
- 宽度方向分成 4 组，每组 16 像素

所有样本必须使用完全相同的划分规则。

---

## 3. 蛇形扫描与时间展开

12 个 patch 不作为 12 个独立静态输入，而是按照固定蛇形路径转化为 12 个时间步。

例如：

$$
Q_1\rightarrow Q_2\rightarrow Q_3\rightarrow Q_4
\rightarrow Q_8\rightarrow Q_7\rightarrow Q_6\rightarrow Q_5
\rightarrow Q_9\rightarrow Q_{10}\rightarrow Q_{11}\rightarrow Q_{12}
$$

具体编号可以根据实际图示重新确定，但必须满足：

1. 相邻时间步尽量对应空间相邻区域；
2. 所有训练、验证、测试样本使用同一条扫描路径；
3. RC 仿真和真实阵列实验使用相同顺序。

因此：

$$
t=1,2,\ldots,12
$$

分别对应 12 次仿生“注视”或空间采样。

---

## 4. RGB Tonic 特征提取

对第 $t$ 个区域 $Q_t$，分别计算 RGB 三个通道的平均值：

$$
R(t)=\frac{1}{|Q_t|}\sum_{(x,y)\in Q_t}I(x,y,R)
$$

$$
G(t)=\frac{1}{|Q_t|}\sum_{(x,y)\in Q_t}I(x,y,G)
$$

$$
B(t)=\frac{1}{|Q_t|}\sum_{(x,y)\in Q_t}I(x,y,B)
$$

于是每张图像得到：

$$
12\times3=36
$$

个连续 RGB 数值。

这三条序列定义为：

$$
\mathbf u_{\mathrm{tonic}}(t) = [R(t),G(t),B(t)]
$$

其物理意义为当前注视区域的绝对颜色/亮度信息。

---

## 5. ON/OFF 事件生成

### 5.1 事件由图像差分产生

ON/OFF 事件应由 **相邻 patch 的 RGB 特征差分**产生，而不是由器件前后响应决定。

这样预处理仍属于前馈事件编码，不引入硬件反馈闭环。

### 5.2 推荐使用相对差分

对于颜色通道 $c\in\{R,G,B\}$，定义：

$$
d_c(t)=
\frac{
c(t)-c(t-1)
}{
c(t)+c(t-1)+\varepsilon
}
,\qquad t=2,\ldots,12
$$

其中：

$$
\varepsilon=10^{-6}
$$

可作为防止分母为 0 的小常数。

相比直接使用 $c(t)-c(t-1)$，相对差分对整体亮度变化更稳定。

### 5.3 ON 事件

$$
ON_c(t)=
\begin{cases}
1,& d_c(t)>\theta_c\\
0,& \text{otherwise}
\end{cases}
$$

表示当前区域相对于前一个区域在该颜色通道上显著增强。

### 5.4 OFF 事件

$$
OFF_c(t)=
\begin{cases}
1,& d_c(t)<-\theta_c\\
0,& \text{otherwise}
\end{cases}
$$

表示当前区域相对于前一个区域在该颜色通道上显著减弱。

### 5.5 第一个时间步

因为第一个 patch 没有前一个区域，因此统一定义：

$$
ON_c(1)=OFF_c(1)=0
$$

所以：

- RGB tonic 通路：12 个有效时间步
- ON/OFF：11 次真实差分判断
- 整个系统仍统一为 12 个时间步

---

## 6. ON/OFF 阈值选择

### 6.1 推荐默认方案

针对训练集分别收集：

$$
|d_R|,\quad |d_G|,\quad |d_B|
$$

然后分别设置：

$$
\theta_R = Q_{0.70}(|d_R|)
$$

$$
\theta_G = Q_{0.70}(|d_G|)
$$

$$
\theta_B = Q_{0.70}(|d_B|)
$$

其中 $Q_{0.70}$ 表示训练集分布的 70% 分位数。

即仅保留变化幅度较大的约 30% 作为事件。

### 6.2 建议搜索范围

仿真阶段建议比较：

$$
q\in\{0.60,0.70,0.80\}
$$

也可扩展为：

$$
q\in\{0.50,0.60,0.70,0.75,0.80,0.85,0.90\}
$$

并记录：

- 分类准确率
- 宏召回率
- 平均事件数
- ON/OFF 比例
- R/G/B 三通道事件比例

最终阈值只能利用训练集/验证集确定，测试集不能参与阈值优化。

---

## 7. 每个时间步的完整 9 维输入

在第 $t$ 个时间步，形成：

$$
\mathbf u(t)=
[
R(t),G(t),B(t),
ON_R(t),ON_G(t),ON_B(t),
OFF_R(t),OFF_G(t),OFF_B(t)
]
$$

因此单张图像的预处理结果可以保存为：

$$
\mathbf U\in\mathbb R^{12\times9}
$$

其中：

- 行：12 个扫描时间步
- 列：9 条输入通路

| 索引 | 通路    |
| ---- | ------- |
| 1    | R tonic |
| 2    | G tonic |
| 3    | B tonic |
| 4    | R ON    |
| 5    | G ON    |
| 6    | B ON    |
| 7    | R OFF   |
| 8    | G OFF   |
| 9    | B OFF   |

这是后续 RC 仿真的核心输入张量。

---

## 8. 数据到光输入的映射

### 8.1 Tonic 通路

RGB tonic 使用连续幅值编码：

$$
P_c(t) =
P_{\min} +
c(t)(P_{\max}-P_{\min})
$$

其中：

- $c(t)\in[0,1]$
- $P_{\min},P_{\max}$ 根据器件有效动态范围确定

仿真中可先归一化为：

$$
P_c(t)\in[0,1]
$$

后续再根据真实器件光强范围进行标定。

### 8.2 ON/OFF 通路

第一版建议采用 **固定幅度二值事件**：

$$
P_{\mathrm{event}}(t)=
\begin{cases}
P_0,& event=1\\
0,& event=0
\end{cases}
$$

因此：

- ON/OFF 只编码“是否发生显著变化”
- 不编码变化幅度
- 事件稀疏性容易统计
- 适合分析事件驱动的计算效率

仿真中可先令：

$$
P_0=1
$$

---

## 9. 阵列映射方式

### 9.1 12×12 物理阵列

当前推荐使用其中：

$$
12\times9=108
$$

个有效物理单元。

| 阵列列 | 功能                |
| ------ | ------------------- |
| 1      | R tonic             |
| 2      | G tonic             |
| 3      | B tonic             |
| 4      | R ON                |
| 5      | G ON                |
| 6      | B ON                |
| 7      | R OFF               |
| 8      | G OFF               |
| 9      | B OFF               |
| 10–12  | 暂不参与 / 参考通道 |

---

## 10. 12 行的作用：并行储备池节点

12 行不对应 12 个 patch。

12 个 patch 对应的是：

$$
12\text{个时间步}
$$

12 行对应的是：

$$
12\text{个并行动态物理节点}
$$

因此，同一列的 12 个器件会处理同一条输入时间序列，但由于：

- 固定掩码
- 器件固有非均匀性
- 不同响应增益
- 不同衰减
- 不同非线性

最终得到不同状态。

---

## 11. 固定掩码

对于第 $j$ 列的输入 $u_j(t)$，第 $i$ 行实际接收到：

$$
v_{ij}(t)=m_{ij}u_j(t)
$$

其中 $m_{ij}$ 为固定输入掩码。

推荐仿真首先测试两种形式。

### A. 连续随机掩码

$$
m_{ij}\sim U(m_{\min},m_{\max})
$$

建议初始范围：

$$
m_{ij}\in[0.5,1.5]
$$

固定随机种子生成后，训练和测试期间保持不变。

### B. 二值掩码

$$
m_{ij}\in\{0,1\}
$$

更接近后续可能的行选择硬件。

> 如果真实阵列未来无法实现逐行不同光强，则仿真中应优先使用可由实际寻址逻辑实现的掩码方式，避免模拟结果无法迁移到硬件。

---

## 12. RC 状态更新模型

第 $i$ 行、第 $j$ 列节点的状态记为：

$$
x_{ij}(t)
$$

一般写为：

$$
x_{ij}(t) = F_{ij} \left[ x_{ij}(t-1), v_{ij}(t) \right]
$$

其中 $F_{ij}$ 应尽可能来自真实器件响应。

---

## 13. RC 仿真的模型层级

建议分三个层次逐步实现。

### Level 1：简化动力学模型

$$
x_{ij}(t) = \alpha_{ij}x_{ij}(t-1) + \beta_{ij}f(v_{ij}(t))
$$

其中：

- $\alpha_{ij}$：记忆衰减因子
- $\beta_{ij}$：响应增益
- $f(\cdot)$：非线性输入函数

例如：

$$
f(u)=\tanh(\gamma u)
$$

或：

$$
f(u)=1-e^{-\gamma u}
$$

### Level 2：实验拟合模型

根据真实器件的：

- 光强响应
- 脉冲累积
- 衰减时间
- 不同器件响应差异

拟合：

$$
F_{ij}
$$

此时各节点拥有不同参数。

### Level 3：查找表 / 实测轨迹驱动

直接使用实测器件数据库建立 LUT：

$$
x(t)=
LUT[
x(t-1),u(t),\Delta t
]
$$

用于硬件感知仿真。

---

## 14. 器件尾态与视觉残留

在单张图像的 12 步扫描过程中，不进行复位。

器件当前状态包含前几个输入留下的残余响应：

$$
x(t)
\sim
\sum_{\tau=1}^{t}
\alpha^{t-\tau}f[u(\tau)]
$$

因此：

- 最近 patch 影响较强
- 更早 patch 影响逐渐衰减
- 空间扫描顺序被编码进时间状态

这可以解释为一种：

> bio-inspired visual persistence / fading memory

即器件尾态对应有限时间尺度的视觉残留。

---

## 15. 单张图像的最终储备态

经过 12 个时间步后，读取第 1–9 列全部节点：

$$
X_{ij}=x_{ij}(12)
$$

得到：

$$
\mathbf X\in\mathbb R^{12\times9}
$$

展平：

$$
\mathbf x\in\mathbb R^{108}
$$

这 108 维向量作为该图像的最终 reservoir state。

---

## 16. 多时间采样扩展

如果后续需要增加状态维度，可在扫描结束后不同延迟采样：

$$
\tau_1,\tau_2,\ldots,\tau_M
$$

则状态维度变为：

$$
108\times M
$$

第一版仿真建议：

$$
M=1
$$

即只使用一个固定采样时刻，避免模型过度复杂。

---

## 17. 输出层

最终使用线性读出层完成分类。

对于 10 分类：

$$
\mathbf y=
\mathbf W_{\mathrm{out}}\mathbf x+\mathbf b
$$

其中：

$$
\mathbf W_{\mathrm{out}}
\in
\mathbb R^{10\times108}
$$

预测类别：

$$
\hat y=\arg\max_k y_k
$$

推荐优先测试：

1. Ridge regression
2. Multinomial logistic regression
3. Linear SVM

如果强调标准 RC，优先使用 ridge regression 或简单线性分类器。

储备池本身不进行反向传播训练，只训练输出层。

---

## 18. 样本之间的状态处理

### 单张图像内部

12 个 patch 连续输入：

- 不复位
- 保留尾态
- 利用历史信息
- 形成 fading memory

### 不同图像之间

需要恢复到可重复的起始工作区。

仿真第一版可以直接：

$$
x_{ij}(0)=0
$$

即每张图像开始前重置模拟节点状态。

真实器件实验则对应：

- 暗态自然恢复
- 输入前读取基线
- 必要时做 baseline normalization

仿真中不强制加入 395 nm 完全擦除机制。

---

## 19. 数据集预处理时必须避免数据泄漏

所有以下参数只能从训练集确定：

- RGB 归一化参数
- ON/OFF 阈值
- 固定掩码的随机种子
- 输入光强映射范围（若由数据分布确定）
- RC 模型超参数
- 输出层正则化参数

验证集可用于选择超参数。

最终测试集只能进行冻结模型评估。

---

## 20. 推荐保存的数据格式

每个样本建议保存：

```text
sample_id
label
rgb_tonic: [12, 3]
rgb_diff:  [12, 3]
on_events: [12, 3]
off_events:[12, 3]
rc_input:  [12, 9]
```

其中第一时间步：

```text
rgb_diff[0]   = 0
on_events[0]  = 0
off_events[0] = 0
```

推荐最终保存为：

```text
train_inputs.npy      # [N_train, 12, 9]
train_labels.npy

val_inputs.npy        # [N_val, 12, 9]
val_labels.npy

test_inputs.npy       # [N_test, 12, 9]
test_labels.npy
```

同时单独保存配置文件：

```text
preprocess_config.json
```

建议至少包含：

```json
{
  "image_size": [64, 64],
  "patch_grid": [3, 4],
  "scan_order": [],
  "difference": "symmetric_relative",
  "threshold_quantile": 0.7,
  "theta_R": null,
  "theta_G": null,
  "theta_B": null,
  "event_amplitude": 1.0,
  "active_columns": 9,
  "reservoir_rows": 12,
  "mask_seed": 42
}
```

---

## 21. 第一版 RC 仿真的推荐执行顺序

1. 建立 train / validation / test 数据划分
2. 所有图像统一为 `64×64×3`
3. 划分为 3×4 patch
4. 按固定蛇形路径展开为 12 步
5. 计算每个 patch 的 RGB tonic，得到 `[12,3]`
6. 计算相邻区域相对差分，得到 `[12,3]`
7. 仅利用训练集确定 RGB 阈值
8. 生成 ON/OFF 二值事件，分别得到 `[12,3]`
9. 拼接得到 RC 输入 `U:[12,9]`
10. 施加固定掩码
11. 通过器件动态模型
12. 获取最终 `12×9` 状态矩阵
13. 展平为 108 维 reservoir state
14. 训练线性读出层
15. 在验证集调参
16. 最终在测试集评估

---

## 22. 必须做的消融实验

为证明事件通路和器件动力学确实有作用，建议至少比较：

### A. RGB tonic only

仅使用第 1–3 列。

### B. Event only

仅使用第 4–9 列。

### C. RGB tonic + ON/OFF

使用完整第 1–9 列。

### D. 无记忆模型

令：

$$
\alpha=0
$$

用于证明 fading memory 的贡献。

### E. 无非线性模型

令：

$$
f(u)=u
$$

用于判断器件非线性的贡献。

### F. 不同事件阈值

比较：

$$
q=0.60,\ 0.70,\ 0.80
$$

并绘制：

$$
\text{Accuracy vs. event rate}
$$

---

## 23. 当前方案的核心计算逻辑

$$
\boxed{
\text{RGB image}
\rightarrow
\text{3×4 patches}
\rightarrow
\text{12-step snake scan}
\rightarrow
\text{RGB tonic + ON/OFF events}
\rightarrow
\text{fixed mask}
\rightarrow
\text{device dynamics}
\rightarrow
108\text{-D reservoir state}
\rightarrow
\text{linear classifier}
}
$$

其中：

- **空间信息**：由 patch 位置与蛇形扫描顺序保留
- **颜色信息**：由 RGB tonic 通路保留
- **局部变化信息**：由 ON/OFF 事件通路提取
- **时间历史**：由器件尾态和自然衰减保留
- **高维映射**：由 12 行并行动态节点及固定掩码产生
- **分类**：由线性读出层完成

---

## 24. 推荐的第一版仿真默认参数

| 参数            | 推荐初始值                    |
| --------------- | ----------------------------- |
| 图像尺寸        | 64×64 RGB                     |
| Patch           | 3×4                           |
| 时间步          | 12                            |
| Tonic 通道      | 3                             |
| ON 通道         | 3                             |
| OFF 通道        | 3                             |
| 总输入通道      | 9                             |
| Reservoir 行数  | 12                            |
| 状态维度        | 108                           |
| 差分方式        | symmetric relative difference |
| 阈值            | RGB 各自 70% quantile         |
| Event amplitude | 1                             |
| Mask            | fixed random mask             |
| Mask seed       | 42                            |
| 单样本初态      | 0                             |
| 输出层          | ridge / logistic regression   |
| 状态采样        | 最终单一时刻                  |

---

## 25. 推荐工程拆分

建议后续实现拆成三个独立脚本：

```text
01_preprocess_dataset.py
02_run_reservoir.py
03_train_readout.py
```

### `01_preprocess_dataset.py`

负责：

- 图像读取
- 64×64 处理
- 3×4 patch
- 蛇形展开
- RGB 均值
- 相对差分
- ON/OFF
- 保存 `[N,12,9]`

### `02_run_reservoir.py`

负责：

- 固定掩码
- 器件模型
- 状态更新
- 输出 108 维 reservoir states

### `03_train_readout.py`

负责：

- 输出层训练
- validation
- test
- confusion matrix
- accuracy / macro recall
- 消融实验

这样可以保证：

> 数据预处理、器件模型和分类器彼此解耦，后续替换真实器件模型时，不需要重新设计整个数据流水线。
