from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .preprocessing import (
    batch_relative_difference,
    grid_geometry,
    make_graded_rc_inputs,
    make_rc_inputs,
    normalize_texture,
)


METADATA_COLUMNS = (
    "sample_index",
    "sample_id",
    "label_index",
    "class_name",
    "step",
    "frame_index",
    "patch_row",
    "patch_col",
)
INPUT9_COLUMNS = (
    "R",
    "G",
    "B",
    "ON_R",
    "ON_G",
    "ON_B",
    "OFF_R",
    "OFF_G",
    "OFF_B",
)
TEXTURE_COLUMNS = ("luminance_std", "sobel_mean", "entropy_16bin")


def encode_hardware_inputs(
    tonic: np.ndarray,
    texture: np.ndarray,
    *,
    texture_scales: np.ndarray,
    graded_parameters: dict[str, object],
    binary_parameters: dict[str, object],
) -> dict[str, np.ndarray]:
    """Build the three hardware delivery tensors from patch-local base features."""
    difference = batch_relative_difference(tonic)
    continuous9 = make_graded_rc_inputs(
        tonic,
        difference,
        np.asarray(graded_parameters["thresholds"], dtype=np.float32),
        np.asarray(graded_parameters["scales"], dtype=np.float32),
    )
    binary9 = make_rc_inputs(
        tonic,
        difference,
        np.asarray(binary_parameters["thresholds"], dtype=np.float32),
    )
    texture_normalized = normalize_texture(texture, texture_scales)
    continuous12 = np.concatenate([continuous9, texture_normalized], axis=2).astype(
        np.float32, copy=False
    )
    return {
        "inputs9_continuous": continuous9,
        "inputs9_binary_q080": binary9,
        "inputs12_continuous": continuous12,
    }


def write_long_csv(
    path: Path | str,
    inputs: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    classes: list[str],
    scan_order: Iterable[tuple[int, int]],
    *,
    batch_samples: int = 256,
) -> int:
    """Stream an [N,T,C] tensor as one hardware frame per CSV row."""
    path = Path(path)
    scan_order = tuple(tuple(position) for position in scan_order)
    if inputs.ndim != 3 or inputs.shape[1] != len(scan_order):
        raise ValueError("inputs must have shape [N, len(scan_order), C]")
    if inputs.shape[2] not in (9, 12):
        raise ValueError("hardware CSV supports 9 or 12 input channels")
    if len(inputs) != len(labels) or len(inputs) != len(sample_ids):
        raise ValueError("inputs, labels, and sample_ids must have equal length")
    columns = INPUT9_COLUMNS + (TEXTURE_COLUMNS if inputs.shape[2] == 12 else ())
    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(METADATA_COLUMNS + columns)
        for batch_start in range(0, len(inputs), batch_samples):
            batch_stop = min(batch_start + batch_samples, len(inputs))
            rows = []
            for sample_index in range(batch_start, batch_stop):
                label = int(labels[sample_index])
                for frame_index, (patch_row, patch_col) in enumerate(scan_order):
                    values = [f"{value:.7f}" for value in inputs[sample_index, frame_index]]
                    rows.append(
                        [
                            sample_index,
                            str(sample_ids[sample_index]),
                            label,
                            classes[label],
                            frame_index + 1,
                            frame_index,
                            patch_row,
                            patch_col,
                            *values,
                        ]
                    )
            writer.writerows(rows)
            row_count += len(rows)
    return row_count


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _write_readme(
    output_dir: Path,
    *,
    classes: list[str],
    splits: dict[str, object],
    graded: dict[str, object],
    binary: dict[str, object],
    texture_scales: np.ndarray,
    example_rows: list[list[object]],
) -> None:
    class_lines = "\n".join(
        f"| {index} | `{name}` |" for index, name in enumerate(classes)
    )
    split_lines = "\n".join(
        f"| `{name}` | {details['sample_count']:,} | "
        f"{details['sample_count'] * 64:,} | {details['source']} |"
        for name, details in splits.items()
    )
    file_lines = []
    for split, details in splits.items():
        count = details["sample_count"]
        for name, channels in (
            ("inputs9_continuous", 9),
            ("inputs9_binary_q080", 9),
            ("inputs12_continuous", 12),
        ):
            csv_path = output_dir / f"{split}_{name}.csv"
            npy_path = output_dir / f"{split}_{name}.npy"
            file_lines.append(
                f"| `{split}_{name}` | `{count}×64×{channels}` | "
                f"{csv_path.stat().st_size / 1024**2:.1f} MiB | "
                f"{npy_path.stat().st_size / 1024**2:.1f} MiB |"
            )
    files_table = "\n".join(file_lines)
    on_scales, off_scales = graded["scales"]
    thresholds = binary["thresholds"]
    example_lines = "\n".join(
        "| " + " | ".join(str(value) for value in row[4:17]) + " |"
        for row in example_rows[:2]
    )
    document = f"""# PlantVillage 精选十分类：8×8 光电阵列输入数据交接说明

> 文档版本：v2  
> 交付格式：CSV（人工检查和硬件联调）+ NPY（推荐程序批量读取）  
> 单个样本：64 个连续光学时间步；每步同时输入 9 路或 12 路归一化信号

本文档是完整的数据接口约定。接收方不需要阅读训练代码或项目历史，即可选择文件、解析字段，并按正确顺序向器件播放数据。

## 1. 一分钟快速开始

1. 首先使用 `preview_10_samples.csv`：十个类别各包含一个完整的 64 帧样本，适合首次联调。
2. 硬件只有 9 个输入列时，使用 `*_inputs9_continuous.csv`。
3. 硬件 12 列均可使用，且需要复现当前最佳十分类流程时，使用 `*_inputs12_continuous.csv`。
4. 只有硬件无法产生连续事件幅值时，才使用 `*_inputs9_binary_q080.csv`。
5. 同一个 `sample_index` 的 `step=1..64` 必须连续、按顺序播放；两个样本之间必须让器件恢复到基线。

程序读取推荐使用 NPY：

```python
import numpy as np

inputs = np.load("test_inputs9_continuous.npy")  # [4738, 64, 9]
labels = np.load("test_labels.npy")              # [4738]

reset_device_to_baseline()
for frame in inputs[0]:                           # frame.shape == (9,)
    send_channels_simultaneously(frame)
    read_device_state()
```

## 2. 数据集、类别与划分

这是从 PlantVillage 38 类任务中，依据先前验证集的单类召回率选出的十个类别。该子集适合硬件十分类实验，但其结果不能作为无偏的 38 类结果替代品。

| split | 样本数 | 输入帧数 | 来源与用途 |
|---|---:|---:|---|
{split_lines}

类别编号在所有文件中固定：

| label_index | class_name |
|---:|---|
{class_lines}

`sample_index` 只在各自 split 内有效，并从 0 重新开始。跨 split 定位样本时，应使用 `(split, sample_index)` 或唯一的 `sample_id`。

## 3. 从图像到 64 帧硬件信号

```text
原始图像
  │ 转为 RGB，双线性缩放到 64×64，像素除以 255
  ▼
8×8 patch 网格（每个 patch 为 8×8 像素）
  │ 每个 patch 计算 RGB 均值和局部纹理
  ▼
固定蛇形扫描，将二维空间变成 64 个连续时间步
  │ 相邻时间步计算 RGB 相对变化
  ▼
RGB 3 路 + ON 3 路 + OFF 3 路 [+ 纹理 3 路]
  ▼
每步同时送入器件的 9/12 个输入列
```

蛇形时间顺序如下，数字就是 CSV 中的 `step`：

```text
  1 →  2 →  3 →  4 →  5 →  6 →  7 →  8
 16 ← 15 ← 14 ← 13 ← 12 ← 11 ← 10 ←  9
 17 → 18 → 19 → 20 → 21 → 22 → 23 → 24
 32 ← 31 ← 30 ← 29 ← 28 ← 27 ← 26 ← 25
 33 → 34 → 35 → 36 → 37 → 38 → 39 → 40
 48 ← 47 ← 46 ← 45 ← 44 ← 43 ← 42 ← 41
 49 → 50 → 51 → 52 → 53 → 54 → 55 → 56
 64 ← 63 ← 62 ← 61 ← 60 ← 59 ← 58 ← 57
```

因此，`8×8` 不是 64 个并行输入端，而是同一个器件按顺序运行的 64 帧。

### 3.1 RGB 绝对强度

对于时间步 `t` 对应的 8×8 patch，分别取 R、G、B 像素平均值：

```text
C(t) = patch 内颜色通道 C 的平均值，C ∈ {{R,G,B}}
```

三个值均在 `[0,1]`，代表当前区域的绝对颜色和亮度信息。

### 3.2 连续 ON/OFF 事件

相邻 patch 使用对称相对差分：

```text
d_C(t)    = [C(t) - C(t-1)] / [C(t) + C(t-1) + 1e-6]
ON_C(t)   = clip(max(d_C(t), 0) / scale_ON_C,  0, 1)
OFF_C(t)  = clip(max(-d_C(t),0) / scale_OFF_C, 0, 1)
```

本数据采用最佳 `graded_q000` 编码，即不设置额外事件死区。scale 只由训练集统计得到：

| 颜色 | scale_ON | scale_OFF |
|---|---:|---:|
| R | {on_scales[0]:.9f} | {off_scales[0]:.9f} |
| G | {on_scales[1]:.9f} | {off_scales[1]:.9f} |
| B | {on_scales[2]:.9f} | {off_scales[2]:.9f} |

- `step=1` 没有前一个 patch，六路 ON/OFF 固定为 0。
- 同一颜色的 ON 和 OFF 不会同时大于 0。
- 事件幅值越大，表示相邻区域在相应颜色上的变化越强。

### 3.3 二值兼容事件

`inputs9_binary_q080` 保留连续 RGB，但把六路事件变成 0/1。训练集 80% 分位阈值为：

| 颜色 | 阈值 |
|---|---:|
| R | {thresholds[0]:.9f} |
| G | {thresholds[1]:.9f} |
| B | {thresholds[2]:.9f} |

只有 `d_C(t)>threshold_C` 才产生 ON=1；只有 `d_C(t)<-threshold_C` 才产生 OFF=1。该版本用于只能接收开关脉冲的硬件，不是当前最佳模型输入。

### 3.4 三路纹理扩展

12 路版本在前九路之后增加：

| 列 | 含义 | 计算方式 |
|---|---|---|
| `luminance_std` | 局部亮度对比度 | patch 灰度标准差 |
| `sobel_mean` | 局部边缘强度 | patch Sobel 梯度幅值均值 |
| `entropy_16bin` | 局部纹理复杂度 | 16-bin 灰度直方图熵 |

三路特征分别除以训练集 99% 分位数 `{texture_scales[0]:.9f}`、`{texture_scales[1]:.9f}`、`{texture_scales[2]:.9f}`，再裁剪到 `[0,1]`。

## 4. 应该选择哪个输入文件

| 输入版本 | 每帧列数 | 事件形式 | 用途 | 完整复现当前 97.02% 结果 |
|---|---:|---|---|---|
| `inputs9_continuous` | 9 | 连续幅值 | 9 列硬件推荐主表 | 否，缺少三路纹理 |
| `inputs9_binary_q080` | 9 | 0/1 | 只能产生开关事件的兼容硬件 | 否 |
| `inputs12_continuous` | 12 | 连续幅值 | 12 列器件及当前最佳流程 | 是 |

十分类测试准确率 97.02%、宏召回率 96.80%，对应 `inputs12_continuous` 经过 12×12 动态阵列仿真和线性读出后的结果。该指标不能直接标注在九路版本上。

## 5. CSV 数据字典

CSV 使用 UTF-8 编码，第一行是表头；每个后续行对应一个硬件时间步。行首先按 `sample_index` 排序，再按 `step` 排序。

| 字段 | 类型 | 范围/例子 | 说明 |
|---|---|---|---|
| `sample_index` | int | `0..N-1` | 当前 split 内的样本编号 |
| `sample_id` | string | `train/...JPG` | 相对原始 `data/` 的图像路径 |
| `label_index` | int | `0..9` | 固定类别编号 |
| `class_name` | string | `Blueberry___healthy` | PlantVillage 原始类别名 |
| `step` | int | `1..64` | 推荐播放顺序，一基编号 |
| `frame_index` | int | `0..63` | 与 step 对应的零基编号 |
| `patch_row` | int | `0..7` | patch 网格行，零基编号 |
| `patch_col` | int | `0..7` | patch 网格列，零基编号 |
| `R,G,B` | float | `[0,1]` | 当前 patch 的 RGB 平均值 |
| `ON_R..ON_B` | float | `[0,1]` 或 `0/1` | 相邻 patch 的增强事件 |
| `OFF_R..OFF_B` | float | `[0,1]` 或 `0/1` | 相邻 patch 的减弱事件 |
| 三路纹理 | float | `[0,1]` | 仅存在于 12 路文件 |

CSV 浮点数固定保留七位小数。路径或类别名包含逗号时会按标准 CSV 规则加引号；不能用简单的 `split(',')` 解析，必须使用正式 CSV 库。

真实样本前两帧：

| step | frame_index | patch_row | patch_col | R | G | B | ON_R | ON_G | ON_B | OFF_R | OFF_G | OFF_B |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{example_lines}

第一帧没有前序区域，所以事件全为零。第二帧 RGB 均下降，因此变化进入 OFF 列而不是 ON 列。

## 6. 文件清单与 NPY 对齐规则

每个基础名称同时提供 CSV 与 NPY：

| 基础文件名 | NPY shape | CSV 大小 | NPY 大小 |
|---|---|---:|---:|
{files_table}

辅助文件：

| 文件 | 用途 |
|---|---|
| `class_mapping.csv` | `label_index` 与类别名的唯一映射 |
| `samples.csv` | 所有 split 的样本索引、路径与标签 |
| `preview_10_samples.csv` | 每类一个完整样本，共 640 帧；首次联调优先使用 |
| `*_labels.npy` | 与输入 NPY 第一维对齐的 int64 标签 |
| `*_sample_ids.npy` | 与输入 NPY 第一维对齐的原图相对路径 |
| `manifest.json` | 通道、shape、编码参数、字节数和 SHA-256 |

NPY 的轴顺序固定为 `[sample, step, channel]`。例如 `inputs[5,10,3]` 表示第 6 个样本、第 11 帧、`ON_R` 通道。NPY 是 float32 原值，没有 CSV 七位小数带来的舍入误差。

```python
import csv
import numpy as np

x = np.load("train_inputs12_continuous.npy", mmap_mode="r")
y = np.load("train_labels.npy")
assert x.shape == (15162, 64, 12)
assert y.shape == (15162,)

with open("preview_10_samples.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
index = rows[0]["sample_index"]
sample = [row for row in rows if row["sample_index"] == index]
assert [int(row["step"]) for row in sample] == list(range(1, 65))
```

## 7. 硬件播放协议

每个样本执行以下顺序：

1. **恢复基线**：主动复位器件，或等待其充分衰减到规定基线。
2. **连续播放**：从 step 1 到 64 顺序播放，中途不得重排或复位。
3. **同步输入**：每个 step 的 9/12 路数值同时施加到对应输入列。
4. **状态读取**：每帧输入后读取全部 12×12 器件状态；当前最佳模型使用 64 次读取。
5. **结束样本**：保存该样本的 64 帧输出，再执行下一个样本的基线恢复。

```text
列1 R       列2 G       列3 B
列4 ON_R    列5 ON_G    列6 ON_B
列7 OFF_R   列8 OFF_G   列9 OFF_B
列10 luminance_std（12路版本）
列11 sobel_mean（12路版本）
列12 entropy_16bin（12路版本）
```

### 必须由硬件侧标定的参数

CSV 中 `[0,1]` 是归一化逻辑输入，不是伏特、光功率、电流或脉宽。以下参数不由本数据文件规定：

- `[0,1]` 到实际驱动电压、光功率或脉宽的映射；
- 单帧持续时间与帧间隔；
- 每帧状态读取时刻；
- 样本间复位方式及等待时间；
- 实际器件通道增益与漂移补偿。

这些参数必须通过器件标定确定。数据接口只要求映射保持单调、所有样本一致，并保证每个新样本从相同基线开始。

## 8. 完整性检查

`manifest.json` 为每个交付文件记录字节数和 SHA-256：

```python
import hashlib
import json
from pathlib import Path

root = Path(".")
manifest = json.loads((root / "manifest.json").read_text())
name = "test_inputs9_continuous.npy"
actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
assert actual == manifest["files"][name]["sha256"]
```

最小数据自检：

```python
import numpy as np

x9 = np.load("test_inputs9_continuous.npy")
x12 = np.load("test_inputs12_continuous.npy")
xb = np.load("test_inputs9_binary_q080.npy")

assert x9.shape == (4738, 64, 9)
assert x12.shape == (4738, 64, 12)
assert np.array_equal(x9, x12[:, :, :9])
assert np.all(x9[:, 0, 3:9] == 0)
assert not np.any((x9[:, :, 3:6] > 0) & (x9[:, :, 6:9] > 0))
assert set(np.unique(xb[:, :, 3:9])) <= {{0.0, 1.0}}
```

## 9. 常见错误

- 把 8×8 patch 当成 64 个并行端口；实际应按蛇形路径连续运行 64 帧。
- 按普通逐行顺序重排数据；CSV 的 `step` 已经是最终播放顺序。
- 将 RGB 三列也二值化；二值兼容版只有 ON/OFF 六列是 0/1。
- 在同一样本的 64 帧之间复位器件；动态记忆必须在样本内部连续保留。
- 样本之间不恢复基线；这会把前一张图的状态带入下一张图。
- 使用九路输入实验，却引用十二路最佳模型的 97.02% 测试准确率。
- 将归一化值直接当作物理电压；必须先完成器件输入范围标定。

硬件程序首次联调应从 `preview_10_samples.csv` 开始；确认 64 帧顺序、各列同步输入和样本间复位无误后，再切换到完整 split。
"""
    (output_dir / "README_zh.md").write_text(document, encoding="utf-8")


def export_hardware_dataset(
    input_dir: Path | str,
    experiment_dir: Path | str,
    output_dir: Path | str,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, object]:
    input_dir, experiment_dir, output_dir = (
        Path(input_dir),
        Path(experiment_dir),
        Path(output_dir),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # v1 generated compressed duplicates; v2 deliberately delivers CSV + NPY only.
    for stale_gzip in output_dir.glob("*.csv.gz"):
        stale_gzip.unlink()
    config = json.loads((input_dir / "preprocess_config.json").read_text())
    parameters = json.loads((experiment_dir / "encoding_parameters.json").read_text())
    if config.get("patch_grid") != [8, 8] or config.get("time_steps") != 64:
        raise ValueError("hardware export requires the 8x8, 64-step preprocessing")
    graded = parameters["graded_q000"]
    binary = parameters["binary_q080"]
    classes = list(config["classes"])
    scan_order = tuple(tuple(position) for position in config["scan_order"])
    texture_scales = np.asarray(config["texture_scales"], dtype=np.float32)
    generated: list[Path] = []
    split_manifest: dict[str, object] = {}
    all_samples = []
    preview_rows = []

    with (output_dir / "class_mapping.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("label_index", "class_name"))
        writer.writerows(enumerate(classes))
    generated.append(output_dir / "class_mapping.csv")

    for split in splits:
        with np.load(input_dir / f"{split}_base.npz") as archive:
            base = {key: archive[key] for key in archive.files}
        tensors = encode_hardware_inputs(
            base["tonic"],
            base["texture"],
            texture_scales=texture_scales,
            graded_parameters=graded,
            binary_parameters=binary,
        )
        labels = base["labels"]
        sample_ids = base["sample_ids"]
        np.save(output_dir / f"{split}_labels.npy", labels)
        np.save(output_dir / f"{split}_sample_ids.npy", sample_ids)
        generated.extend(
            [
                output_dir / f"{split}_labels.npy",
                output_dir / f"{split}_sample_ids.npy",
            ]
        )
        csv_rows = {}
        for name, tensor in tensors.items():
            npy_path = output_dir / f"{split}_{name}.npy"
            csv_path = output_dir / f"{split}_{name}.csv"
            np.save(npy_path, tensor)
            row_count = write_long_csv(
                csv_path, tensor, labels, sample_ids, classes, scan_order
            )
            generated.extend([npy_path, csv_path])
            csv_rows[name] = row_count
        split_manifest[split] = {
            "sample_count": int(len(labels)),
            "frame_count": 64,
            "source": (
                "原 data/train 的分层训练部分"
                if split == "train"
                else "原 data/train 的分层验证部分"
                if split == "val"
                else "原 data/val，冻结测试集"
            ),
            "class_counts": {
                classes[label]: int(np.sum(labels == label))
                for label in range(len(classes))
            },
            "csv_data_rows": csv_rows,
            "tensor_shapes": {
                name: list(tensor.shape) for name, tensor in tensors.items()
            },
        }
        all_samples.extend(
            (split, index, str(sample_id), int(label), classes[int(label)])
            for index, (sample_id, label) in enumerate(zip(sample_ids, labels))
        )
        if split == "train":
            for label in range(len(classes)):
                selected = int(np.flatnonzero(labels == label)[0])
                for frame_index, (patch_row, patch_col) in enumerate(scan_order):
                    preview_rows.append(
                        [
                            selected,
                            str(sample_ids[selected]),
                            label,
                            classes[label],
                            frame_index + 1,
                            frame_index,
                            patch_row,
                            patch_col,
                            *[
                                f"{value:.7f}"
                                for value in tensors["inputs9_continuous"][
                                    selected, frame_index
                                ]
                            ],
                        ]
                    )

    with (output_dir / "samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("split", "sample_index", "sample_id", "label_index", "class_name")
        )
        writer.writerows(all_samples)
    generated.append(output_dir / "samples.csv")
    with (output_dir / "preview_10_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(METADATA_COLUMNS + INPUT9_COLUMNS)
        writer.writerows(preview_rows)
    generated.append(output_dir / "preview_10_samples.csv")
    _write_readme(
        output_dir,
        classes=classes,
        splits=split_manifest,
        graded=graded,
        binary=binary,
        texture_scales=texture_scales,
        example_rows=preview_rows,
    )
    generated.append(output_dir / "README_zh.md")

    manifest: dict[str, object] = {
        "format_version": 2,
        "dataset": "PlantVillage validation-selected best 10 classes",
        "patch_grid": [8, 8],
        "optical_frames_per_sample": 64,
        "index_conventions": {
            "step": "1-based (1..64)",
            "frame_index": "0-based (0..63)",
            "patch_row": "0-based (0..7)",
            "patch_col": "0-based (0..7)",
        },
        "scan_order": [list(position) for position in scan_order],
        "classes": classes,
        "channels": {
            "inputs9_continuous": list(INPUT9_COLUMNS),
            "inputs9_binary_q080": list(INPUT9_COLUMNS),
            "inputs12_continuous": list(INPUT9_COLUMNS + TEXTURE_COLUMNS),
        },
        "encoding": {
            "continuous": graded,
            "binary_q080": binary,
            "texture_scales": texture_scales.tolist(),
            "value_range": [0.0, 1.0],
            "float_dtype": "float32",
            "csv_decimal_places": 7,
        },
        "splits": split_manifest,
        "files": {},
    }
    manifest["files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(generated)
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
