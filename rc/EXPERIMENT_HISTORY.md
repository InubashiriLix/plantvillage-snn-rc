> 本文件记录迁移前各阶段的实验历史。当前安装及根目录命令见 [README](README.md)；历史 artifacts 路径仅描述本地输出。

# PlantVillage RGB-Tonic ON/OFF Reservoir Computing

This repository implements the Level-1 simulation specified in `note.md` for a
12×12 optoelectronic dynamic array. The current v4 path converts a 64×64 RGB
image into an 8×8 snake scan, drives 12 columns for 64 optical frames, reads all
12 reservoir rows at eight checkpoints, and trains a balanced ridge readout.

## Run

```bash
python rc/scripts/01_preprocess_dataset.py
python rc/scripts/02_run_reservoir.py --device-profile row-spectrum
python rc/scripts/03_train_readout.py
```

The preprocessing command treats the existing `data/val` directory as the
frozen test set and creates a stratified validation fold from `data/train`.
Generated arrays and configurations are written below `artifacts/`. The v2
experiments use heterogeneous row decay, a class-balanced SVD Ridge readout,
and macro recall as the primary validation objective; their reports are saved
to `artifacts/experiments_v2/` without replacing the v1 baseline.

The current full 38-class v2 run selected `q=0.60`, an alpha row spectrum from
0.0 to 0.9, `gamma=0.5`, and Ridge alpha `1e-4`. It reached 49.19% validation
accuracy / 49.26% validation macro recall and 48.70% test accuracy / 47.68%
test macro recall while predicting all 38 classes.

The v3 hardware path adds a second 12-step texture scan without resetting the
reservoir between passes:

```bash
python rc/scripts/01_preprocess_dataset.py
python rc/scripts/04_train_dual_pass.py
```

It uses local contrast, Sobel edge strength, and entropy as the three texture
tonic channels and concatenates the RGB and inherited texture tail states into
216 features. If validation macro recall stays below 80%, the frozen test split
is not evaluated. A separate pretrained CNN ceiling check, which is explicitly
not counted as an RC result, is available through `05_cnn_upper_bound.py`.

The current full-data v3 stage run reached 61.96% validation accuracy and
62.09% validation macro recall. This is a 12.83-point macro-recall improvement
over v2, but it is below the predefined 65% continuation gate, so the v3 code
did not access or evaluate the frozen test split.

The v4 spatial path preserves much more image structure while requiring only
64 optical frames instead of 4096 pixel-by-pixel frames:

```bash
python rc/scripts/01_preprocess_dataset.py --patch-grid 8 8 --active-columns 12 \
  --output-dir artifacts/preprocessed_v4
python rc/scripts/06_train_spatial_v4.py --device cuda \
  --input-dir artifacts/preprocessed_v4 \
  --output-dir artifacts/experiments_v4
```

Columns 1–9 carry RGB tonic/ON/OFF signals; columns 10–12 carry local contrast,
Sobel strength, and entropy. Rows use a spectrum of time constants, and the
readout compares final-state and eight-checkpoint variants. The balanced ridge
normal equations are solved on CUDA by default, while the exported model uses
NumPy for inference. Use `--device cpu` only as an explicit fallback.

The full 38-class v4 search selected 12 columns, eight checkpoint reads,
`q=0.80`, maximum time constant 128, input gain 2.0, and ridge alpha 1.0. It
reached **72.02% validation accuracy / 72.17% validation macro recall**, a
10.08-point macro-recall gain over v3. Since this remains below the predefined
80% gate, the frozen test set was not evaluated.

The v5 dense-read path keeps the same 64 optical frames, replaces binary events
with normalized continuous ON/OFF magnitudes, and electrically reads the 12×12
state after every frame:

```bash
python rc/scripts/07_train_dense_v5.py --device cuda \
  --input-dir artifacts/preprocessed_v4 \
  --output-dir artifacts/experiments_v5
```

It screens event/dynamics settings with 16 reads, then trains a class-balanced
9216-to-38 linear Softmax readout on the full state trace. State arrays are
written in batches as NumPy memmaps. Training-only horizontal/vertical flips
are evaluated automatically and retained only if validation macro recall
improves.

The full 38-class v5 run selected continuous events without a dead zone,
maximum time constant 128, input gain 2.0, weight decay `1e-3`, and 46 epochs.
It reached **79.91% validation accuracy / 77.23% validation macro recall**, a
5.06-point macro-recall gain over v4. Flip augmentation did not improve the
validation result and was rejected. Since macro recall remained below 80%, the
frozen test split was not evaluated.

For a fast integration run before processing all images:

```bash
python rc/scripts/01_preprocess_dataset.py --max-per-class 10 --output-dir /tmp/plantvillage-smoke/preprocessed
python rc/scripts/03_train_readout.py --quick --input-dir /tmp/plantvillage-smoke/preprocessed --output-dir /tmp/plantvillage-smoke/experiments
```

For the natural 10-class tomato subset (nine diseases plus healthy), use:

```bash
python rc/scripts/01_preprocess_dataset.py --class-prefix 'Tomato___' --output-dir artifacts_tomato10/preprocessed
python rc/scripts/03_train_readout.py --input-dir artifacts_tomato10/preprocessed --output-dir artifacts_tomato10/experiments_v2
python rc/scripts/04_train_dual_pass.py --input-dir artifacts_tomato10/preprocessed --output-dir artifacts_tomato10/experiments_v3
```

The current tomato-only run contains 11,623 training, 2,906 validation, and
3,631 frozen test images. The v2 single-pass model reached 61.70% validation
macro recall. The v3 dual-pass model reached 74.67% validation accuracy and
71.69% validation macro recall, predicting all ten classes. Because it did not
reach the configured 80% target, v3 did not score the test split.

For the validation-selected best-performing ten classes, the fixed class list
is stored in `artifacts_best10/preprocessed_v4/preprocess_config.json`. The v5
run contains 15,162 training, 3,791 validation, and 4,738 frozen test images.
It reached **97.36% validation accuracy / 97.98% validation macro recall** and,
after passing the 80% gate, **97.02% test accuracy / 96.80% test macro recall**.
These figures describe a subset selected using the earlier 38-class validation
report and must not be compared as an unbiased replacement for the 38-class
result.

Export that fixed ten-class subset as hardware-ready, one-frame-per-row CSV
files plus lossless NumPy tensors with:

```bash
python rc/scripts/08_export_hardware_inputs.py
```

The default output is `artifacts_best10/hardware_export_v1/`. Its Chinese
README and `manifest.json` define the nine-column continuous hardware input,
the binary-event compatibility input, the twelve-column texture extension,
class mapping, scan coordinates, shapes, and SHA-256 checksums.

## Tests

```bash
python -m unittest discover -s rc/tests -p test_pipeline.py -v
```
