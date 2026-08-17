# PlantVillage RGB-Tonic ON/OFF Reservoir Computing

This repository implements the Level-1 simulation specified in `note.md`: a
64×64 RGB image is converted into a 12-step snake scan, encoded as tonic and
ON/OFF pathways, passed through a fixed 12×9 dynamic reservoir, and classified
with a ridge readout.

## Run

```bash
python 01_preprocess_dataset.py
python 02_run_reservoir.py --device-profile row-spectrum
python 03_train_readout.py
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
python 01_preprocess_dataset.py
python 04_train_dual_pass.py
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

For a fast integration run before processing all images:

```bash
python 01_preprocess_dataset.py --max-per-class 10 --output-dir /tmp/plantvillage-smoke/preprocessed
python 03_train_readout.py --quick --input-dir /tmp/plantvillage-smoke/preprocessed --output-dir /tmp/plantvillage-smoke/experiments
```

## Tests

```bash
python -m unittest discover -s tests -v
```
