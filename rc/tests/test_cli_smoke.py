"""Exercise relocated entrypoints from outside the repository on synthetic images."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def test_preprocess_dense_train_and_export_cli(tmp_path):
    data = tmp_path / "data"
    classes = ["synthetic_a", "synthetic_b"]
    rng = np.random.default_rng(10)
    for index, name in enumerate(classes):
        for split, count in (("train", 10), ("val", 3)):
            directory = data / split / name
            directory.mkdir(parents=True)
            for sample in range(count):
                pixels = rng.integers(0, 50, (16, 16, 3), dtype=np.uint8)
                pixels[:, :, index] += 180
                Image.fromarray(pixels).save(directory / f"{sample}.png")
    selected = tmp_path / "classes.json"
    selected.write_text(json.dumps(classes))
    pre, experiment, output = (tmp_path / name for name in ("preprocessed", "experiment", "export"))

    def run(script, *args):
        result = subprocess.run(
            [sys.executable, str(ROOT / "rc/scripts" / script), *map(str, args)],
            cwd=tmp_path, env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    run("01_preprocess_dataset.py", "--data-root", data, "--classes-file", selected,
        "--patch-grid", 8, 8, "--active-columns", 12, "--output-dir", pre)
    run("07_train_dense_v5.py", "--input-dir", pre, "--output-dir", experiment,
        "--device", "cpu", "--quick", "--no-augmentation")
    result = json.loads((experiment / "v5_results.json").read_text())
    assert result["training_device"] == "cpu"
    assert 0 <= result["primary"]["val_accuracy"] <= 1
    assert result["test_evaluated"] == result["reached_target"]
    run("08_export_hardware_inputs.py", "--input-dir", pre, "--experiment-dir", experiment, "--output-dir", output)
    assert np.load(output / "test_inputs12_continuous.npy").shape == (6, 64, 12)
