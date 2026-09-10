import csv
from pathlib import Path

import numpy as np
import pytest

from plantvillage_rc.hardware_preview import read_preview

BASE = Path(__file__).resolve().parents[1] / "hardware_example"


def test_shipped_preview():
    rgb, labels, signals = read_preview(BASE / "preview_10_samples.csv", BASE / "class_mapping.csv")
    assert rgb.shape == (10, 192)
    np.testing.assert_array_equal(labels, np.arange(10))
    np.testing.assert_array_equal(rgb[0, :6], signals[0, :2, :3].flatten())


@pytest.mark.parametrize("field,value", [("step", "3"), ("label_index", "9"), ("patch_col", "7"), ("ON_R", "nan")])
def test_reject_corrupt_handoff(tmp_path, field, value):
    with (BASE / "preview_10_samples.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    rows[1][field] = value
    path = tmp_path / "corrupt.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        read_preview(path, BASE / "class_mapping.csv")
