from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def tiny_plantvillage(tmp_path: Path) -> Path:
    root = tmp_path / "PlantVillage"
    rng = np.random.default_rng(7)
    for class_index in range(38):
        name = f"class_{class_index:02d}"
        for partition, count in (("train", 3), ("val", 1)):
            directory = root / partition / name
            directory.mkdir(parents=True)
            for image_index in range(count):
                pixels = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
                pixels[..., class_index % 3] = min(255, class_index * 6)
                Image.fromarray(pixels).save(directory / f"{image_index}.png")
    return root
