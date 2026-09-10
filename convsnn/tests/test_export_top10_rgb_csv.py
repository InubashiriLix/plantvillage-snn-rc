from __future__ import annotations

import numpy as np
from PIL import Image

from tools.export_top10_rgb_csv import channel_first_flat, ranked_top10


def test_channel_first_rgb_flattening():
    image = Image.fromarray(np.asarray([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8), "RGB")
    assert channel_first_flat(image, 2).tolist() == [1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6]


def test_top10_tie_breakers():
    classes = {
        f"class_{index:02d}": {
            "index": index, "recall": 0.9 if index < 11 else 0.1,
            "precision": 0.5 + (0.01 if index == 10 else 0),
        }
        for index in range(12)
    }
    ranked = ranked_top10({"selection_metrics": {"per_class": classes}})
    assert ranked[0]["label_name"] == "class_10"
    assert len(ranked) == 10
