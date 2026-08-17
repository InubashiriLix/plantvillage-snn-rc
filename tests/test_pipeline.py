from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from plantvillage_rc.preprocessing import (
    PATCH_X_EDGES,
    PATCH_Y_EDGES,
    SCAN_ORDER,
    compute_thresholds,
    extract_texture,
    extract_tonic,
    make_rc_inputs,
    normalize_texture,
    preprocess_dataset,
    symmetric_relative_difference,
)
from plantvillage_rc.experiments import _selection_key
from plantvillage_rc.reservoir import (
    dual_pass_states,
    generate_mask,
    reservoir_states,
    row_spectrum,
)


class FeatureTests(unittest.TestCase):
    def test_snake_scan_and_patch_boundaries(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.float32)
        for row in range(3):
            for col in range(4):
                value = row * 4 + col
                image[
                    PATCH_Y_EDGES[row] : PATCH_Y_EDGES[row + 1],
                    PATCH_X_EDGES[col] : PATCH_X_EDGES[col + 1],
                ] = value
        tonic = extract_tonic(image)
        expected = np.asarray([row * 4 + col for row, col in SCAN_ORDER])
        np.testing.assert_allclose(tonic[:, 0], expected)
        np.testing.assert_allclose(tonic[:, 0], tonic[:, 1])

    def test_relative_difference_and_first_step(self) -> None:
        tonic = np.ones((12, 3), dtype=np.float32)
        tonic[1] = 3.0
        diff = symmetric_relative_difference(tonic)
        np.testing.assert_array_equal(diff[0], 0.0)
        np.testing.assert_allclose(diff[1], 0.5, atol=1e-6)
        np.testing.assert_allclose(diff[2], -0.5, atol=1e-6)

    def test_events_are_binary_and_first_step_zero(self) -> None:
        tonic = np.full((2, 12, 3), 0.5, dtype=np.float32)
        diff = np.zeros_like(tonic)
        diff[:, 1, 0] = 0.3
        diff[:, 1, 1] = -0.3
        inputs = make_rc_inputs(tonic, diff, np.array([0.2, 0.2, 0.2]))
        np.testing.assert_array_equal(inputs[:, 0, 3:], 0.0)
        np.testing.assert_array_equal(inputs[:, 1, 3:6], [[1, 0, 0], [1, 0, 0]])
        np.testing.assert_array_equal(inputs[:, 1, 6:9], [[0, 1, 0], [0, 1, 0]])

    def test_threshold_shape(self) -> None:
        diff = np.zeros((3, 12, 3), dtype=np.float32)
        diff[:, 1:, 0] = np.arange(11)
        thresholds = compute_thresholds(diff, 0.7)
        self.assertEqual(thresholds.shape, (3,))
        self.assertGreater(thresholds[0], thresholds[1])

    def test_texture_features_respond_to_structure(self) -> None:
        constant = np.full((64, 64, 3), 0.5, dtype=np.float32)
        edge = constant.copy()
        edge[:, 32:] = 1.0
        checker = np.indices((64, 64)).sum(axis=0) % 2
        checker = np.repeat(checker[:, :, None], 3, axis=2).astype(np.float32)
        constant_features = extract_texture(constant)
        edge_features = extract_texture(edge)
        checker_features = extract_texture(checker)
        np.testing.assert_allclose(constant_features, 0.0, atol=1e-6)
        self.assertGreater(edge_features[:, 1].max(), 0.0)
        self.assertGreater(checker_features[:, 0].mean(), 0.0)
        self.assertGreater(checker_features[:, 2].mean(), 0.0)

    def test_texture_normalization_clips_to_unit_interval(self) -> None:
        texture = np.ones((2, 12, 3), dtype=np.float32)
        normalized = normalize_texture(texture, np.array([2.0, 1.0, 0.5]))
        np.testing.assert_allclose(normalized[0, 0], [0.5, 1.0, 1.0])


class ReservoirTests(unittest.TestCase):
    def test_shapes_determinism_and_variants(self) -> None:
        inputs = np.ones((4, 12, 9), dtype=np.float32)
        mask = generate_mask(42)
        np.testing.assert_array_equal(mask, generate_mask(42))
        full = reservoir_states(inputs, mask)
        tonic = reservoir_states(inputs, mask, variant="tonic")
        events = reservoir_states(inputs, mask, variant="event")
        self.assertEqual(full.shape, (4, 108))
        self.assertEqual(tonic.shape, (4, 36))
        self.assertEqual(events.shape, (4, 72))
        np.testing.assert_allclose(full[0], full[1])

    def test_no_memory_depends_only_on_final_step(self) -> None:
        mask = generate_mask(42)
        first = np.zeros((1, 12, 9), dtype=np.float32)
        second = first.copy()
        first[:, :11] = 1.0
        first[:, 11] = 0.25
        second[:, 11] = 0.25
        np.testing.assert_allclose(
            reservoir_states(first, mask, alpha=0.0),
            reservoir_states(second, mask, alpha=0.0),
        )

    def test_row_spectrum_is_deterministic_and_heterogeneous(self) -> None:
        alpha = row_spectrum(0.05, 0.98)
        self.assertEqual(alpha.shape, (12, 9))
        np.testing.assert_allclose(alpha[0], 0.05)
        np.testing.assert_allclose(alpha[-1], 0.98)
        self.assertTrue(np.all(np.diff(alpha[:, 0]) > 0))

        inputs = np.zeros((1, 12, 9), dtype=np.float32)
        inputs[:, :11] = 1.0
        states = reservoir_states(
            inputs,
            np.ones((12, 9), dtype=np.float32),
            alpha=alpha,
            gamma=1.0,
        ).reshape(1, 12, 9)
        self.assertGreater(np.unique(states[0, :, 0]).size, 1)

    def test_node_parameter_shape_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            reservoir_states(
                np.ones((1, 12, 9), dtype=np.float32),
                generate_mask(42),
                alpha=np.ones((12, 8), dtype=np.float32),
            )

    def test_dual_pass_carries_state_and_returns_216_features(self) -> None:
        rgb = np.ones((2, 12, 9), dtype=np.float32)
        texture = np.zeros_like(rgb)
        mask = np.ones((12, 9), dtype=np.float32)
        carried = dual_pass_states(
            rgb, texture, mask, alpha=row_spectrum(0.1, 0.9), gamma=1.0
        )
        reset = dual_pass_states(
            rgb,
            texture,
            mask,
            alpha=row_spectrum(0.1, 0.9),
            gamma=1.0,
            carry_state=False,
        )
        self.assertEqual(carried["combined_states"].shape, (2, 216))
        self.assertTrue(np.any(carried["texture_states"] > 0))
        np.testing.assert_array_equal(reset["texture_states"], 0.0)


class SelectionTests(unittest.TestCase):
    def test_macro_recall_is_primary_selection_metric(self) -> None:
        high_accuracy = {
            "val_accuracy": 0.80,
            "val_macro_recall": 0.30,
            "val_predicted_class_count": 20,
        }
        balanced = {
            "val_accuracy": 0.50,
            "val_macro_recall": 0.45,
            "val_predicted_class_count": 38,
        }
        self.assertGreater(_selection_key(balanced), _selection_key(high_accuracy))


class EndToEndTests(unittest.TestCase):
    def test_small_dataset_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            for split, count in (("train", 5), ("val", 2)):
                for class_index, class_name in enumerate(("class_a", "class_b")):
                    directory = data / split / class_name
                    directory.mkdir(parents=True)
                    for image_index in range(count):
                        value = 30 + 80 * class_index + image_index
                        array = np.full((32, 32, 3), value, dtype=np.uint8)
                        Image.fromarray(array).save(directory / f"{image_index}.jpg")
            output = root / "artifacts"
            config = preprocess_dataset(data, output)
            self.assertEqual(config["split_sizes"], {"train": 8, "val": 2, "test": 4})
            self.assertEqual(np.load(output / "train_inputs.npy").shape, (8, 12, 9))
            self.assertEqual(np.load(output / "test_labels.npy").shape, (4,))
            self.assertEqual(
                np.load(output / "train_texture_inputs.npy").shape, (8, 12, 9)
            )
            with (output / "preprocess_config.json").open() as handle:
                persisted = json.load(handle)
            self.assertEqual(persisted["classes"], ["class_a", "class_b"])
            self.assertEqual(len(persisted["texture_scales"]), 3)


if __name__ == "__main__":
    unittest.main()
