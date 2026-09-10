from __future__ import annotations

import json
import csv
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
    compute_graded_event_scales,
    extract_texture,
    extract_tonic,
    grid_geometry,
    flip_scanned_features,
    make_graded_rc_inputs,
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
    reservoir_states_to_memmap,
    row_spectrum,
    time_constant_spectrum,
)
from plantvillage_rc.v4_experiments import TorchRidgeClassifier
from plantvillage_rc.v5_experiments import train_linear_softmax
from plantvillage_rc.hardware_export import (
    encode_hardware_inputs,
    export_hardware_dataset,
    write_long_csv,
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

    def test_graded_events_preserve_magnitude_and_are_mutually_exclusive(self) -> None:
        tonic = np.full((2, 4, 3), 0.5, dtype=np.float32)
        diff = np.zeros_like(tonic)
        diff[:, 1, 0] = 0.2
        diff[:, 2, 0] = 0.4
        diff[:, 3, 1] = -0.3
        thresholds = np.asarray([0.1, 0.1, 0.1], dtype=np.float32)
        scales = compute_graded_event_scales(diff, thresholds, 1.0)
        inputs = make_graded_rc_inputs(tonic, diff, thresholds, scales)
        self.assertEqual(inputs.shape, (2, 4, 9))
        np.testing.assert_array_equal(inputs[:, 0, 3:], 0.0)
        self.assertGreater(inputs[0, 2, 3], inputs[0, 1, 3])
        self.assertGreater(inputs[0, 3, 7], 0.0)
        self.assertFalse(np.any((inputs[:, :, 3:6] > 0) & (inputs[:, :, 6:9] > 0)))

    def test_flip_features_restores_snake_order(self) -> None:
        _, _, order = grid_geometry((2, 2))
        values = np.asarray([[row * 2 + col for row, col in order]], dtype=np.float32)
        values = values[:, :, None]
        horizontal = flip_scanned_features(values, (2, 2), "horizontal")
        np.testing.assert_array_equal(horizontal[0, :, 0], [1, 0, 2, 3])

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

    def test_eight_by_eight_grid_has_64_snake_steps(self) -> None:
        y_edges, x_edges, order = grid_geometry((8, 8))
        self.assertEqual(y_edges, tuple(range(0, 65, 8)))
        self.assertEqual(x_edges, tuple(range(0, 65, 8)))
        self.assertEqual(len(order), 64)
        self.assertEqual(order[:8], tuple((0, col) for col in range(8)))
        self.assertEqual(order[8:16], tuple((1, col) for col in range(7, -1, -1)))
        image = np.zeros((64, 64, 3), dtype=np.float32)
        for index, (row, column) in enumerate(order):
            image[row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8] = index
        tonic = extract_tonic(image, (8, 8))
        self.assertEqual(tonic.shape, (64, 3))
        np.testing.assert_allclose(tonic[:, 0], np.arange(64))


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

    def test_64_step_12_channel_checkpoints_match_final_state(self) -> None:
        inputs = np.random.default_rng(42).random((2, 64, 12), dtype=np.float32)
        mask = generate_mask(42, channels=12)
        alpha = time_constant_spectrum(1.0, 128.0, channels=12)
        final_state = reservoir_states(inputs, mask, alpha=alpha, gamma=1.0)
        checkpoints = reservoir_states(
            inputs,
            mask,
            alpha=alpha,
            gamma=1.0,
            checkpoint_steps=(8, 16, 24, 32, 40, 48, 56, 64),
        )
        self.assertEqual(final_state.shape, (2, 144))
        self.assertEqual(checkpoints.shape, (2, 1152))
        np.testing.assert_allclose(checkpoints.reshape(2, 8, 144)[:, -1], final_state)

    def test_time_constant_spectrum_spans_short_and_long_memory(self) -> None:
        alpha = time_constant_spectrum(1.0, 128.0, channels=12)
        self.assertEqual(alpha.shape, (12, 12))
        self.assertTrue(np.all(np.diff(alpha[:, 0]) > 0))
        np.testing.assert_allclose(alpha[0, 0], np.exp(-1.0), rtol=1e-6)
        np.testing.assert_allclose(alpha[-1, 0], np.exp(-1.0 / 128.0), rtol=1e-6)

    def test_memmap_states_match_in_memory_states(self) -> None:
        inputs = np.random.default_rng(3).random((5, 8, 12), dtype=np.float32)
        mask = generate_mask(42, channels=12)
        checkpoints = (2, 4, 8)
        expected = reservoir_states(
            inputs, mask, alpha=0.8, checkpoint_steps=checkpoints
        )
        with tempfile.TemporaryDirectory() as temporary:
            actual = reservoir_states_to_memmap(
                inputs,
                Path(temporary) / "states.npy",
                mask,
                alpha=0.8,
                gamma=1.0,
                checkpoint_steps=checkpoints,
                batch_size=2,
            )
            np.testing.assert_allclose(actual, expected)


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

    def test_torch_ridge_cpu_fallback_learns_separable_classes(self) -> None:
        features = np.asarray(
            [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1])
        model = TorchRidgeClassifier(alpha=0.01, device="cpu").fit(features, labels)
        np.testing.assert_array_equal(model.predict(features), labels)

    def test_linear_softmax_cpu_learns_separable_classes(self) -> None:
        features = np.repeat(
            np.asarray([[-2.0, -1.0], [2.0, 1.0]], dtype=np.float32), 20, axis=0
        )
        labels = np.repeat(np.asarray([0, 1]), 20)
        model, _ = train_linear_softmax(
            features,
            labels,
            class_count=2,
            weight_decay=1e-4,
            device="cpu",
            fixed_epochs=20,
            batch_size=8,
        )
        np.testing.assert_array_equal(model.predict(features), labels)


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
            self.assertEqual(np.load(output / "train_inputs12.npy").shape, (8, 12, 12))
            with (output / "preprocess_config.json").open() as handle:
                persisted = json.load(handle)
            self.assertEqual(persisted["classes"], ["class_a", "class_b"])
            self.assertEqual(len(persisted["texture_scales"]), 3)

    def test_class_prefix_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            for split, count in (("train", 5), ("val", 2)):
                for class_name in ("Tomato___healthy", "Apple___healthy"):
                    directory = data / split / class_name
                    directory.mkdir(parents=True)
                    for image_index in range(count):
                        Image.fromarray(
                            np.full((16, 16, 3), 80 + image_index, dtype=np.uint8)
                        ).save(directory / f"{image_index}.jpg")
            output = root / "filtered"
            config = preprocess_dataset(data, output, class_prefix="Tomato___")
            self.assertEqual(config["classes"], ["Tomato___healthy"])
            self.assertEqual(config["split_sizes"], {"train": 4, "val": 1, "test": 2})


class HardwareExportTests(unittest.TestCase):
    def test_hardware_encodings_have_expected_channels_and_event_rules(self) -> None:
        tonic = np.asarray(
            [[[0.2, 0.3, 0.4], [0.4, 0.2, 0.4], [0.1, 0.5, 0.3]]],
            dtype=np.float32,
        )
        texture = np.full((1, 3, 3), 0.25, dtype=np.float32)
        tensors = encode_hardware_inputs(
            tonic,
            texture,
            texture_scales=np.ones(3, dtype=np.float32),
            graded_parameters={
                "thresholds": [0.0, 0.0, 0.0],
                "scales": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            },
            binary_parameters={"thresholds": [0.1, 0.1, 0.1]},
        )
        self.assertEqual(tensors["inputs9_continuous"].shape, (1, 3, 9))
        self.assertEqual(tensors["inputs9_binary_q080"].shape, (1, 3, 9))
        self.assertEqual(tensors["inputs12_continuous"].shape, (1, 3, 12))
        np.testing.assert_array_equal(
            tensors["inputs12_continuous"][:, :, :9],
            tensors["inputs9_continuous"],
        )
        np.testing.assert_array_equal(tensors["inputs9_continuous"][:, 0, 3:], 0)
        continuous = tensors["inputs9_continuous"]
        self.assertFalse(np.any((continuous[:, :, 3:6] > 0) & (continuous[:, :, 6:9] > 0)))
        self.assertTrue(
            set(np.unique(tensors["inputs9_binary_q080"][:, :, 3:]).tolist())
            <= {0.0, 1.0}
        )

    def test_long_csv_round_trips_tensor_values(self) -> None:
        inputs = np.arange(2 * 4 * 9, dtype=np.float32).reshape(2, 4, 9) / 100
        labels = np.asarray([0, 1])
        sample_ids = np.asarray(["a.jpg", "b.jpg"])
        order = ((0, 0), (0, 1), (1, 1), (1, 0))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.csv"
            rows = write_long_csv(
                path, inputs, labels, sample_ids, ["a", "b"], order
            )
            self.assertEqual(rows, 8)
            with path.open() as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 8)
            self.assertEqual(records[0]["step"], "1")
            self.assertEqual(records[3]["patch_col"], "0")
            reconstructed = np.asarray(
                [[float(record[name]) for name in (
                    "R", "G", "B", "ON_R", "ON_G", "ON_B",
                    "OFF_R", "OFF_G", "OFF_B",
                )] for record in records],
                dtype=np.float32,
            ).reshape(2, 4, 9)
            np.testing.assert_allclose(reconstructed, inputs, atol=5e-8)

    def test_small_hardware_export_writes_manifest_and_all_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir, experiment_dir, output_dir = (
                root / "input", root / "experiment", root / "output"
            )
            input_dir.mkdir()
            experiment_dir.mkdir()
            _, _, order = grid_geometry((8, 8))
            config = {
                "patch_grid": [8, 8],
                "time_steps": 64,
                "classes": ["class_a", "class_b"],
                "scan_order": order,
                "texture_scales": [1.0, 1.0, 1.0],
            }
            (input_dir / "preprocess_config.json").write_text(json.dumps(config))
            parameters = {
                "graded_q000": {
                    "thresholds": [0.0, 0.0, 0.0],
                    "scales": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                },
                "binary_q080": {"thresholds": [0.1, 0.1, 0.1]},
            }
            (experiment_dir / "encoding_parameters.json").write_text(
                json.dumps(parameters)
            )
            output_dir.mkdir()
            (output_dir / "stale.csv.gz").write_bytes(b"obsolete")
            tonic = np.full((2, 64, 3), 0.4, dtype=np.float32)
            texture = np.full_like(tonic, 0.2)
            for split in ("train", "val", "test"):
                np.savez(
                    input_dir / f"{split}_base.npz",
                    tonic=tonic,
                    diff=np.zeros_like(tonic),
                    texture=texture,
                    labels=np.asarray([0, 1]),
                    sample_ids=np.asarray([f"{split}/a.jpg", f"{split}/b.jpg"]),
                )
            manifest = export_hardware_dataset(input_dir, experiment_dir, output_dir)
            self.assertEqual(manifest["splits"]["train"]["sample_count"], 2)
            self.assertEqual(
                manifest["splits"]["train"]["csv_data_rows"]["inputs9_continuous"],
                128,
            )
            self.assertTrue((output_dir / "manifest.json").is_file())
            self.assertTrue((output_dir / "preview_10_samples.csv").is_file())
            self.assertFalse(any(output_dir.glob("*.gz")))
            readme = (output_dir / "README_zh.md").read_text()
            for heading in (
                "## 1. 一分钟快速开始",
                "## 3. 从图像到 64 帧硬件信号",
                "## 5. CSV 数据字典",
                "## 7. 硬件播放协议",
                "## 9. 常见错误",
            ):
                self.assertIn(heading, readme)
            self.assertNotIn(".csv.gz", readme)
            np.testing.assert_array_equal(
                np.load(output_dir / "test_inputs9_continuous.npy")[:, 0, 3:], 0
            )


if __name__ == "__main__":
    unittest.main()
