from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


IMAGE_SIZE = (64, 64)
PATCH_GRID = (3, 4)
PATCH_Y_EDGES = (0, 21, 42, 64)
PATCH_X_EDGES = (0, 16, 32, 48, 64)
SCAN_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (1, 2), (1, 1), (1, 0),
    (2, 0), (2, 1), (2, 2), (2, 3),
)
EPSILON = 1e-6
TEXTURE_FEATURES = ("luminance_std", "sobel_mean", "entropy_16bin")


@dataclass(frozen=True)
class Sample:
    path: Path
    sample_id: str
    label: int


@dataclass(frozen=True)
class PreprocessSettings:
    image_size: tuple[int, int] = IMAGE_SIZE
    patch_grid: tuple[int, int] = PATCH_GRID
    y_edges: tuple[int, ...] = PATCH_Y_EDGES
    x_edges: tuple[int, ...] = PATCH_X_EDGES
    scan_order: tuple[tuple[int, int], ...] = SCAN_ORDER
    difference: str = "symmetric_relative"
    epsilon: float = EPSILON
    threshold_quantiles: tuple[float, ...] = (0.6, 0.7, 0.8)
    default_quantile: float = 0.7
    event_amplitude: float = 1.0
    active_columns: int = 9
    reservoir_rows: int = 12
    mask_seed: int = 42
    split_seed: int = 42
    validation_fraction: float = 0.2


def discover_classes(data_root: Path) -> list[str]:
    train_root = data_root / "train"
    test_root = data_root / "val"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(
            f"Expected class directories below {train_root} and {test_root}"
        )
    train_classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    test_classes = sorted(p.name for p in test_root.iterdir() if p.is_dir())
    if not train_classes:
        raise ValueError(f"No classes found in {train_root}")
    if train_classes != test_classes:
        missing_test = sorted(set(train_classes) - set(test_classes))
        missing_train = sorted(set(test_classes) - set(train_classes))
        raise ValueError(
            "Train/test class directories differ: "
            f"missing from test={missing_test}, missing from train={missing_train}"
        )
    return train_classes


def _image_paths(class_dir: Path) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed
    )


def build_samples(
    split_root: Path,
    classes: Sequence[str],
    *,
    data_root: Path,
    max_per_class: int | None = None,
) -> list[Sample]:
    samples: list[Sample] = []
    for label, class_name in enumerate(classes):
        paths = _image_paths(split_root / class_name)
        if max_per_class is not None:
            paths = paths[:max_per_class]
        if not paths:
            raise ValueError(f"No supported images found in {split_root / class_name}")
        samples.extend(
            Sample(
                path=path,
                sample_id=path.relative_to(data_root).as_posix(),
                label=label,
            )
            for path in paths
        )
    return samples


def stratified_development_split(
    samples: Sequence[Sample], validation_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    indices = np.arange(len(samples))
    labels = np.asarray([sample.label for sample in samples])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels,
    )
    return (
        [samples[int(i)] for i in sorted(train_idx)],
        [samples[int(i)] for i in sorted(val_idx)],
    )


def grid_geometry(
    patch_grid: tuple[int, int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]]:
    rows, columns = patch_grid
    if rows <= 0 or columns <= 0 or rows > 64 or columns > 64:
        raise ValueError("patch_grid dimensions must be between 1 and 64")
    y_edges = tuple(np.linspace(0, 64, rows + 1, dtype=int).tolist())
    x_edges = tuple(np.linspace(0, 64, columns + 1, dtype=int).tolist())
    order = []
    for row in range(rows):
        column_order = range(columns) if row % 2 == 0 else range(columns - 1, -1, -1)
        order.extend((row, column) for column in column_order)
    return y_edges, x_edges, tuple(order)


def extract_tonic(
    image: np.ndarray, patch_grid: tuple[int, int] = PATCH_GRID
) -> np.ndarray:
    if image.shape != (64, 64, 3):
        raise ValueError(f"Expected image shape (64, 64, 3), got {image.shape}")
    y_edges, x_edges, scan_order = grid_geometry(patch_grid)
    patches = []
    for row, col in scan_order:
        patch = image[
            y_edges[row] : y_edges[row + 1],
            x_edges[col] : x_edges[col + 1],
        ]
        patches.append(patch.mean(axis=(0, 1), dtype=np.float64))
    return np.asarray(patches, dtype=np.float32)


def symmetric_relative_difference(
    tonic: np.ndarray, epsilon: float = EPSILON
) -> np.ndarray:
    if tonic.ndim != 2 or tonic.shape[1] != 3:
        raise ValueError(f"Expected tonic shape [T, 3], got {tonic.shape}")
    diff = np.zeros_like(tonic, dtype=np.float32)
    diff[1:] = (tonic[1:] - tonic[:-1]) / (
        tonic[1:] + tonic[:-1] + epsilon
    )
    return diff


def _sobel_magnitude(grayscale: np.ndarray) -> np.ndarray:
    padded = np.pad(grayscale, 1, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    kernel_x = np.asarray(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    kernel_y = kernel_x.T
    gradient_x = np.einsum("ijkl,kl->ij", windows, kernel_x, optimize=True)
    gradient_y = np.einsum("ijkl,kl->ij", windows, kernel_y, optimize=True)
    return np.hypot(gradient_x, gradient_y)


def extract_texture(
    image: np.ndarray, patch_grid: tuple[int, int] = PATCH_GRID
) -> np.ndarray:
    if image.shape != (64, 64, 3):
        raise ValueError(f"Expected image shape (64, 64, 3), got {image.shape}")
    grayscale = np.tensordot(
        image,
        np.asarray([0.299, 0.587, 0.114], dtype=np.float32),
        axes=([2], [0]),
    )
    gradient = _sobel_magnitude(grayscale)
    y_edges, x_edges, scan_order = grid_geometry(patch_grid)
    features = []
    for row, col in scan_order:
        y0, y1 = y_edges[row], y_edges[row + 1]
        x0, x1 = x_edges[col], x_edges[col + 1]
        gray_patch = grayscale[y0:y1, x0:x1]
        gradient_patch = gradient[y0:y1, x0:x1]
        histogram, _ = np.histogram(gray_patch, bins=16, range=(0.0, 1.0))
        probabilities = histogram[histogram > 0].astype(np.float64)
        probabilities /= probabilities.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        features.append(
            (float(gray_patch.std()), float(gradient_patch.mean()), float(entropy))
        )
    return np.asarray(features, dtype=np.float32)


def normalize_texture(texture: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if texture.ndim != 3 or texture.shape[2] != 3:
        raise ValueError(f"Expected texture shape [N, T, 3], got {texture.shape}")
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape != (3,) or np.any(scales <= 0):
        raise ValueError("Texture scales must contain three positive values")
    return np.clip(texture / scales.reshape(1, 1, 3), 0.0, 1.0).astype(np.float32)


def batch_relative_difference(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"Expected values shape [N, T, 3], got {values.shape}")
    diff = np.zeros_like(values, dtype=np.float32)
    diff[:, 1:] = (values[:, 1:] - values[:, :-1]) / (
        values[:, 1:] + values[:, :-1] + epsilon
    )
    return diff


def load_image_features(
    path: Path, patch_grid: tuple[int, int] = PATCH_GRID
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unable to decode image {path}: {exc}") from exc
    tonic = extract_tonic(array, patch_grid)
    return (
        tonic,
        symmetric_relative_difference(tonic),
        extract_texture(array, patch_grid),
    )


def extract_split(
    samples: Sequence[Sample], patch_grid: tuple[int, int] = PATCH_GRID
) -> dict[str, np.ndarray]:
    time_steps = patch_grid[0] * patch_grid[1]
    tonic = np.empty((len(samples), time_steps, 3), dtype=np.float32)
    diff = np.empty_like(tonic)
    texture = np.empty_like(tonic)
    labels = np.empty(len(samples), dtype=np.int64)
    sample_ids: list[str] = []
    for index, sample in enumerate(samples):
        tonic[index], diff[index], texture[index] = load_image_features(
            sample.path, patch_grid
        )
        labels[index] = sample.label
        sample_ids.append(sample.sample_id)
    return {
        "tonic": tonic,
        "diff": diff,
        "texture": texture,
        "labels": labels,
        "sample_ids": np.asarray(sample_ids),
    }


def compute_thresholds(train_diff: np.ndarray, quantile: float) -> np.ndarray:
    if train_diff.ndim != 3 or train_diff.shape[2] != 3:
        raise ValueError(f"Expected train diff shape [N, T, 3], got {train_diff.shape}")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    return np.quantile(np.abs(train_diff[:, 1:, :]), quantile, axis=(0, 1)).astype(
        np.float32
    )


def make_rc_inputs(
    tonic: np.ndarray,
    diff: np.ndarray,
    thresholds: np.ndarray,
    event_amplitude: float = 1.0,
) -> np.ndarray:
    expected = tonic.shape
    if tonic.ndim != 3 or tonic.shape[2] != 3 or diff.shape != expected:
        raise ValueError("tonic and diff must both have shape [N, T, 3]")
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.shape != (3,):
        raise ValueError(f"Expected three RGB thresholds, got {thresholds.shape}")
    on = (diff > thresholds.reshape(1, 1, 3)).astype(np.float32)
    off = (diff < -thresholds.reshape(1, 1, 3)).astype(np.float32)
    on[:, 0] = 0.0
    off[:, 0] = 0.0
    return np.concatenate(
        [tonic, on * event_amplitude, off * event_amplitude], axis=2
    ).astype(np.float32, copy=False)


def compute_graded_event_scales(
    train_diff: np.ndarray,
    thresholds: np.ndarray,
    scale_quantile: float = 0.99,
) -> np.ndarray:
    """Return training-only ON/OFF excess scales with shape [2, 3]."""
    if train_diff.ndim != 3 or train_diff.shape[2] != 3:
        raise ValueError(f"Expected train diff shape [N, T, 3], got {train_diff.shape}")
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.shape != (3,):
        raise ValueError(f"Expected three RGB thresholds, got {thresholds.shape}")
    if not 0.0 < scale_quantile <= 1.0:
        raise ValueError("scale_quantile must be in (0, 1]")
    values = train_diff[:, 1:, :]
    excess = np.stack(
        [np.maximum(values - thresholds, 0.0), np.maximum(-values - thresholds, 0.0)]
    )
    scales = np.quantile(excess, scale_quantile, axis=(1, 2)).astype(np.float32)
    return np.maximum(scales, np.float32(EPSILON))


def make_graded_rc_inputs(
    tonic: np.ndarray,
    diff: np.ndarray,
    thresholds: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Encode signed RGB changes as continuous, mutually exclusive ON/OFF signals."""
    expected = tonic.shape
    if tonic.ndim != 3 or tonic.shape[2] != 3 or diff.shape != expected:
        raise ValueError("tonic and diff must both have shape [N, T, 3]")
    thresholds = np.asarray(thresholds, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    if thresholds.shape != (3,) or scales.shape != (2, 3):
        raise ValueError("thresholds must be [3] and scales must be [2, 3]")
    if np.any(scales <= 0):
        raise ValueError("graded event scales must be positive")
    on = np.clip(
        np.maximum(diff - thresholds.reshape(1, 1, 3), 0.0)
        / scales[0].reshape(1, 1, 3),
        0.0,
        1.0,
    )
    off = np.clip(
        np.maximum(-diff - thresholds.reshape(1, 1, 3), 0.0)
        / scales[1].reshape(1, 1, 3),
        0.0,
        1.0,
    )
    on[:, 0] = 0.0
    off[:, 0] = 0.0
    return np.concatenate([tonic, on, off], axis=2).astype(np.float32, copy=False)


def flip_scanned_features(
    values: np.ndarray,
    patch_grid: tuple[int, int],
    direction: str,
) -> np.ndarray:
    """Flip patch-local features spatially, then restore the fixed snake order."""
    if values.ndim != 3 or values.shape[1] != patch_grid[0] * patch_grid[1]:
        raise ValueError("values must have shape [N, rows*columns, C]")
    _, _, order = grid_geometry(patch_grid)
    grid = np.empty(
        (len(values), patch_grid[0], patch_grid[1], values.shape[2]),
        dtype=values.dtype,
    )
    for index, (row, column) in enumerate(order):
        grid[:, row, column] = values[:, index]
    if direction == "horizontal":
        grid = grid[:, :, ::-1]
    elif direction == "vertical":
        grid = grid[:, ::-1, :]
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    return np.stack([grid[:, row, column] for row, column in order], axis=1)


def quantile_tag(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):03d}"


def _save_base(output_dir: Path, split: str, arrays: dict[str, np.ndarray]) -> None:
    np.savez(
        output_dir / f"{split}_base.npz",
        tonic=arrays["tonic"],
        diff=arrays["diff"],
        texture=arrays["texture"],
        labels=arrays["labels"],
        sample_ids=arrays["sample_ids"],
    )
    np.save(output_dir / f"{split}_labels.npy", arrays["labels"])


def _event_statistics(
    inputs: np.ndarray, channel_names: tuple[str, str, str] = ("R", "G", "B")
) -> dict[str, object]:
    events = inputs[:, 1:, 3:]
    per_channel = events.mean(axis=(0, 1))
    return {
        "mean_events_per_sample": float(events.sum(axis=(1, 2)).mean()),
        "event_rate": float(events.mean()),
        "channel_event_rates": {
            name: float(value)
            for name, value in zip(
                tuple(f"{name}_ON" for name in channel_names)
                + tuple(f"{name}_OFF" for name in channel_names),
                per_channel,
            )
        },
    }


def preprocess_dataset(
    data_root: Path | str,
    output_dir: Path | str,
    *,
    settings: PreprocessSettings | None = None,
    max_per_class: int | None = None,
    class_prefix: str | None = None,
    selected_classes: Sequence[str] | None = None,
) -> dict[str, object]:
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir)
    settings = settings or PreprocessSettings()
    if max_per_class is not None and max_per_class < 2:
        raise ValueError("max_per_class must be at least 2")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_classes = discover_classes(data_root)
    if class_prefix is not None and selected_classes is not None:
        raise ValueError("Use either class_prefix or selected_classes, not both")
    if class_prefix is not None:
        classes = [name for name in all_classes if name.startswith(class_prefix)]
        if not classes:
            raise ValueError(f"No classes match prefix {class_prefix!r}")
    elif selected_classes is not None:
        requested = list(selected_classes)
        missing = sorted(set(requested) - set(all_classes))
        if missing:
            raise ValueError(f"Requested classes are missing from the dataset: {missing}")
        if len(requested) != len(set(requested)):
            raise ValueError("selected_classes contains duplicates")
        classes = sorted(requested)
    else:
        classes = all_classes
    development = build_samples(
        data_root / "train",
        classes,
        data_root=data_root,
        max_per_class=max_per_class,
    )
    test_samples = build_samples(
        data_root / "val",
        classes,
        data_root=data_root,
        max_per_class=max_per_class,
    )
    train_samples, val_samples = stratified_development_split(
        development, settings.validation_fraction, settings.split_seed
    )
    splits = {
        "train": extract_split(train_samples, settings.patch_grid),
        "val": extract_split(val_samples, settings.patch_grid),
        "test": extract_split(test_samples, settings.patch_grid),
    }
    for split, arrays in splits.items():
        _save_base(output_dir, split, arrays)

    texture_scales = np.quantile(
        splits["train"]["texture"], 0.99, axis=(0, 1)
    ).astype(np.float32)
    texture_scales = np.maximum(texture_scales, np.float32(EPSILON))
    for arrays in splits.values():
        arrays["texture_normalized"] = normalize_texture(
            arrays["texture"], texture_scales
        )
        arrays["texture_diff"] = batch_relative_difference(
            arrays["texture_normalized"]
        )

    thresholds_by_quantile: dict[str, list[float]] = {}
    texture_thresholds_by_quantile: dict[str, list[float]] = {}
    event_statistics: dict[str, dict[str, object]] = {}
    texture_event_statistics: dict[str, dict[str, object]] = {}
    for quantile in settings.threshold_quantiles:
        tag = quantile_tag(quantile)
        thresholds = compute_thresholds(splits["train"]["diff"], quantile)
        thresholds_by_quantile[tag] = thresholds.tolist()
        texture_thresholds = compute_thresholds(
            splits["train"]["texture_diff"], quantile
        )
        texture_thresholds_by_quantile[tag] = texture_thresholds.tolist()
        for split, arrays in splits.items():
            inputs = make_rc_inputs(
                arrays["tonic"], arrays["diff"], thresholds, settings.event_amplitude
            )
            np.save(output_dir / f"{split}_inputs_{tag}.npy", inputs)
            if quantile == settings.default_quantile:
                np.save(output_dir / f"{split}_inputs.npy", inputs)
            if split == "train":
                event_statistics[tag] = _event_statistics(inputs)
            texture_inputs = make_rc_inputs(
                arrays["texture_normalized"],
                arrays["texture_diff"],
                texture_thresholds,
                settings.event_amplitude,
            )
            np.save(output_dir / f"{split}_texture_inputs_{tag}.npy", texture_inputs)
            if quantile == settings.default_quantile:
                np.save(output_dir / f"{split}_texture_inputs.npy", texture_inputs)
            combined_inputs = np.concatenate(
                [inputs, arrays["texture_normalized"]], axis=2
            ).astype(np.float32, copy=False)
            np.save(output_dir / f"{split}_inputs12_{tag}.npy", combined_inputs)
            if quantile == settings.default_quantile:
                np.save(output_dir / f"{split}_inputs12.npy", combined_inputs)
            if split == "train":
                texture_event_statistics[tag] = _event_statistics(
                    texture_inputs, TEXTURE_FEATURES
                )

    y_edges, x_edges, scan_order = grid_geometry(settings.patch_grid)
    config: dict[str, object] = asdict(settings)
    config.update(
        {
            "y_edges": y_edges,
            "x_edges": x_edges,
            "scan_order": scan_order,
            "time_steps": settings.patch_grid[0] * settings.patch_grid[1],
            "input12_channels": (
                "R_tonic", "G_tonic", "B_tonic",
                "R_ON", "G_ON", "B_ON",
                "R_OFF", "G_OFF", "B_OFF",
                *TEXTURE_FEATURES,
            ),
            "data_root": str(data_root),
            "classes": classes,
            "class_to_index": {name: index for index, name in enumerate(classes)},
            "split_sizes": {name: int(len(value["labels"])) for name, value in splits.items()},
            "thresholds_by_quantile": thresholds_by_quantile,
            "texture_features": TEXTURE_FEATURES,
            "texture_scale_quantile": 0.99,
            "texture_scales": texture_scales.tolist(),
            "texture_thresholds_by_quantile": texture_thresholds_by_quantile,
            "event_statistics_train": event_statistics,
            "texture_event_statistics_train": texture_event_statistics,
            "max_per_class": max_per_class,
            "class_filter": {
                "prefix": class_prefix,
                "selected_classes": None
                if selected_classes is None
                else list(selected_classes),
            },
        }
    )
    with (output_dir / "preprocess_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return config
