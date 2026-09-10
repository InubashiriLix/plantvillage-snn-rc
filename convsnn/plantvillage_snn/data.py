from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import ExperimentConfig
from .utils import read_json, write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def discover_classes(data_root: str | Path) -> list[str]:
    train_root = Path(data_root) / "train"
    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"no class directories found under {train_root}")
    return classes


def class_mapping(data_root: str | Path) -> dict[str, int]:
    return {name: index for index, name in enumerate(discover_classes(data_root))}


def _images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_split_manifest(
    data_root: str | Path,
    output_path: str | Path,
    seed: int,
    selection_fraction: float,
) -> dict:
    """Create a stratified train/selection split using only ``train/``.

    The supplied ``val/`` paths never appear in this manifest; that directory is
    reserved for one-shot final evaluation.
    """
    root = Path(data_root).resolve()
    mapping = class_mapping(root)
    rng = np.random.default_rng(seed)
    train_records: list[dict] = []
    selection_records: list[dict] = []
    for name, original_index in mapping.items():
        paths = _images(root / "train" / name)
        if len(paths) < 2:
            raise RuntimeError(f"class {name!r} needs at least two training images")
        order = rng.permutation(len(paths))
        n_selection = min(max(1, round(len(paths) * selection_fraction)), len(paths) - 1)
        selection_indices = set(order[:n_selection].tolist())
        for i, path in enumerate(paths):
            record = {
                "path": path.relative_to(root).as_posix(),
                "class_name": name,
                "original_index": original_index,
            }
            (selection_records if i in selection_indices else train_records).append(record)
    manifest = {
        "version": 1,
        "data_root": str(root),
        "split_seed": seed,
        "selection_fraction": selection_fraction,
        "class_to_original_index": mapping,
        "partitions": {"train": train_records, "selection": selection_records},
        "final_test_directory": "val",
        "final_test_used_for_selection": False,
    }
    write_json(output_path, manifest)
    return manifest


def load_or_create_manifest(config: ExperimentConfig) -> tuple[dict, Path]:
    fraction = str(config.selection_fraction).replace(".", "p")
    path = (
        Path(config.split_manifest_path)
        if config.split_manifest_path
        else Path(config.output_dir) / "splits" / f"seed{config.split_seed}_val{fraction}.json"
    )
    if path.is_file():
        manifest = read_json(path)
        expected_root = str(Path(config.data_root).resolve())
        if manifest["data_root"] != expected_root:
            raise ValueError("existing split manifest belongs to a different data root")
        if manifest["split_seed"] != config.split_seed or not np.isclose(
            manifest["selection_fraction"], config.selection_fraction
        ):
            raise ValueError("existing split manifest does not match split configuration")
        return manifest, path
    return (
        create_split_manifest(
            config.data_root, path, config.split_seed, config.selection_fraction
        ),
        path,
    )


def load_stage_mapping(config: ExperimentConfig, original_mapping: dict[str, int]) -> dict[str, int]:
    if config.stage == "all38":
        if len(original_mapping) != 38:
            raise ValueError(f"all38 requires exactly 38 classes, found {len(original_mapping)}")
        return dict(original_mapping)
    artifact = read_json(config.top10_path)
    if artifact.get("version") != 1 or artifact.get("source_stage") != "all38":
        raise ValueError("unsupported or invalid top10_classes.json")
    selected = artifact.get("classes", [])
    if len(selected) != 10:
        raise ValueError("top10_classes.json must contain exactly 10 classes")
    names: list[str] = []
    for expected_new_index, item in enumerate(selected):
        name = item["name"]
        if name in names:
            raise ValueError(f"duplicate top-10 class: {name}")
        if name not in original_mapping or item["original_index"] != original_mapping[name]:
            raise ValueError(f"top-10 original index mismatch for {name}")
        if item.get("new_index") != expected_new_index:
            raise ValueError("top-10 new indices must be contiguous and ordered 0..9")
        names.append(name)
    return {name: i for i, name in enumerate(names)}


class PlantVillageRecords(Dataset):
    def __init__(
        self,
        root: Path,
        records: list[dict],
        stage_mapping: dict[str, int],
        transform: Callable,
    ):
        self.root = root
        self.records = [r for r in records if r["class_name"] in stage_mapping]
        self.stage_mapping = stage_mapping
        self.transform = transform
        self.targets = [stage_mapping[r["class_name"]] for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, self.targets[index]


def _final_records(root: Path, original_mapping: dict[str, int]) -> list[dict]:
    records = []
    final_root = root / "val"
    found_names = sorted(p.name for p in final_root.iterdir() if p.is_dir())
    if found_names != sorted(original_mapping):
        raise ValueError("final-test classes differ from training classes")
    for name, index in original_mapping.items():
        for path in _images(final_root / name):
            records.append({
                "path": path.relative_to(root).as_posix(),
                "class_name": name,
                "original_index": index,
            })
    return records


def _transform(image_size: int, augmentation: str, training: bool,
               input_encoding: str = "direct_current"):
    operations: list = [transforms.Resize((image_size, image_size))]
    if training and augmentation in {"basic", "strong"}:
        operations.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12 if augmentation == "basic" else 25),
        ])
        if augmentation == "strong":
            operations.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
    elif augmentation not in {"none", "basic", "strong"}:
        raise ValueError("augmentation must be none, basic, or strong")
    operations.append(transforms.ToTensor())
    if input_encoding != "direct_current":
        raise ValueError("unsupported input encoding")
    return transforms.Compose(operations)


def _limit_records(records: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or len(records) <= limit:
        return records
    if limit < 1:
        raise ValueError("sample limits must be positive")
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict]] = {}
    for record in records:
        by_class.setdefault(record["class_name"], []).append(record)
    chosen: list[dict] = []
    # Round-robin sampling preserves all classes whenever the limit permits it.
    shuffled = {k: [v[i] for i in rng.permutation(len(v))] for k, v in by_class.items()}
    keys = sorted(shuffled)
    cursor = 0
    while len(chosen) < limit and keys:
        key = keys[cursor % len(keys)]
        if shuffled[key]:
            chosen.append(shuffled[key].pop())
        keys = [k for k in keys if shuffled[k]]
        cursor += 1
    return chosen


@dataclass
class DataBundle:
    train: DataLoader
    selection: DataLoader
    final_test: DataLoader | None
    class_to_index: dict[str, int]
    class_to_original_index: dict[str, int]
    split_manifest_path: str
    sample_counts: dict[str, list[int]]


def build_dataloaders(config: ExperimentConfig) -> DataBundle:
    manifest, manifest_path = load_or_create_manifest(config)
    root = Path(config.data_root).resolve()
    original_mapping = manifest["class_to_original_index"]
    stage_mapping = load_stage_mapping(config, original_mapping)
    train_records = [r for r in manifest["partitions"]["train"] if r["class_name"] in stage_mapping]
    selection_records = [r for r in manifest["partitions"]["selection"] if r["class_name"] in stage_mapping]
    train_records = _limit_records(train_records, config.max_train_samples, config.seed)
    selection_records = _limit_records(selection_records, config.max_selection_samples, config.seed + 1)
    train_ds = PlantVillageRecords(
        root, train_records, stage_mapping,
        _transform(config.image_size, config.augmentation, True, config.input_encoding),
    )
    selection_ds = PlantVillageRecords(
        root, selection_records, stage_mapping,
        _transform(config.image_size, "none", False, config.input_encoding),
    )

    generator = torch.Generator().manual_seed(config.seed)
    sampler = None
    shuffle = True
    if config.class_balance == "weighted_sampler":
        counts = np.bincount(train_ds.targets, minlength=len(stage_mapping))
        sample_weights = [1.0 / max(counts[y], 1) for y in train_ds.targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), generator=generator)
        shuffle = False
    elif config.class_balance not in {"none", "class_weights"}:
        raise ValueError("class_balance must be none, class_weights, or weighted_sampler")

    common = dict(batch_size=config.batch_size, num_workers=config.num_workers)
    train_loader = DataLoader(train_ds, shuffle=shuffle, sampler=sampler, generator=generator, **common)
    selection_loader = DataLoader(selection_ds, shuffle=False, **common)
    final_loader = None
    final_targets: list[int] = []
    if config.evaluate_final:
        final_records = [r for r in _final_records(root, original_mapping) if r["class_name"] in stage_mapping]
        final_records = _limit_records(final_records, config.max_final_samples, config.seed + 2)
        final_ds = PlantVillageRecords(
            root, final_records, stage_mapping,
            _transform(config.image_size, "none", False, config.input_encoding),
        )
        final_targets = final_ds.targets
        final_loader = DataLoader(final_ds, shuffle=False, **common)

    def counts(targets: list[int]) -> list[int]:
        return np.bincount(targets, minlength=len(stage_mapping)).astype(int).tolist()

    return DataBundle(
        train=train_loader,
        selection=selection_loader,
        final_test=final_loader,
        class_to_index=stage_mapping,
        class_to_original_index=original_mapping,
        split_manifest_path=str(manifest_path.resolve()),
        sample_counts={
            "train": counts(train_ds.targets),
            "selection": counts(selection_ds.targets),
            "final_test": counts(final_targets),
        },
    )


def class_weights(sample_counts: list[int], device: torch.device) -> torch.Tensor:
    counts = torch.tensor(sample_counts, dtype=torch.float32, device=device)
    weights = counts.sum() / (counts.clamp_min(1) * len(counts))
    return weights
