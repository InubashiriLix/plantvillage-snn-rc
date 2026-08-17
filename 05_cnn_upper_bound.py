#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ManifestDataset(Dataset):
    def __init__(self, data_root: Path, archive: Path, transform) -> None:
        with np.load(archive) as arrays:
            self.sample_ids = arrays["sample_ids"].tolist()
            self.labels = arrays["labels"].astype(np.int64)
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        with Image.open(self.data_root / self.sample_ids[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.labels[index])


@torch.inference_mode()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    for images, targets in loader:
        outputs = model(images.to(device))
        labels.extend(targets.tolist())
        predictions.extend(outputs.argmax(dim=1).cpu().tolist())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ImageNet MobileNetV3-Small upper-bound baseline; not part of RC"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cnn_upper_bound"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--unfreeze-last", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    config = json.loads((args.input_dir / "preprocess_config.json").read_text())
    data_root = Path(config["data_root"])
    class_count = len(config["classes"])
    weights = MobileNet_V3_Small_Weights.DEFAULT
    preset = weights.transforms()
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=preset.mean, std=preset.std),
        ]
    )
    eval_transform = preset
    datasets = {
        split: ManifestDataset(
            data_root,
            args.input_dir / f"{split}_base.npz",
            train_transform if split == "train" else eval_transform,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }

    model = mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if args.unfreeze_last:
        for parameter in model.features[-2:].parameters():
            parameter.requires_grad = True
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    counts = np.bincount(datasets["train"].labels, minlength=class_count)
    class_weights = counts.sum() / (class_count * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    history = []
    best_macro = -1.0
    checkpoint = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.features.eval()
        if args.unfreeze_last:
            model.features[-2:].train()
        for images, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        validation = evaluate(model, loaders["val"], device)
        validation["epoch"] = epoch
        history.append(validation)
        if validation["macro_recall"] > best_macro:
            best_macro = validation["macro_recall"]
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    result: dict[str, object] = {
        "model": "mobilenet_v3_small_imagenet",
        "purpose": "upper_bound_only_not_rc",
        "device": str(device),
        "history": history,
        "best_validation": evaluate(model, loaders["val"], device),
        "target_macro_recall": args.target,
        "test_evaluated": False,
    }
    if result["best_validation"]["macro_recall"] >= args.target:
        result["test"] = evaluate(model, loaders["test"], device)
        result["test_evaluated"] = True
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
