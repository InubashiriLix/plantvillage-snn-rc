"""Strict readers for measured current traces and the historical OOF workbook."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

CHANNELS = ("R", "G", "B", "ON-red", "ON-green", "ON-blue", "OFF-red", "OFF-green", "OFF-blue")
METADATA = ("subset_sample_index", "label_index", "class_name", "source_sample_index", "sample_id")
REPEAT_SEEDS = (20260909, 20260910, 20260911)
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scores(y, predictions):
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, predictions, average="macro", zero_division=0)),
        "correct": int(np.sum(y == predictions)),
        "sample_count": int(len(y)),
    }


def read_responses(path, require_labels=True):
    """Read by header coordinates, not CSV column order; values remain amperes."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(set(header)) != len(header):
            raise ValueError("duplicate hardware CSV columns")
        required = set(METADATA if require_labels else ("subset_sample_index", "sample_id"))
        if not required.issubset(header):
            raise ValueError(f"missing hardware metadata: {sorted(required - set(header))}")
        coordinates = {}
        for name in header:
            if name in METADATA:
                continue
            match = re.fullmatch(r"(.+)_row(\d{2})_(?:step(\d{2})|(tail))_A", name)
            if not match or match[1] not in CHANNELS:
                raise ValueError(f"unknown hardware column: {name}")
            channel, row = CHANNELS.index(match[1]), int(match[2]) - 1
            step = int(match[3]) - 1 if match[3] else 64
            if not 0 <= row < 12 or not 0 <= step <= 64 or (match[3] and step == 64):
                raise ValueError(f"invalid row/step: {name}")
            key = (channel, row, step)
            if key in coordinates:
                raise ValueError("duplicate hardware coordinates")
            coordinates[key] = name
        if len(coordinates) != 7020:
            raise ValueError(f"expected 7020 response columns, got {len(coordinates)}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("empty CSV or inconsistent row width")
    rows.sort(key=lambda row: int(row["subset_sample_index"]))
    if len({row["subset_sample_index"] for row in rows}) != len(rows) or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample identifiers")
    order = [coordinates[(c, r, t)] for c in range(9) for r in range(12) for t in range(65)]
    x = np.asarray([[float(row[name]) for name in order] for row in rows], dtype=np.float64).reshape(-1, 9, 12, 65)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite current measurement")
    metadata = [{key: row[key] for key in METADATA if key in row} for row in rows]
    return x, metadata


def read_workbook(path):
    """Extract values only; never executes workbook formulas or embedded content."""
    sheets = {}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(node.itertext()) for node in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        relations = {node.attrib["Id"]: node.attrib["Target"] for node in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in workbook.find("s:sheets", NS):
            target = relations[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
            target = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            matrix = []
            for row in ET.fromstring(archive.read(target)).findall("s:sheetData/s:row", NS):
                values = {}
                for cell in row:
                    column = re.match(r"[A-Z]+", cell.attrib["r"])[0]
                    index = 0
                    for character in column:
                        index = index * 26 + ord(character) - ord("A") + 1
                    value = cell.find("s:v", NS)
                    text = value.text if value is not None else ""
                    if cell.attrib.get("t") == "s":
                        text = strings[int(text)]
                    elif cell.attrib.get("t") == "inlineStr":
                        text = "".join(cell.find("s:is", NS).itertext())
                    values[index - 1] = text
                matrix.append([values.get(i, "") for i in range(max(values, default=-1) + 1)])
            sheets[sheet.attrib["name"]] = matrix
    return sheets


@dataclass
class HardwareDataset:
    x: np.ndarray
    y: np.ndarray
    metadata: list[dict]
    classes: list[str]
    folds: dict[int, np.ndarray]
    historical_predictions: dict[int, np.ndarray]
    hashes: dict[str, str]
    audit: dict


def load_dataset(responses, inputs, workbook, *, expected_samples=300, expected_classes=10):
    x, metadata = read_responses(responses)
    y = np.asarray([int(row["label_index"]) for row in metadata], dtype=np.int64)
    if len(x) != expected_samples or not np.array_equal(np.unique(y), np.arange(expected_classes)):
        raise ValueError("unexpected sample count or noncontiguous class labels")
    counts = np.bincount(y)
    if not np.all(counts == expected_samples // expected_classes):
        raise ValueError("expected equal per-class sample counts")
    classes = []
    for label in range(expected_classes):
        names = {row["class_name"] for row in metadata if int(row["label_index"]) == label}
        if len(names) != 1:
            raise ValueError("inconsistent label-to-class mapping")
        classes.append(names.pop())
    if len(set(classes)) != len(classes):
        raise ValueError("class names must be unique")
    lookup = {int(row["subset_sample_index"]): i for i, row in enumerate(metadata)}
    seen_steps = [set() for _ in metadata]
    with Path(inputs).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                i = lookup[int(row["subset_sample_index"])]
            except KeyError as error:
                raise ValueError("input has an unknown sample") from error
            if any(row[key] != metadata[i][key] for key in METADATA):
                raise ValueError("hardware/input sample or label mismatch")
            step = int(row["step"]) - 1
            if not 0 <= step < 64 or step in seen_steps[i]:
                raise ValueError("invalid or duplicate input step")
            seen_steps[i].add(step)
            grid_row, within = divmod(step, 8)
            grid_col = within if grid_row % 2 == 0 else 7 - within
            if (int(row["patch_row"]), int(row["patch_col"])) != (grid_row, grid_col):
                raise ValueError("input scan coordinates disagree with snake order")
            values = np.asarray([float(row[channel]) for channel in CHANNELS])
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("invalid normalized optical input")
            if (step == 0 and values[3:].any()) or np.any((values[3:6] > 0) & (values[6:] > 0)):
                raise ValueError("invalid ON/OFF event input")
    if any(len(steps) != 64 for steps in seen_steps):
        raise ValueError("input is missing one or more scan steps")

    sheets = read_workbook(workbook)
    raw = sheets["推荐模型预测"]
    predictions = [dict(zip(raw[0], row)) for row in raw[1:] if row]
    folds = {seed: np.full(len(x), -1, dtype=int) for seed in REPEAT_SEEDS}
    historical = {seed: np.full(len(x), -1, dtype=int) for seed in REPEAT_SEEDS}
    for row in predictions:
        seed, i = int(row["划分种子"]), lookup[int(row["sample_index"])]
        if seed not in folds or folds[seed][i] != -1:
            raise ValueError("unknown repeat or duplicate historical prediction")
        if (int(row["真实标签"]) != y[i] or row["类别名"] != classes[y[i]] or row["sample_id"] != metadata[i]["sample_id"]):
            raise ValueError("workbook/hardware alignment mismatch")
        fold, predicted = int(row["外层折"]) - 1, int(row["预测标签"])
        if not 0 <= fold < 5 or not 0 <= predicted < expected_classes:
            raise ValueError("invalid historical fold or prediction")
        if int(row["正确"]) != int(predicted == y[i]):
            raise ValueError("historical correctness flag is inconsistent")
        folds[seed][i], historical[seed][i] = fold, predicted
    for seed in REPEAT_SEEDS:
        if np.any(folds[seed] < 0):
            raise ValueError("incomplete historical OOF predictions")
        for fold in range(5):
            if not np.all(np.bincount(y[folds[seed] == fold], minlength=expected_classes) == counts // 5):
                raise ValueError("historical fold is not class-balanced")
    historical_scores = [{"seed": seed, **scores(y, historical[seed])} for seed in REPEAT_SEEDS]
    flat = x.reshape(len(x), -1)
    if np.unique(flat, axis=0).shape[0] != len(x):
        raise ValueError("duplicate response samples require grouped validation")
    correlations = {}
    for index, channel in enumerate(CHANNELS):
        values = x[:, index, :, :64].transpose(1, 0, 2).reshape(12, -1)
        correlation = np.corrcoef(values)
        correlations[channel] = float(correlation[np.triu_indices(12, 1)].mean())
    hashes = {"responses": sha256(responses), "inputs": sha256(inputs), "workbook": sha256(workbook)}
    audit = {
        "sample_count": len(x), "response_shape": list(x.shape), "channels": list(CHANNELS),
        "classes": classes, "class_counts": counts.tolist(), "source_sha256": hashes,
        "range_amperes": [float(x.min()), float(x.max())],
        "constant_columns": int(np.sum(flat.std(axis=0) == 0)),
        "mean_inter_row_correlations": correlations,
        "historical_scores": historical_scores,
        "historical_mean_accuracy": float(np.mean([row["accuracy"] for row in historical_scores])),
        "historical_always_wrong_indices": [int(metadata[i]["subset_sample_index"]) for i in range(len(y)) if all(historical[seed][i] != y[i] for seed in REPEAT_SEEDS)],
        "independent_test_available": False,
        "baseline_training_code_available": False,
    }
    return HardwareDataset(x, y, metadata, classes, folds, historical, hashes, audit)
