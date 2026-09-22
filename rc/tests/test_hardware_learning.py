import csv
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from plantvillage_rc.hardware_data import CHANNELS, METADATA, REPEAT_SEEDS, load_dataset, read_responses
from plantvillage_rc.hardware_learning import evaluate, fit_final, select_readout, summarize
from plantvillage_rc.hardware_models import Candidate, HardwareFeatures, build_model, freeze_candidates, probabilities


@pytest.fixture
def measured_fixture(tmp_path):
    rng = np.random.default_rng(9)
    n = 20
    x = rng.normal(0, 1e-7, (n, 9, 12, 65))
    x[10:] += 2e-6
    response = tmp_path / "responses.csv"
    feature_names = [f"{channel}_row{row:02d}_{'step'+str(step).zfill(2) if step < 65 else 'tail'}_A"
                     for channel in CHANNELS for row in range(1, 13) for step in range(1, 66)]
    metadata = [{"subset_sample_index": str(i), "label_index": str(i // 10), "class_name": f"class{i // 10}",
                 "source_sample_index": str(i + 70), "sample_id": f"train/class{i // 10}/{i}.png"} for i in range(n)]
    with response.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*METADATA, *feature_names])
        for meta, values in zip(metadata, x):
            writer.writerow([*[meta[key] for key in METADATA], *values.flatten()])
    inputs = tmp_path / "inputs.csv"
    with inputs.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*METADATA, "step", "patch_row", "patch_col", *CHANNELS])
        writer.writeheader()
        for meta in metadata:
            for step in range(64):
                row, col = divmod(step, 8)
                writer.writerow({**meta, "step": step + 1, "patch_row": row, "patch_col": col if row % 2 == 0 else 7 - col,
                                 **dict(zip(CHANNELS, [0.2, 0.3, 0.4, 0, 0, 0, 0, 0, 0]))})
    headers = ["sample_index", "真实标签", "类别名", "划分种子", "外层折", "预测标签", "正确", "sample_id"]
    rows = [headers]
    for seed in REPEAT_SEEDS:
        for i, meta in enumerate(metadata):
            rows.append([str(i), meta["label_index"], meta["class_name"], str(seed), str(i % 5 + 1), meta["label_index"], "1", meta["sample_id"]])
    from xml.sax.saxutils import escape
    content = "".join('<row>' + "".join(f'<c r="{chr(65+j)}{i+1}" t="inlineStr"><is><t>{escape(value)}</t></is></c>' for j, value in enumerate(row)) + '</row>' for i, row in enumerate(rows))
    book = tmp_path / "history.xlsx"
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(book, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="推荐模型预测" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData>{content}</sheetData></worksheet>')
    return dict(responses=response, inputs=inputs, workbook=book), x


def test_header_order_and_no_metadata_features(measured_fixture, tmp_path):
    paths, expected = measured_fixture
    x, meta = read_responses(paths["responses"])
    np.testing.assert_array_equal(x, expected)
    with paths["responses"].open() as f:
        reader = csv.DictReader(f)
        header, rows = list(reversed(reader.fieldnames)), list(reader)
    shuffled = tmp_path / "shuffled.csv"
    with shuffled.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in reversed(rows):
            row["source_sample_index"] = "999999"
            writer.writerow(row)
    np.testing.assert_array_equal(read_responses(shuffled)[0], expected)
    assert len(meta) == 20


@pytest.mark.parametrize("fault", ["missing_step", "sample_mismatch", "nan_current"])
def test_bad_source_data_is_rejected(measured_fixture, fault):
    paths, _ = measured_fixture
    target = paths["responses"] if fault == "nan_current" else paths["inputs"]
    lines = target.read_text().splitlines()
    if fault == "missing_step":
        lines.pop()
    elif fault == "sample_mismatch":
        lines[1] = lines[1].replace("train/class0/0.png", "wrong.png")
    else:
        values = lines[1].split(",")
        values[5] = "nan"
        lines[1] = ",".join(values)
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        load_dataset(**paths, expected_samples=20, expected_classes=2)


def test_tail_not_in_dynamics_and_spatial_snake():
    x = np.zeros((1, 9, 12, 65))
    x[..., :64] = np.arange(64)[None, None, None, :] * 1e-6
    x[..., 64] = 999e-6
    transformed = HardwareFeatures("compensated_100").fit_transform(x)
    np.testing.assert_allclose(transformed[0, :65], [0, *([1] * 63), 999], atol=1e-12)
    assert transformed.shape == (1, 1170)
    spatial = HardwareFeatures("spatial").fit_transform(x)
    assert spatial.shape == (1, 1530)
    # 2x2 and 4x4 pooled planes precede the full-resolution planes.
    full = spatial[0, 18 * (4 + 16):18 * (4 + 16 + 64)].reshape(18, 8, 8)
    np.testing.assert_allclose(full[0, 1], np.arange(15, 7, -1))
    np.testing.assert_allclose(spatial[0, -18:-9], 999)


def test_scaler_and_pca_fit_training_rows_only():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(24, 9, 12, 65)) * 1e-6
    candidate = Candidate("test", "SVM_RBF", "median", {"C": 1, "gamma": "scale"}, 4)
    with threadpool_limits(1):
        model = build_model(candidate).fit(x[:20], np.array([0, 1] * 10))
        mean = model.named_steps["scale"].mean_.copy()
        components = model.named_steps["pca"].components_.copy()
        probabilities(model, x[20:] + 100, 2)
    np.testing.assert_array_equal(mean, model.named_steps["scale"].mean_)
    np.testing.assert_array_equal(components, model.named_steps["pca"].components_)
    np.testing.assert_allclose(mean, HardwareFeatures("median").transform(x[:20]).mean(axis=0))


def test_candidates_frozen_and_ensemble_ties_prefer_single():
    first, second = freeze_candidates(), freeze_candidates()
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]
    assert len(first) == 36 and len({c.id for c in first}) == 36
    y = np.array([0, 1])
    results = [{"id": family, "family": family, "probabilities": np.eye(2),
                "scores": {"accuracy": 1.0, "macro_f1": 1.0}, "feature_count": 10}
               for family in ("RandomForest", "MLP")]
    selected, _ = select_readout(results, y)
    assert len(selected["members"]) == 1


def test_nested_pipeline_and_roundtrip(measured_fixture, tmp_path):
    paths, _ = measured_fixture
    dataset = load_dataset(**paths, expected_samples=20, expected_classes=2)
    assert dataset.x.shape == (20, 9, 12, 65)
    candidates = [Candidate("tiny_ridge", "Ridge", "mean_std", {"alpha": 1.0}),
                  Candidate("tiny_tree", "ExtraTrees", "median", {"n_estimators": 3, "max_features": "sqrt", "min_samples_leaf": 1})]
    output = tmp_path / "experiment"
    summary = evaluate(dataset, output, candidates=candidates, workers=1, hours=0.1)
    assert summary["prediction_count"] == 60 and summary["sample_count"] == 20
    assert len(summary["repeats"]) == 3
    assert summary["mean_accuracy"] > 0.9
    with pytest.raises(ValueError, match="incomplete"):
        summarize(dataset, [])
    # Cached replay is identical and can run even with no time for new fits.
    replay = evaluate(dataset, output, candidates=candidates, workers=1, hours=1e-10)
    assert replay == summary
    fit_final(dataset, output, candidates=candidates, workers=1, hours=0.1)
    model = joblib.load(output / "model.joblib")
    with threadpool_limits(1):
        probabilities_ = model.predict_proba(dataset.x)
    np.testing.assert_allclose(probabilities_.sum(axis=1), 1)
    assert np.isfinite(probabilities_).all()
