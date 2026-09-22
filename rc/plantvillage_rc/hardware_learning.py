"""Resumable nested evaluation; all model and ensemble selection is inner-only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import warnings

import joblib
from joblib import Parallel, delayed, parallel_config
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .hardware_data import CHANNELS, REPEAT_SEEDS, scores, sha256, write_csv, write_json
from .hardware_models import Candidate, FittedReadout, TREE_FAMILIES, build_model, freeze_candidates, probabilities


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def audit_sources(dataset, paths, output):
    output = Path(output)
    source_dir = output / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, original in paths.items():
        destination = source_dir / (name + Path(original).suffix)
        if destination.exists():
            if sha256(destination) != dataset.hashes[name]:
                raise ValueError("saved input differs; use a new output directory")
        else:
            shutil.copy2(original, destination)
    write_json(output / "audit.json", dataset.audit)
    # Local fold/metadata files are never part of the public result export.
    write_csv(output / "sample_mapping.csv", dataset.metadata)
    rows = []
    for seed in REPEAT_SEEDS:
        for i, meta in enumerate(dataset.metadata):
            rows.append({"sample_index": meta["subset_sample_index"], "seed": seed,
                         "fold": int(dataset.folds[seed][i]) + 1, "label": int(dataset.y[i]),
                         "historical_prediction": int(dataset.historical_predictions[seed][i])})
    write_csv(output / "historical_folds.csv", rows)
    return dataset.audit


def freeze_run(dataset, candidates, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    module_dir = Path(__file__).resolve().parent
    definition = {
        "format_version": 1, "data_hashes": dataset.hashes,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "repeat_seeds": list(REPEAT_SEEDS), "inner_folds": 3,
        "probability_mlp_weights": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "code_hashes": {name: sha256(module_dir / name) for name in ("hardware_data.py", "hardware_models.py", "hardware_learning.py")},
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(),
    }
    definition = json.loads(json.dumps(definition))
    path = output / "frozen_run.json"
    if path.exists() and json.loads(path.read_text()) != definition:
        raise ValueError("data/code/candidate version changed; use a new output directory to avoid stale cache")
    write_json(path, definition)
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()


def _candidate_inner(candidate, x, y, split_seed, cache_dir, deadline, fingerprint):
    """Receives only outer-training rows, never the outer holdout or its labels."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{candidate.id}.joblib"
    cache_key = hashlib.sha256((fingerprint + str(split_seed) + json.dumps(candidate.to_dict(), sort_keys=True)).encode() + np.asarray(x).tobytes() + np.asarray(y).tobytes()).hexdigest()
    if cache_path.exists():
        result = joblib.load(cache_path)
        if result["cache_key"] != cache_key:
            raise ValueError("inner CV cache mismatch")
        return result
    started = time.monotonic()
    out = np.full((len(y), len(np.unique(y))), np.nan)
    counts = np.zeros(len(y), dtype=int)
    captured_warnings = set()
    splits = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
    for train, val in splits.split(x, y):
        if time.time() >= deadline:
            raise TimeoutError("evaluation budget reached; completed candidate caches are retained")
        model = build_model(candidate)
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(x[train], y[train])
            out[val] = probabilities(model, x[val], out.shape[1])
        captured_warnings.update(str(item.message) for item in captured)
        counts[val] += 1
    if not np.all(counts == 1) or not np.isfinite(out).all():
        raise ValueError("incomplete inner OOF probabilities")
    result = {
        "id": candidate.id, "family": candidate.family, "candidate": candidate.to_dict(),
        "scores": scores(y, out.argmax(axis=1)), "probabilities": out,
        "feature_count": int(build_model(candidate).steps[0][1].transform(x[:1]).shape[1]),
        "seconds": time.monotonic() - started, "warnings": sorted(captured_warnings),
        "cache_key": cache_key,
    }
    temporary = cache_path.with_suffix(".tmp")
    joblib.dump(result, temporary)
    temporary.replace(cache_path)
    emit("candidate_complete", candidate=candidate.id, cache=cache_dir.name,
         accuracy=result["scores"]["accuracy"], seconds=round(result["seconds"], 2))
    return result


def candidate_rank(result):
    return (-result["scores"]["accuracy"], -result["scores"]["macro_f1"], result["feature_count"], result["id"])


def select_readout(results, y):
    best = min(results, key=candidate_rank)
    options = [{"members": [best["id"]], "weights": [1.0], "scores": best["scores"], "kind": "single"}]
    trees = [result for result in results if result["family"] in TREE_FAMILIES]
    mlps = [result for result in results if result["family"] == "MLP"]
    if trees and mlps:
        tree, mlp = min(trees, key=candidate_rank), min(mlps, key=candidate_rank)
        for weight in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            probability = (1 - weight) * tree["probabilities"] + weight * mlp["probabilities"]
            members, weights = ([tree["id"], mlp["id"]], [1 - weight, weight])
            active = [(member, w) for member, w in zip(members, weights) if w > 0]
            options.append({"members": [v[0] for v in active], "weights": [v[1] for v in active],
                            "scores": scores(y, probability.argmax(axis=1)), "kind": "ensemble" if len(active) == 2 else "single"})
    # Exactly tied performance retains the best single model rather than adding an ensemble.
    selected = min(options, key=lambda value: (-value["scores"]["accuracy"], -value["scores"]["macro_f1"], len(value["members"])))
    return selected, options


def run_inner(candidates, x, y, split_seed, cache_dir, deadline, fingerprint, workers):
    with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=1):
        return Parallel(pre_dispatch=workers)(delayed(_candidate_inner)(candidate, x, y, split_seed, cache_dir, deadline, fingerprint) for candidate in candidates)


def fit_selected(selection, candidates, x, y, classes):
    lookup = {candidate.id: candidate for candidate in candidates}
    models = []
    with threadpool_limits(limits=1):
        for name in selection["members"]:
            model = build_model(lookup[name])
            model.fit(x, y)
            models.append(model)
    return FittedReadout(models, selection["weights"], classes)


def unit_diagnostic(dataset, output):
    """Mechanism check, separate from nested model selection and final claims."""
    candidates = [
        Candidate("unit_rf", "RandomForest", "mean_std", {"n_estimators": 400, "min_samples_leaf": 1, "max_features": "sqrt"}),
        Candidate("unit_et", "ExtraTrees", "mean_std", {"n_estimators": 400, "min_samples_leaf": 1, "max_features": "sqrt"}),
    ]
    rows = []
    seed = REPEAT_SEEDS[0]
    for candidate in candidates:
        for unit, scale in (("A", 1.0), ("uA", 1e6)):
            prediction = np.full(len(dataset.y), -1)
            nodes = []
            with threadpool_limits(limits=1):
                for fold in range(5):
                    train, val = dataset.folds[seed] != fold, dataset.folds[seed] == fold
                    model = build_model(candidate, unit_scale=scale)
                    model.fit(dataset.x[train], dataset.y[train])
                    prediction[val] = probabilities(model, dataset.x[val], len(dataset.classes)).argmax(axis=1)
                    nodes.extend(estimator.tree_.node_count for estimator in model.named_steps["model"].estimators_)
            rows.append({"family": candidate.family, "unit": unit, "repeat_seed": seed,
                         "mean_tree_nodes": float(np.mean(nodes)), **scores(dataset.y, prediction)})
            emit("unit_diagnostic", **rows[-1])
    write_json(Path(output) / "unit_diagnostic.json", rows)
    return rows


def evaluate(dataset, output, *, workers=6, hours=7.0, candidates=None):
    output = Path(output)
    candidates = candidates or freeze_candidates()
    fingerprint = freeze_run(dataset, candidates, output)
    deadline = time.time() + hours * 3600
    started = time.monotonic()
    fold_summaries = []
    try:
        for seed in REPEAT_SEEDS:
            for fold in range(5):
                directory = output / "evaluation" / f"seed{seed}_fold{fold + 1}"
                directory.mkdir(parents=True, exist_ok=True)
                completed = directory / "completed.joblib"
                if completed.exists():
                    result = joblib.load(completed)
                    if result["fingerprint"] != fingerprint:
                        raise ValueError("completed outer fold belongs to another run")
                    fold_summaries.append(result)
                    emit("outer_fold_cached", seed=seed, fold=fold + 1)
                    continue
                if time.time() >= deadline:
                    raise TimeoutError("evaluation time budget reached")
                train = np.flatnonzero(dataset.folds[seed] != fold)
                val = np.flatnonzero(dataset.folds[seed] == fold)
                emit("outer_fold_start", seed=seed, fold=fold + 1, candidates=len(candidates))
                inner = run_inner(candidates, dataset.x[train], dataset.y[train], seed + fold,
                                  directory / "inner", deadline, fingerprint, workers)
                selected, options = select_readout(inner, dataset.y[train])
                # Selection is frozen before the outer labels are consulted.
                write_json(directory / "selection.json", {"selected": selected, "options": options})
                readout = fit_selected(selected, candidates, dataset.x[train], dataset.y[train], dataset.classes)
                with threadpool_limits(limits=1):
                    probability = readout.predict_proba(dataset.x[val])
                predictions = probability.argmax(axis=1)
                result = {"seed": seed, "fold": fold + 1, "indices": val, "probabilities": probability,
                          "selected": selected, "scores": scores(dataset.y[val], predictions), "fingerprint": fingerprint}
                temporary = completed.with_suffix(".tmp")
                joblib.dump(result, temporary)
                temporary.replace(completed)
                write_csv(directory / "inner_scores.csv", [
                    {"candidate": value["id"], "family": value["family"], "feature_count": value["feature_count"],
                     "seconds": value["seconds"], "warnings": " | ".join(value["warnings"]), **value["scores"]}
                    for value in sorted(inner, key=candidate_rank)])
                fold_summaries.append(result)
                write_json(output / "progress.json", {"status": "running", "completed_outer_folds": len(fold_summaries),
                                                       "required_outer_folds": 15, "invocation_seconds": time.monotonic() - started})
                emit("outer_fold_complete", seed=seed, fold=fold + 1, selected=selected["members"], **result["scores"])
    except TimeoutError:
        write_json(output / "progress.json", {"status": "partial", "completed_outer_folds": len(fold_summaries),
                                               "required_outer_folds": 15, "invocation_seconds": time.monotonic() - started})
        raise
    summary = summarize(dataset, fold_summaries)
    write_json(output / "evaluation_summary.json", summary)
    write_json(output / "progress.json", {"status": "complete", "completed_outer_folds": 15,
                                           "required_outer_folds": 15, "invocation_seconds": time.monotonic() - started})
    emit("evaluation_complete", mean_accuracy=summary["mean_accuracy"], historical_mean_accuracy=summary["historical_mean_accuracy"])
    return summary


def summarize(dataset, results):
    if len(results) != 15 or len({(row["seed"], row["fold"]) for row in results}) != 15:
        raise ValueError("cannot summarize incomplete outer CV as a full evaluation")
    repeats, error_rows = [], []
    for seed in REPEAT_SEEDS:
        probability = np.full((len(dataset.y), len(dataset.classes)), np.nan)
        visits = np.zeros(len(dataset.y), dtype=int)
        for result in results:
            if result["seed"] != seed:
                continue
            indices = result["indices"]
            probability[indices] = result["probabilities"]
            visits[indices] += 1
        if not np.all(visits == 1) or not np.isfinite(probability).all():
            raise ValueError("outer predictions must cover every sample exactly once per repeat")
        prediction = probability.argmax(axis=1)
        old = dataset.historical_predictions[seed]
        corrected, regressed = (old != dataset.y) & (prediction == dataset.y), (old == dataset.y) & (prediction != dataset.y)
        repeats.append({"seed": seed, **scores(dataset.y, prediction),
                        "historical": scores(dataset.y, old),
                        "corrected": int(corrected.sum()), "regressed": int(regressed.sum()),
                        "confusion_matrix": confusion_matrix(dataset.y, prediction, labels=np.arange(len(dataset.classes))).tolist(),
                        "historical_confusion_matrix": confusion_matrix(dataset.y, old, labels=np.arange(len(dataset.classes))).tolist()})
        for i, meta in enumerate(dataset.metadata):
            error_rows.append({"sample_index": int(meta["subset_sample_index"]), "sample_id": meta["sample_id"],
                               "class_name": meta["class_name"], "seed": seed, "label": int(dataset.y[i]),
                               "prediction": int(prediction[i]), "historical_prediction": int(old[i]),
                               "corrected": bool(corrected[i]), "regressed": bool(regressed[i])})
    mean = float(np.mean([row["accuracy"] for row in repeats]))
    historical_mean = float(np.mean([row["historical"]["accuracy"] for row in repeats]))
    return {"status": "complete", "sample_count": len(dataset.y), "prediction_count": len(error_rows),
            "independent_test_available": False, "repeats": repeats,
            "mean_accuracy": mean, "std_accuracy": float(np.std([row["accuracy"] for row in repeats], ddof=1)),
            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in repeats])),
            "mean_macro_recall": float(np.mean([row["macro_recall"] for row in repeats])),
            "historical_mean_accuracy": historical_mean, "mean_accuracy_delta": mean - historical_mean,
            "improved_mean": mean > historical_mean, "paired_predictions": error_rows}


def fit_final(dataset, output, *, workers=6, hours=1.0, candidates=None):
    output = Path(output)
    candidates = candidates or freeze_candidates()
    fingerprint = freeze_run(dataset, candidates, output)
    if not (output / "evaluation_summary.json").exists():
        raise ValueError("complete the outer evaluation before fitting the delivery model")
    summary = json.loads((output / "evaluation_summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("outer evaluation is incomplete")
    inner = run_inner(candidates, dataset.x, dataset.y, 20260922, output / "final_inner", time.time() + hours * 3600, fingerprint, workers)
    selected, options = select_readout(inner, dataset.y)
    write_json(output / "final_selection.json", {"selected": selected, "options": options, "training_samples": len(dataset.y),
                                               "note": "All-data CV chooses a delivery configuration; it is not an independent test score."})
    readout = fit_selected(selected, candidates, dataset.x, dataset.y, dataset.classes)
    destination = output / "model.joblib"
    joblib.dump(readout, destination)
    reloaded = joblib.load(destination)
    with threadpool_limits(limits=1):
        np.testing.assert_allclose(readout.predict_proba(dataset.x), reloaded.predict_proba(dataset.x), rtol=0, atol=0)
    emit("fit_complete", selected=selected["members"], weights=selected["weights"], model=str(destination))
    return selected
