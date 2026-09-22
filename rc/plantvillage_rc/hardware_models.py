"""Fold-local estimators for fixed nine-channel measured hardware responses."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ("full", "mean_std", "median", "compensated_090", "compensated_098", "compensated_100", "spatial")
FAMILIES = ("Ridge", "SVM_RBF", "RandomForest", "ExtraTrees", "HistBoost", "MLP")
TREE_FAMILIES = ("RandomForest", "ExtraTrees", "HistBoost")


class HardwareFeatures(TransformerMixin, BaseEstimator):
    """Only deterministic, per-image operations; no label or input-image access."""
    def __init__(self, kind="mean_std", unit_scale=1e6):
        self.kind = kind
        self.unit_scale = unit_scale

    def fit(self, x, y=None):
        self.n_features_in_ = 7020
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 4 or values.shape[1:] != (9, 12, 65) or not np.isfinite(values).all():
            raise ValueError("expected finite responses shaped [N,9,12,65]")
        values = values * self.unit_scale
        if self.kind == "full":
            return values.reshape(len(values), -1)
        if self.kind == "median":
            return np.median(values, axis=2).reshape(len(values), -1)
        if self.kind.startswith("compensated_"):
            rho = {"compensated_090": 0.9, "compensated_098": 0.98, "compensated_100": 1.0}[self.kind]
            # First response and separately timed tail are retained unchanged.
            values = values.copy()
            values[:, :, :, 1:64] -= rho * (np.asarray(x) * self.unit_scale)[:, :, :, :63]
        if self.kind == "mean_std" or self.kind.startswith("compensated_"):
            return np.concatenate((values.mean(axis=2).reshape(len(values), -1), values.std(axis=2).reshape(len(values), -1)), axis=1)
        if self.kind == "spatial":
            mean, std = values.mean(axis=2), values.std(axis=2)
            scans = np.concatenate((mean[:, :, :64], std[:, :, :64]), axis=1)
            grid = scans.reshape(-1, 18, 8, 8).copy()
            grid[:, :, 1::2, :] = grid[:, :, 1::2, ::-1]
            blocks = []
            for size in (2, 4, 8):
                pooled = grid.reshape(len(grid), 18, size, 8 // size, size, 8 // size).mean(axis=(3, 5))
                blocks.append(pooled.reshape(len(grid), -1))
            blocks.append(np.concatenate((mean[:, :, 64], std[:, :, 64]), axis=1))
            return np.concatenate(blocks, axis=1)
        raise ValueError(f"unknown feature representation: {self.kind}")


@dataclass(frozen=True)
class Candidate:
    id: str
    family: str
    feature: str
    parameters: dict
    pca_components: int | None = None
    seed: int = 20260922

    def to_dict(self):
        return asdict(self)


def freeze_candidates(seed=20260922):
    rng = np.random.default_rng(seed)
    candidates = []
    choose = lambda values: values[int(rng.integers(len(values)))]
    for family in FAMILIES:
        features = rng.permutation(FEATURES)[:6]
        for index, feature in enumerate(features):
            pca = None
            if family == "Ridge":
                parameters = {"alpha": choose([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])}
            elif family == "SVM_RBF":
                parameters = {"C": choose([0.1, 1.0, 10.0, 100.0]), "gamma": choose(["scale", 0.001, 0.01, 0.1])}
                pca = choose([None, 32, 64])
            elif family in ("RandomForest", "ExtraTrees"):
                parameters = {"n_estimators": 400, "min_samples_leaf": choose([1, 2, 4]), "max_features": choose(["sqrt", 0.3, 1.0])}
            elif family == "HistBoost":
                parameters = {"max_iter": 200, "max_bins": 32, "learning_rate": choose([0.03, 0.1]), "max_leaf_nodes": choose([7, 15, 31]), "l2_regularization": choose([0.1, 1.0, 10.0])}
            else:
                parameters = {"hidden_layer_sizes": choose([(64,), (128, 32)]), "alpha": choose([0.01, 0.1, 1.0]), "max_iter": 300}
                pca = choose([None, 32, 64])
            candidates.append(Candidate(f"{family}_{index:02d}", family, str(feature), parameters, pca, seed))
    return candidates


def build_model(candidate, *, unit_scale=1e6):
    family, params = candidate.family, candidate.parameters
    steps = [("features", HardwareFeatures(candidate.feature, unit_scale))]
    if family not in TREE_FAMILIES:
        steps.append(("scale", StandardScaler()))
    if candidate.pca_components is not None:
        steps.append(("pca", PCA(n_components=candidate.pca_components, svd_solver="randomized", random_state=candidate.seed)))
    if family == "Ridge":
        model = RidgeClassifier(**params)
    elif family == "SVM_RBF":
        model = SVC(**params, probability=True, random_state=candidate.seed)
    elif family == "RandomForest":
        model = RandomForestClassifier(**params, n_jobs=1, random_state=candidate.seed)
    elif family == "ExtraTrees":
        model = ExtraTreesClassifier(**params, n_jobs=1, random_state=candidate.seed)
    elif family == "HistBoost":
        model = HistGradientBoostingClassifier(**params, early_stopping=True, validation_fraction=0.15, random_state=candidate.seed)
    elif family == "MLP":
        model = MLPClassifier(**params, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=candidate.seed)
    else:
        raise ValueError(f"unknown model family {family}")
    return Pipeline(steps + [("model", model)])


def probabilities(model, x, class_count):
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x)
    else:
        decisions = model.decision_function(x)
        if decisions.ndim == 1:
            decisions = np.column_stack((-decisions, decisions))
        raw = softmax(decisions, axis=1)
    result = np.zeros((len(x), class_count), dtype=np.float64)
    result[:, np.asarray(model.classes_, dtype=int)] = raw
    if not np.isfinite(result).all() or np.any(result < 0) or not np.allclose(result.sum(axis=1), 1):
        raise ValueError("invalid predicted probabilities")
    return result


class FittedReadout:
    def __init__(self, models, weights, classes):
        self.models = models
        self.weights = weights
        self.classes = classes

    def predict_proba(self, x):
        return sum(weight * probabilities(model, x, len(self.classes)) for model, weight in zip(self.models, self.weights))

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)
