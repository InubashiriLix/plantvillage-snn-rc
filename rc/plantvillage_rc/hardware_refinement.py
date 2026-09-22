"""Second-round response features; no changes to the frozen first-round code."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def distribution(values, tail):
    """18 statistics per trace; tail is separate from the 64 regular steps."""
    delta = np.diff(values, axis=-1)
    quantiles = np.moveaxis(np.quantile(values, np.linspace(0, 1, 11), axis=-1), 0, -1)
    return np.concatenate((quantiles, np.stack((values.mean(-1), values.std(-1),
                           np.abs(delta).mean(-1), delta.std(-1), values[..., 0],
                           values[..., -1], tail), axis=-1)), axis=-1)


class ResponseFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, kind="distribution", rho=0.98):
        self.kind = kind
        self.rho = rho

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 4 or x.shape[1:] != (9, 12, 65) or not np.isfinite(x).all():
            raise ValueError("expected finite [N,9,12,65] hardware responses")
        currents = x * 1e6
        regular, tail = currents[..., :64], currents[..., 64]
        compensated = regular.copy()
        compensated[..., 1:] -= self.rho * regular[..., :-1]
        sequence = compensated.mean(axis=2).reshape(len(x), -1)
        if self.kind == "sequence":
            return sequence
        raw_stats = distribution(regular, tail).reshape(len(x), -1)
        if self.kind == "distribution":
            return raw_stats
        compensated_stats = distribution(compensated, tail).reshape(len(x), -1)
        if self.kind == "compensated_distribution":
            return compensated_stats
        if self.kind == "mixed":
            return np.concatenate((raw_stats, sequence), axis=1)
        if self.kind == "multiscale":
            # Device response statistics, local response changes and global color relations.
            mean = compensated.mean(axis=2)
            triples = mean.reshape(len(x), 3, 3, 64)
            relative = triples / (np.abs(triples).sum(axis=2, keepdims=True) + 0.05)
            relative_stats = np.moveaxis(np.quantile(relative, np.linspace(0, 1, 9), axis=-1), 0, -1).reshape(len(x), -1)
            grid = mean.reshape(len(x), 9, 8, 8).copy()
            grid[:, :, 1::2] = grid[:, :, 1::2, ::-1]
            pooled = grid.reshape(len(x), 9, 4, 2, 4, 2).mean(axis=(3, 5)).reshape(len(x), -1)
            return np.concatenate((raw_stats, compensated_stats, relative_stats, pooled), axis=1)
        raise ValueError(f"unknown response feature: {self.kind}")


def build_refined_model(config):
    steps = [("features", ResponseFeatures(config["feature"], config.get("rho", 0.98)))]
    family = config["family"]
    if family in ("SVM", "MLP"):
        steps.append(("scale", StandardScaler()))
    if config.get("pca"):
        steps.append(("pca", PCA(n_components=config["pca"], whiten=config.get("whiten", False),
                                 random_state=20260923, svd_solver="full")))
    if family == "SVM":
        # Multipliers relative to feature dimension, avoiding an over-narrow high-D kernel.
        model = SVC(C=config["C"], gamma=config["gamma"], probability=True, random_state=20260923)
    elif family == "MLP":
        model = MLPClassifier(hidden_layer_sizes=(config.get("hidden", 64),), solver="lbfgs", alpha=config["alpha"],
                              activation=config.get("activation", "relu"), max_iter=400, max_fun=20000,
                              random_state=config.get("seed", 20260923))
    elif family in ("RF", "ET"):
        cls = RandomForestClassifier if family == "RF" else ExtraTreesClassifier
        model = cls(n_estimators=config.get("trees", 400), min_samples_leaf=config.get("leaf", 1),
                    max_features=config["max_features"], random_state=20260923, n_jobs=1)
    else:
        raise ValueError(f"unknown classifier {family}")
    return Pipeline(steps + [("model", model)])
