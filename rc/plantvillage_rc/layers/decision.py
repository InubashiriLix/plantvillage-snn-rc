"""Decision layer: measured currents in, calibrated class ranking out.

This module knows only the response tensor contract.  It can be tested with an
injected sklearn-compatible estimator and has no dependency on the image
processing or acquisition implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contracts import DecisionResult, HardwareResponse, LayerUnavailable


def _model_classes(model: Any) -> tuple[Any, ...]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = getattr(model, "classes", None)
    if classes is None:
        raise LayerUnavailable("readout model does not expose classes_/classes")
    return tuple(classes.tolist() if hasattr(classes, "tolist") else classes)


class ReadoutDecisionLayer:
    """A thin inference boundary around the frozen hardware readout model."""

    layer_name = "decision"

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        model: Any | None = None,
        model_name: str | None = None,
    ):
        self.model_path = Path(model_path) if model_path is not None else None
        self._model = model
        self.model_name = model_name or (self.model_path.name if self.model_path else "injected-readout")
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        if self._model is not None:
            return True
        return self.model_path is not None and self.model_path.is_file()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        return {
            "layer": self.layer_name,
            "available": self.available,
            "loaded": self.loaded,
            "model": self.model_name,
            "model_path": self.model_path.as_posix() if self.model_path else None,
            "error": self._load_error,
        }

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self.model_path is None or not self.model_path.is_file():
            raise LayerUnavailable(
                "decision model is not installed; pass --model or set "
                "PLANTVILLAGE_HARDWARE_MODEL"
            )
        try:
            import joblib

            # Import the module that owns FittedReadout so joblib can resolve
            # the class while loading the trusted local artifact.
            import plantvillage_rc.hardware_models  # noqa: F401

            self._model = joblib.load(self.model_path)
        except Exception as error:
            self._load_error = str(error)
            raise LayerUnavailable(f"unable to load readout model: {error}") from error
        if not hasattr(self._model, "predict_proba"):
            raise LayerUnavailable("readout model must implement predict_proba")
        return self._model

    @property
    def classes(self) -> tuple[str, ...]:
        model = self._load()
        return tuple(str(value) for value in _model_classes(model))

    def _probabilities(self, currents: np.ndarray) -> np.ndarray:
        model = self._load()
        values = np.asarray(currents, dtype=np.float64)
        if values.ndim != 4 or values.shape[1:] != (9, 12, 65):
            raise ValueError("decision layer expects currents shaped [N,9,12,65]")
        if not np.isfinite(values).all():
            raise ValueError("decision layer received non-finite currents")
        probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
        expected = (len(values), len(_model_classes(model)))
        if probabilities.shape != expected:
            raise ValueError(
                f"model returned probabilities {probabilities.shape}, expected {expected}"
            )
        probabilities = np.clip(probabilities, 0.0, None)
        totals = probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(totals).all() or np.any(totals <= 0):
            raise ValueError("model returned invalid probability rows")
        return probabilities / totals

    def decide(self, response: HardwareResponse | np.ndarray) -> DecisionResult:
        currents = response.currents if isinstance(response, HardwareResponse) else response
        values = np.asarray(currents, dtype=np.float64)
        if values.ndim == 3:
            values = values[None, ...]
        if values.shape[0] != 1:
            raise ValueError("decide() accepts exactly one hardware response")
        classes = tuple(str(value) for value in _model_classes(self._load()))
        probabilities = self._probabilities(values)[0]
        predicted = int(np.argmax(probabilities))
        order = np.argsort(-probabilities, kind="stable")
        top_k = tuple(
            {
                "index": int(index),
                "class_name": classes[int(index)],
                "probability": float(probabilities[int(index)]),
            }
            for index in order
        )
        return DecisionResult(
            predicted_index=predicted,
            predicted_class=classes[predicted],
            probabilities=probabilities,
            classes=classes,
            model=self.model_name,
            top_k=top_k,
        )

    def decide_batch(self, currents: Sequence[np.ndarray] | np.ndarray) -> list[DecisionResult]:
        values = np.asarray(currents, dtype=np.float64)
        if values.ndim == 3:
            values = values[None, ...]
        if values.ndim != 4 or values.shape[1:] != (9, 12, 65):
            raise ValueError("decide_batch() expects [N,9,12,65] currents")
        probabilities = self._probabilities(values)
        classes = tuple(str(value) for value in _model_classes(self._load()))
        results: list[DecisionResult] = []
        for row in probabilities:
            predicted = int(np.argmax(row))
            order = np.argsort(-row, kind="stable")
            results.append(
                DecisionResult(
                    predicted_index=predicted,
                    predicted_class=classes[predicted],
                    probabilities=row,
                    classes=classes,
                    model=self.model_name,
                    top_k=tuple(
                        {
                            "index": int(index),
                            "class_name": classes[int(index)],
                            "probability": float(row[int(index)]),
                        }
                        for index in order
                    ),
                )
            )
        return results
