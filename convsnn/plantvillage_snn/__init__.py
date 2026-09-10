"""Reproducible compact hardware-aware PlantVillage SNN pipeline."""

from .config import ExperimentConfig
from .models import build_model

__all__ = ["ExperimentConfig", "build_model"]
__version__ = "2.0.0"
