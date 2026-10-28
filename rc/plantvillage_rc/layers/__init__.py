"""Composable hardware-RC pipeline layers.

The package deliberately keeps the three runtime responsibilities apart:

* :mod:`plantvillage_rc.layers.processing` turns an image into normalized
  drive signals and never imports a classifier.
* :mod:`plantvillage_rc.layers.hardware` turns drive signals into measured
  currents and never imports an image decoder or classifier.
* :mod:`plantvillage_rc.layers.decision` turns currents into class
  probabilities and never imports the image/hardware implementation.

``plantvillage_rc.layers.orchestrator`` is the only place that knows all three.
"""

from .contracts import (
    DRIVE_CHANNELS,
    RESPONSE_CHANNELS,
    DecisionResult,
    DriveSignals,
    HardwareResponse,
    LayerUnavailable,
    PipelineResult,
)
from .decision import ReadoutDecisionLayer
from .hardware import (
    HttpHardwareBackend,
    RecordedHardwareBackend,
    RecordedHardwareDataset,
    response_from_nested,
)
from .orchestrator import HardwareRCPipeline
from .processing import LeafImageProcessor, ProcessingConfig

__all__ = [
    "DRIVE_CHANNELS",
    "RESPONSE_CHANNELS",
    "DecisionResult",
    "DriveSignals",
    "HardwareResponse",
    "LayerUnavailable",
    "PipelineResult",
    "HttpHardwareBackend",
    "RecordedHardwareBackend",
    "RecordedHardwareDataset",
    "response_from_nested",
    "HardwareRCPipeline",
    "LeafImageProcessor",
    "ProcessingConfig",
    "ReadoutDecisionLayer",
]
