"""Thin composition root; the only runtime object that sees all three layers."""
from __future__ import annotations

import time
from typing import Any

from .contracts import DriveSignals, PipelineResult
from .decision import ReadoutDecisionLayer
from .hardware import HardwareBackend
from .processing import LeafImageProcessor


class HardwareRCPipeline:
    def __init__(
        self,
        processor: LeafImageProcessor,
        hardware: HardwareBackend,
        decision: ReadoutDecisionLayer,
    ):
        self.processor = processor
        self.hardware = hardware
        self.decision = decision

    def run_drive(self, drive: DriveSignals) -> PipelineResult:
        started = time.perf_counter()
        response = self.hardware.acquire(drive)
        acquired = time.perf_counter()
        decision = self.decision.decide(response)
        finished = time.perf_counter()
        return PipelineResult(
            drive=drive,
            response=response,
            decision=decision,
            timings_ms={
                "hardware_ms": round((acquired - started) * 1000, 3),
                "decision_ms": round((finished - acquired) * 1000, 3),
                "total_ms": round((finished - started) * 1000, 3),
            },
        )

    def run_image(
        self,
        image: Any,
        *,
        sample_id: str,
        hardware: HardwareBackend | None = None,
    ) -> PipelineResult:
        started = time.perf_counter()
        drive = self.processor.process(image, sample_id=sample_id)
        processed = time.perf_counter()
        result = self.run_drive_with(drive, hardware or self.hardware)
        timings = dict(result.timings_ms)
        timings["processing_ms"] = round((processed - started) * 1000, 3)
        timings["total_ms"] = round(
            timings["total_ms"] + timings["processing_ms"], 3
        )
        return PipelineResult(drive, result.response, result.decision, timings)

    def run_drive_with(
        self, drive: DriveSignals, hardware: HardwareBackend
    ) -> PipelineResult:
        response = hardware.acquire(drive)
        decision = self.decision.decide(response)
        return PipelineResult(drive, response, decision)
