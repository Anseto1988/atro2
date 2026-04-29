"""Pipeline step wrapper for UnsharpMask sharpening."""
from __future__ import annotations

import logging

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.processing.sharpening.unsharp_mask import UnsharpMask

__all__ = ["SharpeningStep"]

logger = logging.getLogger(__name__)


class SharpeningStep(PipelineStep):
    """Sharpen the stacked result using unsharp masking."""

    def __init__(
        self,
        radius: float = 1.0,
        amount: float = 0.5,
        threshold: float = 0.02,
    ) -> None:
        self._sharpener = UnsharpMask(
            radius=radius,
            amount=amount,
            threshold=threshold,
        )

    @property
    def name(self) -> str:
        return "Schärfung"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        if context.result is not None:
            progress(PipelineProgress(
                stage=self.stage, current=0, total=1, message="Schärfung läuft…",
            ))
            context.result = self._sharpener.apply(context.result)
            progress(PipelineProgress(
                stage=self.stage, current=1, total=1, message="Schärfung abgeschlossen",
            ))
        return context
