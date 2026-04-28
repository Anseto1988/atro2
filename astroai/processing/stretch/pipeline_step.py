"""Pipeline step for intelligent histogram stretching."""
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
from astroai.processing.stretch.stretcher import IntelligentStretcher

__all__ = ["StretchStep"]

logger = logging.getLogger(__name__)


class StretchStep(PipelineStep):
    """Auto-stretch linear astrophotography frames via STF/MTF curve."""

    def __init__(
        self,
        target_background: float = 0.25,
        shadow_clipping_sigmas: float = -2.8,
        linked_channels: bool = True,
    ) -> None:
        self._stretcher = IntelligentStretcher(
            target_background=target_background,
            shadow_clipping_sigmas=shadow_clipping_sigmas,
            linked_channels=linked_channels,
        )

    @property
    def name(self) -> str:
        return "Stretch"

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
                stage=self.stage, current=0, total=1, message="Stretch läuft…",
            ))
            context.result = self._stretcher.stretch(context.result)
            progress(PipelineProgress(
                stage=self.stage, current=1, total=1, message="Stretch abgeschlossen",
            ))
        elif context.images:
            total = len(context.images)
            for i in range(total):
                progress(PipelineProgress(
                    stage=self.stage,
                    current=i,
                    total=total,
                    message=f"Stretch {i + 1}/{total}",
                ))
                context.images[i] = self._stretcher.stretch(context.images[i])
        return context
