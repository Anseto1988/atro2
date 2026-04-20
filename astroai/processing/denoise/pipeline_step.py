"""Pipeline step for AI-powered denoising."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    _noop_callback,
)
from astroai.processing.denoise.denoiser import Denoiser

__all__ = ["DenoiseStep"]

logger = logging.getLogger(__name__)


class DenoiseStep(PipelineStep):
    """Denoise frames using statistical or AI-backed Denoiser."""

    def __init__(
        self,
        strength: float = 1.0,
        tile_size: int = 256,
        tile_overlap: int = 32,
    ) -> None:
        self._denoiser = Denoiser(
            strength=strength,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

    @property
    def name(self) -> str:
        return "Denoise"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = _noop_callback,
    ) -> PipelineContext:
        if context.result is not None:
            progress(PipelineProgress(
                stage=self.stage, current=0, total=1, message="Rauschreduktion läuft…",
            ))
            context.result = self._denoiser.denoise(context.result)
            progress(PipelineProgress(
                stage=self.stage, current=1, total=1, message="Rauschreduktion abgeschlossen",
            ))
        elif context.images:
            total = len(context.images)
            for i in range(total):
                progress(PipelineProgress(
                    stage=self.stage,
                    current=i,
                    total=total,
                    message=f"Denoise {i + 1}/{total}",
                ))
                context.images[i] = self._denoiser.denoise(context.images[i])
        return context
