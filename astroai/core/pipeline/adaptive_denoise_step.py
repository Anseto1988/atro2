"""Pipeline step for adaptive AI-powered denoising based on noise estimation."""
from __future__ import annotations

import logging

from astroai.core.noise_estimator import NoiseEstimator
from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.processing.denoise.pipeline_step import DenoiseStep

__all__ = ["AdaptiveDenoiseStep"]

logger = logging.getLogger(__name__)


class AdaptiveDenoiseStep(PipelineStep):
    """Estimates noise then delegates to DenoiseStep with suggested strength."""

    METADATA_KEY = "estimated_denoise_strength"

    def __init__(
        self,
        tile_size: int = 512,
        tile_overlap: int = 64,
        estimator_iterations: int = 5,
        estimator_kappa: float = 3.0,
    ) -> None:
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._estimator = NoiseEstimator(
            iterations=estimator_iterations,
            kappa=estimator_kappa,
        )

    @property
    def name(self) -> str:
        return "Adaptive Denoise"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        image = context.result if context.result is not None else (
            context.images[0] if context.images else None
        )
        if image is None:
            logger.warning("AdaptiveDenoiseStep: no image available, skipping")
            return context

        progress(PipelineProgress(
            stage=self.stage, current=0, total=2,
            message="Rauschanalyse läuft…",
        ))

        estimate = self._estimator.estimate(image)
        context.metadata[self.METADATA_KEY] = estimate.suggested_strength
        logger.info("Adaptive denoise: %s", estimate)

        progress(PipelineProgress(
            stage=self.stage, current=1, total=2,
            message=f"Entrauschung mit Stärke {estimate.suggested_strength:.2f}…",
        ))

        delegate = DenoiseStep(
            strength=estimate.suggested_strength,
            tile_size=self._tile_size,
            tile_overlap=self._tile_overlap,
        )
        context = delegate.execute(context, progress)
        return context
