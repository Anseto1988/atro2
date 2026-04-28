"""Pipeline step: generate and apply a synthetic flat frame."""
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
from astroai.processing.background.extractor import ModelMethod
from astroai.processing.flat.synthetic_generator import SyntheticFlatGenerator

__all__ = ["SyntheticFlatStep"]

logger = logging.getLogger(__name__)


class SyntheticFlatStep(PipelineStep):
    """Generate a synthetic flat from loaded light frames and apply it.

    When real flat frames are unavailable this step models the illumination
    gradient from the science frames themselves and divides it out.
    """

    def __init__(
        self,
        tile_size: int = 64,
        method: ModelMethod = ModelMethod.RBF,
        poly_degree: int = 4,
        smoothing_sigma: float = 8.0,
    ) -> None:
        self._generator = SyntheticFlatGenerator(
            tile_size=tile_size,
            method=method,
            poly_degree=poly_degree,
            smoothing_sigma=smoothing_sigma,
        )

    @property
    def name(self) -> str:
        return "Synthetic Flat"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.CALIBRATION

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        import numpy as np

        frames = context.images
        if not frames:
            logger.warning("SyntheticFlatStep: no images in context, skipping")
            return context

        progress(PipelineProgress(
            stage=self.stage,
            current=0,
            total=len(frames) + 1,
            message="Generating synthetic flat…",
        ))

        synthetic_flat = self._generator.generate(frames)

        total = len(frames)
        for i in range(total):
            progress(PipelineProgress(
                stage=self.stage,
                current=i + 1,
                total=total + 1,
                message=f"Applying synthetic flat {i + 1}/{total}",
            ))
            flat_safe = np.maximum(synthetic_flat, 1e-7)
            context.images[i] = (frames[i] / flat_safe).astype(frames[i].dtype)

        progress(PipelineProgress(
            stage=self.stage,
            current=total + 1,
            total=total + 1,
            message="Synthetic flat applied",
        ))

        context.metadata["synthetic_flat_applied"] = True
        return context
