"""Pipeline step for background extraction and gradient removal."""

from __future__ import annotations

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.processing.background.extractor import BackgroundExtractor, ModelMethod
from astroai.processing.background.gradient_remover import GradientRemover

__all__ = ["BackgroundRemovalStep"]


class BackgroundRemovalStep(PipelineStep):
    """Pipeline step that removes background gradients from stacked images."""

    def __init__(
        self,
        tile_size: int = 64,
        method: ModelMethod = ModelMethod.RBF,
        poly_degree: int = 3,
        preserve_median: bool = True,
    ) -> None:
        extractor = BackgroundExtractor(
            tile_size=tile_size,
            method=method,
            poly_degree=poly_degree,
        )
        self._remover = GradientRemover(
            extractor=extractor,
            preserve_median=preserve_median,
        )

    @property
    def name(self) -> str:
        return "Background Removal"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        if context.result is not None:
            context.result = self._remover.remove(context.result)
        elif context.images:
            total = len(context.images)
            for i in range(total):
                progress(PipelineProgress(
                    stage=self.stage,
                    current=i,
                    total=total,
                    message=f"Removing background {i + 1}/{total}",
                ))
                context.images[i] = self._remover.remove(context.images[i])
        return context
