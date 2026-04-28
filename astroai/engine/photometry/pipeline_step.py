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
from astroai.engine.photometry.engine import PhotometryEngine
from astroai.engine.photometry.models import PhotometryResult

__all__ = ["PhotometryStep"]

logger = logging.getLogger(__name__)


class PhotometryStep(PipelineStep):
    def __init__(
        self,
        engine: PhotometryEngine | None = None,
        fail_silently: bool = True,
    ) -> None:
        self._engine = engine or PhotometryEngine()
        self._fail_silently = fail_silently

    @property
    def name(self) -> str:
        return "Photometry"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PHOTOMETRY

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        progress(PipelineProgress(
            stage=self.stage, current=0, total=1,
            message="Running photometry…",
        ))

        wcs = context.metadata.get("wcs_solution") or context.metadata.get("wcs")
        if wcs is None:
            msg = "PhotometryStep: no WCS solution in context"
            if self._fail_silently:
                logger.warning(msg)
                return context
            raise RuntimeError(msg)

        image = context.result
        if image is None and context.images:
            image = context.images[0]

        if image is None:
            msg = "PhotometryStep: no image in context"
            if self._fail_silently:
                logger.warning(msg)
                return context
            raise RuntimeError(msg)

        try:
            result: PhotometryResult = self._engine.run(image, wcs)
            context.metadata["photometry_result"] = result
            logger.info(
                "Photometry complete: %d stars matched, R²=%.4f",
                result.n_matched, result.r_squared,
            )
        except Exception as exc:
            if self._fail_silently:
                logger.warning("Photometry failed (skipped): %s", exc)
            else:
                raise

        progress(PipelineProgress(
            stage=self.stage, current=1, total=1,
            message="Photometry complete",
        ))
        return context
