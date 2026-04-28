"""Pipeline step for frame stacking (mean / median / sigma-clip)."""
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
from astroai.engine.stacking.stacker import FrameStacker

__all__ = ["StackingStep"]

logger = logging.getLogger(__name__)

_STACKING_STAGE = PipelineStage.STACKING


class StackingStep(PipelineStep):
    """Combine frames in context.images into a single stacked result."""

    def __init__(
        self,
        method: str = "sigma_clip",
        sigma_low: float = 2.5,
        sigma_high: float = 2.5,
    ) -> None:
        self._stacker = FrameStacker()
        self._method = method
        self._sigma_low = sigma_low
        self._sigma_high = sigma_high

    @property
    def name(self) -> str:
        return "Stacking"

    @property
    def stage(self) -> PipelineStage:
        return _STACKING_STAGE

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        frames = context.images
        if not frames:
            return context

        n = len(frames)
        progress(PipelineProgress(
            stage=self.stage, current=0, total=1,
            message=f"Stacke {n} Frames ({self._method})…",
        ))

        kwargs = {}
        if self._method == "sigma_clip":
            kwargs = {"sigma_low": self._sigma_low, "sigma_high": self._sigma_high}

        result = self._stacker.stack(frames, method=self._method, **kwargs)
        context.result = result
        context.metadata["stacking_method"] = self._method
        context.metadata["stacking_frame_count"] = n

        progress(PipelineProgress(
            stage=self.stage, current=1, total=1,
            message=f"Stacking abgeschlossen ({n} Frames)",
        ))
        logger.info("Stacking complete: %d frames → %s", n, self._method)
        return context
