from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.inference.scoring import FrameScorer

__all__ = ["FrameSelectionStep"]


class FrameSelectionStep(PipelineStep):
    """Filters frames below a quality threshold using AI scoring."""

    def __init__(self, min_score: float = 0.5, max_rejected_fraction: float = 0.8) -> None:
        self._min_score = float(np.clip(min_score, 0.0, 1.0))
        self._max_rejected_fraction = float(np.clip(max_rejected_fraction, 0.0, 1.0))
        self._scorer = FrameScorer()

    @property
    def name(self) -> str:
        return "Frame Selection"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.CALIBRATION

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        if not context.images:
            return context

        scores: list[float] = [self._scorer.score(f) for f in context.images]
        passed: list[NDArray[np.floating[Any]]] = []
        rejected_indices: list[int] = []

        for i, (frame, score) in enumerate(zip(context.images, scores)):
            if score >= self._min_score:
                passed.append(frame)
            else:
                rejected_indices.append(i)

        # Safety: never reject more than max_rejected_fraction of frames
        n_total = len(context.images)
        max_rejected = int(n_total * self._max_rejected_fraction)
        if len(rejected_indices) > max_rejected:
            # Re-admit the least-bad rejected frames to stay within limit
            rejected_scores = [(i, scores[i]) for i in rejected_indices]
            rejected_scores.sort(key=lambda x: x[1], reverse=True)
            kept_extra = rejected_scores[: len(rejected_indices) - max_rejected]
            extra_indices = {i for i, _ in kept_extra}
            passed = [
                f for i, f in enumerate(context.images)
                if scores[i] >= self._min_score or i in extra_indices
            ]
            rejected_indices = [i for i in rejected_indices if i not in extra_indices]

        context.images = passed
        context.metadata["frame_scores"] = scores
        context.metadata["frame_selection_rejected"] = rejected_indices
        context.metadata["frame_selection_kept"] = len(passed)
        context.metadata["frame_selection_total"] = n_total
        return context
