"""Pipeline step for AI-based noise-robust frame registration (FR-2.2)."""
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
from astroai.engine.registration.ai_aligner import AIAligner
from astroai.engine.registration.star_aligner import StarAligner

__all__ = ["AIAlignmentStep"]

logger = logging.getLogger(__name__)


class AIAlignmentStep(PipelineStep):
    """Noise-robust AI-based frame alignment with per-frame confidence scoring.

    Frames with ``alignment_quality < quality_threshold`` are rejected
    (removed from ``context.images``).  ``StarAligner`` is kept as an
    independent fallback for pipelines that do not use this step.
    """

    def __init__(
        self,
        model_path: str | None = None,
        quality_threshold: float = 0.3,
        inlier_threshold: float = 3.0,
        ransac_iterations: int = 1000,
        reference_frame_index: int = 0,
        rng_seed: int | None = None,
    ) -> None:
        self._quality_threshold = quality_threshold
        self._ref_index = max(0, reference_frame_index)
        self._aligner = AIAligner(
            model_path=model_path,
            inlier_threshold=inlier_threshold,
            ransac_iterations=ransac_iterations,
            quality_threshold=quality_threshold,
            rng_seed=rng_seed,
        )

    @property
    def name(self) -> str:
        return "AI Alignment"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.REGISTRATION

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        frames = context.images
        if len(frames) <= 1:
            context.metadata.setdefault("ai_alignment_scores", [1.0] * len(frames))
            context.metadata.setdefault("ai_alignment_rejected", [False] * len(frames))
            context.metadata.setdefault("ai_alignment_reasons", [""] * len(frames))
            return context

        ref_idx = min(self._ref_index, len(frames) - 1)
        reference = frames[ref_idx]
        n = len(frames)

        progress(PipelineProgress(
            stage=self.stage, current=0, total=n,
            message=f"AI-Alignment: {n} Frames…",
        ))

        scores: list[float] = []
        rejected: list[bool] = []
        reasons: list[str] = []
        accepted_frames: list = []

        for i, frame in enumerate(frames):
            if i == ref_idx:
                accepted_frames.append(frame)
                scores.append(1.0)
                rejected.append(False)
                reasons.append("")
            else:
                result = self._aligner.align(reference, frame)
                scores.append(result.confidence)

                if result.confidence < self._quality_threshold:
                    reason = (
                        f"confidence={result.confidence:.3f} < "
                        f"threshold={self._quality_threshold:.3f} "
                        f"(inliers={result.inlier_count}, "
                        f"matched={result.keypoints_matched})"
                    )
                    rejected.append(True)
                    reasons.append(reason)
                    logger.info("Frame %d rejected: %s", i, reason)
                else:
                    accepted_frames.append(result.aligned)
                    rejected.append(False)
                    reasons.append("")

            progress(PipelineProgress(
                stage=self.stage, current=i + 1, total=n,
                message=f"Frame {i + 1}/{n} verarbeitet",
            ))

        n_rejected = sum(rejected)
        context.images = accepted_frames
        context.metadata["ai_alignment_scores"] = scores
        context.metadata["ai_alignment_rejected"] = rejected
        context.metadata["ai_alignment_reasons"] = reasons
        context.metadata["ai_alignment_reference_index"] = ref_idx
        context.metadata["ai_alignment_frames_total"] = n
        context.metadata["ai_alignment_frames_accepted"] = n - n_rejected
        context.metadata["ai_alignment_quality_threshold"] = self._quality_threshold

        logger.info(
            "AI Alignment: %d/%d frames accepted (threshold=%.2f)",
            n - n_rejected, n, self._quality_threshold,
        )
        return context
