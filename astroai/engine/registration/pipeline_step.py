"""Pipeline step for phase-correlation frame registration."""
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
from astroai.engine.registration.aligner import FrameAligner

__all__ = ["RegistrationStep"]

logger = logging.getLogger(__name__)


class RegistrationStep(PipelineStep):
    """Align all frames in context.images to a reference using phase correlation."""

    def __init__(
        self,
        upsample_factor: int = 10,
        reference_frame_index: int = 0,
    ) -> None:
        self._aligner = FrameAligner(upsample_factor=upsample_factor)
        self._ref_index = max(0, reference_frame_index)

    @property
    def name(self) -> str:
        return "Registration"

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
            return context

        ref_idx = min(self._ref_index, len(frames) - 1)
        reference = frames[ref_idx]
        n = len(frames)

        progress(PipelineProgress(
            stage=self.stage, current=0, total=n,
            message=f"Registriere {n} Frames…",
        ))

        aligned = []
        for i, frame in enumerate(frames):
            if i == ref_idx:
                aligned.append(frame)
            else:
                aligned_frame, _ = self._aligner.align(reference, frame)
                aligned.append(aligned_frame)
            progress(PipelineProgress(
                stage=self.stage, current=i + 1, total=n,
                message=f"Frame {i + 1}/{n} registriert",
            ))

        context.images = aligned
        context.metadata["registration_reference_index"] = ref_idx
        context.metadata["registration_frames_aligned"] = n
        logger.info("Registration complete: %d frames aligned to index %d", n, ref_idx)
        return context
