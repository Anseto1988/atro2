"""Pipeline step for frame registration (star-detection or phase-correlation)."""
from __future__ import annotations

import logging
from typing import Literal

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.engine.registration.aligner import FrameAligner
from astroai.engine.registration.star_aligner import StarAligner

__all__ = ["RegistrationStep"]

logger = logging.getLogger(__name__)

RegistrationMethod = Literal["star", "phase_correlation"]


class RegistrationStep(PipelineStep):
    """Align frames using star detection (LoG) or phase correlation."""

    def __init__(
        self,
        upsample_factor: int = 10,
        reference_frame_index: int = 0,
        method: RegistrationMethod = "star",
    ) -> None:
        self._ref_index = max(0, reference_frame_index)
        self._method: RegistrationMethod = method
        if method == "star":
            self._aligner: FrameAligner | StarAligner = StarAligner(
                upsample_factor=upsample_factor
            )
        else:
            self._aligner = FrameAligner(upsample_factor=upsample_factor)

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
            message=f"Registriere {n} Frames ({self._method})…",
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
        context.metadata["registration_method"] = self._method
        logger.info(
            "Registration complete: %d frames aligned to index %d (method=%s)",
            n, ref_idx, self._method,
        )
        return context
