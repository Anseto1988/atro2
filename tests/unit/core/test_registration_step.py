"""Unit tests for RegistrationStep."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.engine.registration.pipeline_step import RegistrationStep


def _make_context(n: int = 3, h: int = 32, w: int = 32) -> PipelineContext:
    rng = np.random.default_rng(0)
    images = [rng.random((h, w)).astype(np.float32) for _ in range(n)]
    return PipelineContext(images=images)


class TestRegistrationStepBasic:
    def test_name(self) -> None:
        step = RegistrationStep()
        assert step.name == "Registration"

    def test_stage_is_registration(self) -> None:
        step = RegistrationStep()
        assert step.stage == PipelineStage.REGISTRATION

    def test_single_frame_passthrough(self) -> None:
        step = RegistrationStep()
        ctx = _make_context(n=1)
        original = ctx.images[0]
        out = step.execute(ctx)
        assert len(out.images) == 1
        assert out.images[0] is original

    def test_empty_context_passthrough(self) -> None:
        step = RegistrationStep()
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.images == []

    def test_frame_count_preserved(self) -> None:
        step = RegistrationStep(upsample_factor=1)
        ctx = _make_context(n=4)
        out = step.execute(ctx)
        assert len(out.images) == 4

    def test_reference_frame_unchanged(self) -> None:
        step = RegistrationStep(upsample_factor=1, reference_frame_index=0)
        ctx = _make_context(n=3)
        ref_before = ctx.images[0].copy()
        out = step.execute(ctx)
        np.testing.assert_array_equal(out.images[0], ref_before)

    def test_metadata_reference_index_stored(self) -> None:
        step = RegistrationStep(upsample_factor=1, reference_frame_index=1)
        ctx = _make_context(n=3)
        out = step.execute(ctx)
        assert out.metadata["registration_reference_index"] == 1

    def test_metadata_frames_aligned_stored(self) -> None:
        step = RegistrationStep(upsample_factor=1)
        ctx = _make_context(n=3)
        out = step.execute(ctx)
        assert out.metadata["registration_frames_aligned"] == 3

    def test_reference_index_clamped_to_valid_range(self) -> None:
        step = RegistrationStep(upsample_factor=1, reference_frame_index=99)
        ctx = _make_context(n=2)
        out = step.execute(ctx)
        assert out.metadata["registration_reference_index"] == 1

    def test_progress_called(self) -> None:
        step = RegistrationStep(upsample_factor=1)
        ctx = _make_context(n=2)
        calls: list[object] = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) >= 2

    def test_output_shape_matches_input(self) -> None:
        step = RegistrationStep(upsample_factor=1)
        ctx = _make_context(n=2, h=16, w=24)
        out = step.execute(ctx)
        for img in out.images:
            assert img.shape == (16, 24)
