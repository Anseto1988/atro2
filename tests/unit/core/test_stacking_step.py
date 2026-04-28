"""Unit tests for StackingStep."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.engine.stacking.pipeline_step import StackingStep


def _make_context(n: int = 3, h: int = 16, w: int = 16) -> PipelineContext:
    rng = np.random.default_rng(7)
    images = [rng.random((h, w)).astype(np.float32) for _ in range(n)]
    return PipelineContext(images=images)


class TestStackingStepBasic:
    def test_name(self) -> None:
        step = StackingStep()
        assert step.name == "Stacking"

    def test_stage_is_stacking(self) -> None:
        step = StackingStep()
        assert step.stage == PipelineStage.STACKING

    def test_empty_context_passthrough(self) -> None:
        step = StackingStep()
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out is ctx

    def test_produces_result(self) -> None:
        step = StackingStep(method="mean")
        ctx = _make_context(n=3)
        out = step.execute(ctx)
        assert out.result is not None

    def test_result_shape_matches_frames(self) -> None:
        step = StackingStep(method="mean")
        ctx = _make_context(n=3, h=16, w=24)
        out = step.execute(ctx)
        assert out.result.shape == (16, 24)

    def test_metadata_stacking_method_stored(self) -> None:
        step = StackingStep(method="median")
        ctx = _make_context(n=2)
        out = step.execute(ctx)
        assert out.metadata["stacking_method"] == "median"

    def test_metadata_frame_count_stored(self) -> None:
        step = StackingStep(method="mean")
        ctx = _make_context(n=5)
        out = step.execute(ctx)
        assert out.metadata["stacking_frame_count"] == 5

    def test_sigma_clip_method(self) -> None:
        step = StackingStep(method="sigma_clip", sigma_low=2.5, sigma_high=2.5)
        ctx = _make_context(n=4)
        out = step.execute(ctx)
        assert out.result is not None
        assert out.metadata["stacking_method"] == "sigma_clip"

    def test_mean_method(self) -> None:
        step = StackingStep(method="mean")
        ctx = _make_context(n=3)
        out = step.execute(ctx)
        assert out.result is not None

    def test_median_method(self) -> None:
        step = StackingStep(method="median")
        ctx = _make_context(n=3)
        out = step.execute(ctx)
        assert out.result is not None

    def test_progress_called(self) -> None:
        step = StackingStep(method="mean")
        ctx = _make_context(n=2)
        calls: list[object] = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) >= 2

    def test_single_frame_stacked(self) -> None:
        step = StackingStep(method="mean")
        ctx = _make_context(n=1)
        out = step.execute(ctx)
        assert out.result is not None
