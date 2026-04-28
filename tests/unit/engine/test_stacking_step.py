"""Tests for StackingStep pipeline step."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineProgress, PipelineStage
from astroai.engine.stacking.pipeline_step import StackingStep


def _make_frames(n: int = 3, h: int = 8, w: int = 8) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.random((h, w), dtype=np.float64).astype(np.float32) for _ in range(n)]


class TestStackingStepProperties:
    def test_name(self) -> None:
        assert StackingStep().name == "Stacking"

    def test_stage(self) -> None:
        assert StackingStep().stage == PipelineStage.STACKING

    def test_default_method(self) -> None:
        step = StackingStep()
        assert step._method == "sigma_clip"

    def test_custom_sigma(self) -> None:
        step = StackingStep(method="mean", sigma_low=2.0, sigma_high=3.0)
        assert step._method == "mean"
        assert step._sigma_low == pytest.approx(2.0)
        assert step._sigma_high == pytest.approx(3.0)


class TestStackingStepExecute:
    def test_empty_context_returns_early(self) -> None:
        ctx = PipelineContext()
        result = StackingStep().execute(ctx)
        assert result.result is None

    def test_sigma_clip_produces_result(self) -> None:
        ctx = PipelineContext(images=_make_frames(3))
        result = StackingStep(method="sigma_clip").execute(ctx)
        assert isinstance(result.result, np.ndarray)
        assert result.result.shape == (8, 8)

    def test_mean_method(self) -> None:
        ctx = PipelineContext(images=_make_frames(4))
        result = StackingStep(method="mean").execute(ctx)
        assert result.result is not None

    def test_median_method(self) -> None:
        ctx = PipelineContext(images=_make_frames(4))
        result = StackingStep(method="median").execute(ctx)
        assert result.result is not None

    def test_metadata_set(self) -> None:
        frames = _make_frames(3)
        ctx = PipelineContext(images=frames)
        StackingStep(method="mean").execute(ctx)
        assert ctx.metadata["stacking_method"] == "mean"
        assert ctx.metadata["stacking_frame_count"] == 3

    def test_progress_called(self) -> None:
        progress_calls: list[PipelineProgress] = []
        ctx = PipelineContext(images=_make_frames(2))
        StackingStep().execute(ctx, progress=progress_calls.append)
        assert len(progress_calls) == 2
        assert progress_calls[0].current == 0
        assert progress_calls[1].current == 1

    def test_single_frame_stacks(self) -> None:
        frame = np.full((4, 4), 0.5, dtype=np.float32)
        ctx = PipelineContext(images=[frame])
        result = StackingStep().execute(ctx)
        assert result.result is not None
        assert result.result.shape == (4, 4)

    def test_result_numeric(self) -> None:
        ctx = PipelineContext(images=_make_frames(3))
        result = StackingStep().execute(ctx)
        assert result.result is not None
        assert np.issubdtype(result.result.dtype, np.floating)
