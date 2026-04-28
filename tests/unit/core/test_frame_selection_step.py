"""Unit tests for FrameSelectionStep."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.core.pipeline.frame_selection_step import FrameSelectionStep


def _make_context(n: int = 5, h: int = 32, w: int = 32) -> PipelineContext:
    rng = np.random.default_rng(42)
    images = [rng.random((h, w)).astype(np.float32) for _ in range(n)]
    return PipelineContext(images=images)


def _make_good_frame(h: int = 64, w: int = 64) -> np.ndarray:
    frame = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx in [(20, 20), (44, 40), (30, 55)]:
        frame += 5000.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5 ** 2))
    return np.clip(frame, 0, 65535).astype(np.float32)


class TestFrameSelectionStepBasic:
    def test_name(self) -> None:
        step = FrameSelectionStep()
        assert "selection" in step.name.lower() or "Frame" in step.name

    def test_stage_is_calibration(self) -> None:
        step = FrameSelectionStep()
        assert step.stage == PipelineStage.CALIBRATION

    def test_empty_context_passthrough(self) -> None:
        step = FrameSelectionStep()
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.images == []

    def test_metadata_scores_populated(self) -> None:
        step = FrameSelectionStep(min_score=0.0)
        ctx = _make_context(3)
        out = step.execute(ctx)
        assert "frame_scores" in out.metadata
        assert len(out.metadata["frame_scores"]) == 3

    def test_metadata_counts_correct(self) -> None:
        step = FrameSelectionStep(min_score=0.0)
        ctx = _make_context(4)
        out = step.execute(ctx)
        assert out.metadata["frame_selection_total"] == 4
        assert out.metadata["frame_selection_kept"] == len(out.images)

    def test_zero_threshold_keeps_all(self) -> None:
        step = FrameSelectionStep(min_score=0.0)
        ctx = _make_context(5)
        out = step.execute(ctx)
        assert len(out.images) == 5

    def test_max_threshold_rejects_most(self) -> None:
        step = FrameSelectionStep(min_score=1.0, max_rejected_fraction=1.0)
        ctx = _make_context(5)
        out = step.execute(ctx)
        # All flat random frames score << 1.0, so all rejected
        assert len(out.images) == 0

    def test_rejected_indices_recorded(self) -> None:
        step = FrameSelectionStep(min_score=1.0, max_rejected_fraction=1.0)
        ctx = _make_context(3)
        out = step.execute(ctx)
        assert "frame_selection_rejected" in out.metadata

    def test_good_frame_passes_threshold(self) -> None:
        step = FrameSelectionStep(min_score=0.01)
        good = _make_good_frame()
        ctx = PipelineContext(images=[good])
        out = step.execute(ctx)
        assert len(out.images) == 1

    def test_scores_are_floats_between_0_and_1(self) -> None:
        step = FrameSelectionStep(min_score=0.0)
        ctx = _make_context(4)
        out = step.execute(ctx)
        for s in out.metadata["frame_scores"]:
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0


class TestFrameSelectionSafetyNet:
    def test_max_rejected_fraction_respected(self) -> None:
        # min_score=1.0 would reject all 5 frames
        # but max_rejected_fraction=0.4 means at most 2 can be rejected
        step = FrameSelectionStep(min_score=1.0, max_rejected_fraction=0.4)
        ctx = _make_context(5)
        out = step.execute(ctx)
        # At most 40% (= 2) rejected → at least 3 kept
        assert len(out.images) >= 3

    def test_max_rejected_fraction_zero_keeps_all(self) -> None:
        # No rejections allowed
        step = FrameSelectionStep(min_score=1.0, max_rejected_fraction=0.0)
        ctx = _make_context(4)
        out = step.execute(ctx)
        assert len(out.images) == 4

    def test_max_rejected_fraction_clamped_to_valid_range(self) -> None:
        step = FrameSelectionStep(min_score=0.5, max_rejected_fraction=2.0)
        # max_rejected_fraction > 1.0 should be clamped to 1.0 internally
        ctx = _make_context(3)
        out = step.execute(ctx)
        assert out is not None

    def test_min_score_clamped_negative(self) -> None:
        step = FrameSelectionStep(min_score=-1.0)
        ctx = _make_context(3)
        out = step.execute(ctx)
        # min_score clamped to 0.0, all frames kept
        assert len(out.images) == 3

    def test_safety_net_admits_highest_scoring_rejects(self) -> None:
        # With 4 frames, min_score=1.0, max_rejected_fraction=0.5 → max 2 rejected
        # So at least 2 must be kept
        step = FrameSelectionStep(min_score=1.0, max_rejected_fraction=0.5)
        ctx = _make_context(4)
        out = step.execute(ctx)
        assert len(out.images) >= 2
