"""Unit-Tests für CometStackStep."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.core.pipeline.comet_stack_step import (
    CometStackStep,
    _COMET_NUCLEUS_STACK_KEY,
    _COMET_POSITIONS_KEY,
    _COMET_STAR_STACK_KEY,
)


def _make_context(n: int = 4, h: int = 32, w: int = 32) -> PipelineContext:
    rng = np.random.default_rng(7)
    images = [rng.random((h, w)).astype(np.float32) for _ in range(n)]
    # Füge Komet hinzu (wandernder heller Punkt)
    for i, img in enumerate(images):
        img[10 + i, 10 + i] = 2.0
    return PipelineContext(images=images)


class TestCometStackStep:
    def test_name_contains_mode(self) -> None:
        step = CometStackStep(tracking_mode="blend")
        assert "blend" in step.name.lower()

    def test_stage_is_comet_stacking(self) -> None:
        step = CometStackStep()
        assert step.stage == PipelineStage.COMET_STACKING

    def test_empty_context_skipped(self) -> None:
        step = CometStackStep()
        ctx = PipelineContext()
        result = step.execute(ctx)
        assert result.result is None

    def test_stars_mode_sets_result(self) -> None:
        step = CometStackStep(tracking_mode="stars", stack_method="mean")
        ctx = _make_context(3)
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (32, 32)

    def test_comet_mode_sets_result(self) -> None:
        step = CometStackStep(tracking_mode="comet", stack_method="mean")
        ctx = _make_context(3)
        out = step.execute(ctx)
        assert out.result is not None

    def test_blend_mode_sets_result(self) -> None:
        step = CometStackStep(tracking_mode="blend", blend_factor=0.4, stack_method="mean")
        ctx = _make_context(4)
        out = step.execute(ctx)
        assert out.result is not None

    def test_metadata_contains_both_stacks(self) -> None:
        step = CometStackStep(tracking_mode="blend", stack_method="mean")
        ctx = _make_context(4)
        out = step.execute(ctx)
        assert _COMET_STAR_STACK_KEY in out.metadata
        assert _COMET_NUCLEUS_STACK_KEY in out.metadata
        assert _COMET_POSITIONS_KEY in out.metadata

    def test_positions_count_matches_frames(self) -> None:
        n = 5
        step = CometStackStep(stack_method="mean")
        ctx = _make_context(n)
        out = step.execute(ctx)
        positions = out.metadata.get(_COMET_POSITIONS_KEY, [])
        assert len(positions) == n

    def test_fail_silently_on_error(self) -> None:
        step = CometStackStep(fail_silently=True, top_fraction=0.0)
        ctx = PipelineContext(images=[np.zeros((8, 8), dtype=np.float32)])
        # Soll nicht werfen
        out = step.execute(ctx)
        # result kann None sein wenn alle Pixel 0
        assert True  # kein Exception = OK

    def test_fail_not_silent_raises(self) -> None:
        step = CometStackStep(fail_silently=False, stack_method="unknown_method")
        ctx = _make_context(2)
        with pytest.raises(Exception):
            step.execute(ctx)
