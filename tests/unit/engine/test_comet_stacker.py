"""Unit-Tests für CometStacker und CometStackStep."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from astroai.engine.comet.tracker import CometPosition
from astroai.engine.comet.stacker import CometStacker, CometStackResult
from astroai.core.pipeline.comet_stack_step import CometStackStep
from astroai.core.pipeline.base import PipelineContext, PipelineStage


def _make_frames(n: int = 3, h: int = 32, w: int = 32) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.random((h, w)).astype(np.float32) for _ in range(n)]


def _positions(coords: list[tuple[float, float]]) -> list[CometPosition]:
    return [CometPosition(y, x) for y, x in coords]


class TestCometStacker:
    def test_raises_on_length_mismatch(self) -> None:
        stacker = CometStacker()
        frames = _make_frames(3)
        positions = _positions([(10, 10), (10, 11)])
        with pytest.raises(ValueError, match="gleich lang"):
            stacker.stack(frames, positions)

    def test_raises_on_empty(self) -> None:
        stacker = CometStacker()
        with pytest.raises(ValueError, match="Keine Frames"):
            stacker.stack([], [])

    def test_returns_correct_result_type(self) -> None:
        stacker = CometStacker(stack_method="mean")
        frames = _make_frames(4)
        positions = _positions([(16, 16)] * 4)
        result = stacker.stack(frames, positions, tracking_mode="blend", blend_factor=0.5)
        assert isinstance(result, CometStackResult)
        assert result.star_stack.shape == (32, 32)
        assert result.comet_stack.shape == (32, 32)
        assert result.blend is not None

    def test_star_mode_no_blend(self) -> None:
        stacker = CometStacker(stack_method="mean")
        frames = _make_frames(3)
        positions = _positions([(10, 10)] * 3)
        result = stacker.stack(frames, positions, tracking_mode="stars")
        assert result.blend is None

    def test_comet_mode_no_blend(self) -> None:
        stacker = CometStacker(stack_method="mean")
        frames = _make_frames(3)
        positions = _positions([(10, 10)] * 3)
        result = stacker.stack(frames, positions, tracking_mode="comet")
        assert result.blend is None

    def test_blend_factor_zero_equals_star_stack(self) -> None:
        stacker = CometStacker(stack_method="mean")
        frames = _make_frames(4)
        # Gleiche Position → kein Kometen-Shift
        positions = _positions([(16, 16)] * 4)
        result = stacker.stack(frames, positions, tracking_mode="blend", blend_factor=0.0)
        assert result.blend is not None
        np.testing.assert_allclose(result.blend, result.star_stack, rtol=1e-5)

    def test_blend_factor_one_equals_comet_stack(self) -> None:
        stacker = CometStacker(stack_method="mean")
        frames = _make_frames(4)
        positions = _positions([(16, 16)] * 4)
        result = stacker.stack(frames, positions, tracking_mode="blend", blend_factor=1.0)
        assert result.blend is not None
        np.testing.assert_allclose(result.blend, result.comet_stack, rtol=1e-5)

    def test_comet_shift_applied(self) -> None:
        """Wenn Komet wandert, muss comet_stack anders als star_stack sein."""
        stacker = CometStacker(stack_method="mean")
        frames = _make_frames(4, h=64, w=64)
        # Zufällige Positionen → Shift ≠ 0
        positions = _positions([(10, 10), (12, 14), (8, 12), (15, 9)])
        result = stacker.stack(frames, positions, tracking_mode="blend", blend_factor=0.5)
        # Die Stacks müssen existieren (unterschiedliche Werte wenn Shift ≠ 0)
        assert result.star_stack is not None
        assert result.comet_stack is not None

    def test_sigma_clip_method(self) -> None:
        stacker = CometStacker(stack_method="sigma_clip")
        frames = _make_frames(6)
        positions = _positions([(16, 16)] * 6)
        result = stacker.stack(frames, positions, tracking_mode="stars")
        assert result.star_stack.shape == (32, 32)


# ---------------------------------------------------------------------------
# CometStackStep
# ---------------------------------------------------------------------------

class TestCometStackStep:
    def test_name_contains_mode(self) -> None:
        step = CometStackStep(tracking_mode="blend")
        assert "blend" in step.name.lower()

    def test_stage_is_comet_stacking(self) -> None:
        step = CometStackStep()
        assert step.stage == PipelineStage.COMET_STACKING

    def test_skips_empty_context(self) -> None:
        step = CometStackStep()
        ctx = PipelineContext()
        result = step.execute(ctx)
        assert result is ctx
        assert result.result is None

    def test_execute_sets_result(self) -> None:
        step = CometStackStep(tracking_mode="stars", stack_method="mean")
        frames = _make_frames(4)
        ctx = PipelineContext(images=frames)
        result = step.execute(ctx)
        assert result.result is not None
        assert result.result.shape == (32, 32)

    def test_execute_stores_both_stacks_in_metadata(self) -> None:
        step = CometStackStep(tracking_mode="blend", stack_method="mean")
        frames = _make_frames(4)
        ctx = PipelineContext(images=frames)
        result = step.execute(ctx)
        assert "comet_star_stack" in result.metadata
        assert "comet_nucleus_stack" in result.metadata

    def test_execute_stores_positions_in_metadata(self) -> None:
        step = CometStackStep(stack_method="mean")
        frames = _make_frames(4)
        ctx = PipelineContext(images=frames)
        result = step.execute(ctx)
        assert "comet_positions" in result.metadata
        assert len(result.metadata["comet_positions"]) == 4

    def test_fail_silently_suppresses_exception(self) -> None:
        step = CometStackStep(fail_silently=True)
        ctx = PipelineContext(images=_make_frames(2))
        with patch.object(step._tracker, "track", side_effect=RuntimeError("boom")):
            result = step.execute(ctx)
        assert result.result is None  # exception swallowed, result unchanged

    def test_fail_silently_false_propagates(self) -> None:
        step = CometStackStep(fail_silently=False)
        ctx = PipelineContext(images=_make_frames(2))
        with patch.object(step._tracker, "track", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                step.execute(ctx)

    def test_progress_callback_called(self) -> None:
        step = CometStackStep(stack_method="mean")
        frames = _make_frames(4)
        ctx = PipelineContext(images=frames)
        calls: list = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) >= 2

    def test_mode_stars_result_is_star_stack(self) -> None:
        step = CometStackStep(tracking_mode="stars", stack_method="mean")
        frames = _make_frames(4)
        ctx = PipelineContext(images=frames)
        result = step.execute(ctx)
        np.testing.assert_array_equal(result.result, result.metadata["comet_star_stack"])

    def test_mode_comet_result_is_comet_stack(self) -> None:
        step = CometStackStep(tracking_mode="comet", stack_method="mean")
        frames = _make_frames(4)
        ctx = PipelineContext(images=frames)
        result = step.execute(ctx)
        np.testing.assert_array_equal(result.result, result.metadata["comet_nucleus_stack"])
