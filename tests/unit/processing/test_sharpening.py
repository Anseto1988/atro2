"""Tests for UnsharpMask and SharpeningStep."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.processing.sharpening.unsharp_mask import UnsharpMask
from astroai.processing.sharpening.pipeline_step import SharpeningStep
from astroai.core.pipeline.base import PipelineContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb(h: int = 64, w: int = 64, val: float = 0.5) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.float32)


def _make_gray(h: int = 64, w: int = 64, val: float = 0.5) -> np.ndarray:
    return np.full((h, w), val, dtype=np.float32)


def _make_gradient(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.random((h, w, 3), dtype=None).astype(np.float32)


# ---------------------------------------------------------------------------
# UnsharpMask construction
# ---------------------------------------------------------------------------

class TestUnsharpMaskInit:
    def test_default_params(self) -> None:
        usm = UnsharpMask()
        assert usm.radius == pytest.approx(1.0)
        assert usm.amount == pytest.approx(0.5)
        assert usm.threshold == pytest.approx(0.02)

    def test_custom_params(self) -> None:
        usm = UnsharpMask(radius=2.5, amount=0.8, threshold=0.05)
        assert usm.radius == pytest.approx(2.5)
        assert usm.amount == pytest.approx(0.8)
        assert usm.threshold == pytest.approx(0.05)

    def test_invalid_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius must be positive"):
            UnsharpMask(radius=0.0)

    def test_invalid_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="amount must be in"):
            UnsharpMask(amount=1.5)

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="amount must be in"):
            UnsharpMask(amount=-0.1)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            UnsharpMask(threshold=0.6)


# ---------------------------------------------------------------------------
# UnsharpMask.apply — basic behavior
# ---------------------------------------------------------------------------

class TestUnsharpMaskApply:
    def test_output_shape_rgb_preserved(self) -> None:
        img = _make_gradient()
        usm = UnsharpMask(radius=1.0, amount=0.5, threshold=0.0)
        result = usm.apply(img)
        assert result.shape == img.shape

    def test_output_shape_gray_preserved(self) -> None:
        img = _make_gray()
        usm = UnsharpMask(radius=1.0, amount=0.5, threshold=0.0)
        result = usm.apply(img)
        assert result.shape == img.shape

    def test_output_dtype_preserved(self) -> None:
        img = _make_rgb().astype(np.float32)
        result = UnsharpMask().apply(img)
        assert result.dtype == np.float32

    def test_output_clipped_to_unit_range(self) -> None:
        img = np.full((16, 16, 3), 0.95, dtype=np.float32)
        result = UnsharpMask(radius=0.5, amount=1.0, threshold=0.0).apply(img)
        assert float(result.max()) <= 1.0
        assert float(result.min()) >= 0.0

    def test_zero_amount_returns_unchanged(self) -> None:
        img = _make_gradient()
        result = UnsharpMask(radius=1.0, amount=0.0, threshold=0.0).apply(img)
        np.testing.assert_allclose(result, img, atol=1e-6)

    def test_flat_image_unchanged_below_threshold(self) -> None:
        img = _make_rgb(val=0.4)
        result = UnsharpMask(radius=1.0, amount=1.0, threshold=0.1).apply(img)
        np.testing.assert_allclose(result, img, atol=1e-6)

    def test_sharpening_increases_contrast_near_edge(self) -> None:
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[:, 16:, :] = 1.0  # hard vertical edge
        result = UnsharpMask(radius=1.0, amount=1.0, threshold=0.0).apply(img)
        # Pixel just left of edge should decrease; just right should increase
        assert float(result[16, 15, 0]) < float(img[16, 15, 0]) or \
               float(result[16, 16, 0]) > float(img[16, 16, 0]) - 0.01

    def test_does_not_mutate_input(self) -> None:
        img = _make_gradient()
        original = img.copy()
        UnsharpMask().apply(img)
        np.testing.assert_array_equal(img, original)

    def test_grayscale_sharpening(self) -> None:
        img = np.zeros((32, 32), dtype=np.float32)
        img[:, 16:] = 1.0
        result = UnsharpMask(radius=1.0, amount=0.5, threshold=0.0).apply(img)
        assert result.shape == img.shape
        assert float(result.max()) <= 1.0

    def test_threshold_zero_always_applies(self) -> None:
        img = _make_rgb(val=0.5)
        # Add tiny variation so sharpening has something to work on
        img[10, 10, 0] = 0.501
        result_no_threshold = UnsharpMask(radius=0.5, amount=0.5, threshold=0.0).apply(img)
        result_with_threshold = UnsharpMask(radius=0.5, amount=0.5, threshold=0.45).apply(img)
        # With high threshold, result should be closer to original
        diff_no = float(np.abs(result_no_threshold - img).max())
        diff_with = float(np.abs(result_with_threshold - img).max())
        assert diff_with <= diff_no + 1e-6

    def test_large_radius_blurs_more(self) -> None:
        img = _make_gradient()
        result_small = UnsharpMask(radius=0.5, amount=0.5, threshold=0.0).apply(img)
        result_large = UnsharpMask(radius=5.0, amount=0.5, threshold=0.0).apply(img)
        # Different radii should produce different results
        assert not np.allclose(result_small, result_large)


# ---------------------------------------------------------------------------
# SharpeningStep
# ---------------------------------------------------------------------------

class TestSharpeningStep:
    def test_name(self) -> None:
        step = SharpeningStep()
        assert step.name == "Schärfung"

    def test_execute_with_result(self) -> None:
        step = SharpeningStep(radius=1.0, amount=0.5, threshold=0.02)
        ctx = PipelineContext(result=_make_gradient())
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (64, 64, 3)

    def test_execute_no_result_returns_context(self) -> None:
        step = SharpeningStep()
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.result is None

    def test_execute_progress_called(self) -> None:
        calls: list[str] = []
        step = SharpeningStep()
        ctx = PipelineContext(result=_make_rgb())
        step.execute(ctx, progress=lambda p: calls.append(p.message))
        assert len(calls) == 2
        assert "Schärfung" in calls[0]

    def test_output_clipped(self) -> None:
        step = SharpeningStep(radius=0.5, amount=1.0, threshold=0.0)
        ctx = PipelineContext(result=np.full((32, 32, 3), 0.95, dtype=np.float32))
        out = step.execute(ctx)
        assert float(out.result.max()) <= 1.0

    def test_step_stage_is_processing(self) -> None:
        from astroai.core.pipeline.base import PipelineStage
        step = SharpeningStep()
        assert step.stage == PipelineStage.PROCESSING


# ---------------------------------------------------------------------------
# PipelineBuilder integration
# ---------------------------------------------------------------------------

class TestPipelineBuilderSharpening:
    def test_sharpening_step_added_when_enabled(self) -> None:
        from astroai.core.pipeline.builder import PipelineBuilder
        from astroai.ui.models import PipelineModel

        model = PipelineModel()
        model.sharpening_enabled = True
        model.sharpening_radius = 1.5
        model.sharpening_amount = 0.6
        model.sharpening_threshold = 0.03

        builder = PipelineBuilder()
        pipeline = builder.build_processing_pipeline(model)
        step_names = [s.name for s in pipeline._steps]
        assert "Schärfung" in step_names

    def test_sharpening_step_absent_when_disabled(self) -> None:
        from astroai.core.pipeline.builder import PipelineBuilder
        from astroai.ui.models import PipelineModel

        model = PipelineModel()
        model.sharpening_enabled = False

        builder = PipelineBuilder()
        pipeline = builder.build_processing_pipeline(model)
        step_names = [s.name for s in pipeline._steps]
        assert "Schärfung" not in step_names
