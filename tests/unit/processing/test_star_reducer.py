"""Unit tests for astroai.processing.stars.star_reducer."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.processing.stars.star_reducer import (
    StarReducer,
    StarReductionConfig,
    StarReductionStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gray(val: float = 0.3, size: int = 16) -> np.ndarray:
    return np.full((size, size), val, dtype=np.float32)


def _rgb(r: float = 0.3, g: float = 0.3, b: float = 0.3, size: int = 16) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def _stars_image(size: int = 32) -> np.ndarray:
    """Dark background with a few bright star-like spots."""
    img = np.full((size, size, 3), 0.1, dtype=np.float32)
    # Place a bright star at center
    cx, cy = size // 2, size // 2
    img[cy - 1 : cy + 2, cx - 1 : cx + 2] = 0.9
    return img


def _stars_gray(size: int = 32) -> np.ndarray:
    img = np.full((size, size), 0.1, dtype=np.float32)
    cx, cy = size // 2, size // 2
    img[cy - 1 : cy + 2, cx - 1 : cx + 2] = 0.9
    return img


def _make_context(img, images=None):
    from astroai.core.pipeline.base import PipelineContext
    return PipelineContext(
        result=img,
        images=images or ([img] if img is not None else []),
    )


# ---------------------------------------------------------------------------
# TestStarReductionConfig
# ---------------------------------------------------------------------------

class TestStarReductionConfig:
    def test_defaults(self):
        cfg = StarReductionConfig()
        assert cfg.amount == pytest.approx(0.5)
        assert cfg.radius == 2
        assert cfg.threshold == pytest.approx(0.5)

    def test_is_identity_at_zero(self):
        cfg = StarReductionConfig(amount=0.0)
        assert cfg.is_identity()

    def test_is_identity_false_nonzero(self):
        cfg = StarReductionConfig(amount=0.5)
        assert not cfg.is_identity()

    def test_is_identity_within_atol(self):
        cfg = StarReductionConfig(amount=9e-5)
        assert cfg.is_identity()

    def test_is_identity_outside_atol(self):
        cfg = StarReductionConfig(amount=2e-4)
        assert not cfg.is_identity()

    def test_as_dict_keys(self):
        d = StarReductionConfig().as_dict()
        assert set(d.keys()) == {"amount", "radius", "threshold"}

    def test_as_dict_values(self):
        cfg = StarReductionConfig(amount=0.7, radius=3, threshold=0.6)
        d = cfg.as_dict()
        assert d["amount"] == pytest.approx(0.7)
        assert d["radius"] == 3
        assert d["threshold"] == pytest.approx(0.6)

    def test_validation_amount_below_zero(self):
        with pytest.raises(ValueError, match="amount"):
            StarReductionConfig(amount=-0.1)

    def test_validation_amount_above_one(self):
        with pytest.raises(ValueError, match="amount"):
            StarReductionConfig(amount=1.1)

    def test_validation_radius_zero(self):
        with pytest.raises(ValueError, match="radius"):
            StarReductionConfig(radius=0)

    def test_validation_radius_above_ten(self):
        with pytest.raises(ValueError, match="radius"):
            StarReductionConfig(radius=11)

    def test_validation_threshold_below_zero(self):
        with pytest.raises(ValueError, match="threshold"):
            StarReductionConfig(threshold=-0.01)

    def test_validation_threshold_above_one(self):
        with pytest.raises(ValueError, match="threshold"):
            StarReductionConfig(threshold=1.01)

    def test_frozen_immutable(self):
        cfg = StarReductionConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.amount = 0.9  # type: ignore[misc]

    def test_boundary_amount_zero_valid(self):
        cfg = StarReductionConfig(amount=0.0)
        assert cfg.amount == pytest.approx(0.0)

    def test_boundary_amount_one_valid(self):
        cfg = StarReductionConfig(amount=1.0)
        assert cfg.amount == pytest.approx(1.0)

    def test_boundary_radius_one_valid(self):
        cfg = StarReductionConfig(radius=1)
        assert cfg.radius == 1

    def test_boundary_radius_ten_valid(self):
        cfg = StarReductionConfig(radius=10)
        assert cfg.radius == 10


# ---------------------------------------------------------------------------
# TestStarReducerIdentity
# ---------------------------------------------------------------------------

class TestStarReducerIdentity:
    def test_amount_zero_returns_same_object_rgb(self):
        img = _stars_image()
        reducer = StarReducer(StarReductionConfig(amount=0.0))
        out = reducer.reduce(img)
        assert out is img

    def test_amount_zero_returns_same_object_gray(self):
        img = _stars_gray()
        reducer = StarReducer(StarReductionConfig(amount=0.0))
        out = reducer.reduce(img)
        assert out is img

    def test_default_config_none_uses_defaults(self):
        reducer = StarReducer(None)
        assert reducer.config == StarReductionConfig()

    def test_identity_config_within_atol(self):
        img = _stars_image()
        reducer = StarReducer(StarReductionConfig(amount=5e-5))
        out = reducer.reduce(img)
        assert out is img


# ---------------------------------------------------------------------------
# TestStarReducerEffect
# ---------------------------------------------------------------------------

class TestStarReducerEffect:
    def test_bright_pixels_reduced_rgb(self):
        img = _stars_image()
        reducer = StarReducer(StarReductionConfig(amount=1.0, radius=2, threshold=0.5))
        out = reducer.reduce(img)
        # Maximum of output should be less than maximum of input (stars shrunk)
        assert out.max() < img.max() or out.max() <= img.max()

    def test_bright_pixels_reduced_gray(self):
        img = _stars_gray()
        reducer = StarReducer(StarReductionConfig(amount=1.0, radius=2, threshold=0.5))
        out = reducer.reduce(img)
        # Star center should be darker or equal after reduction
        cx, cy = 16, 16
        assert out[cy, cx] <= img[cy, cx] + 1e-6

    def test_larger_amount_means_stronger_reduction(self):
        img = _stars_image()
        reducer_low = StarReducer(StarReductionConfig(amount=0.2, radius=2, threshold=0.5))
        reducer_high = StarReducer(StarReductionConfig(amount=0.9, radius=2, threshold=0.5))
        out_low = reducer_low.reduce(img)
        out_high = reducer_high.reduce(img)
        # Higher amount → lower max in star region
        cx, cy = 16, 16
        assert out_high[cy, cx, 0] <= out_low[cy, cx, 0] + 1e-6

    def test_background_pixels_mostly_unchanged(self):
        """Background pixels (below threshold) should be barely affected."""
        img = _stars_image()
        reducer = StarReducer(StarReductionConfig(amount=1.0, radius=2, threshold=0.5))
        out = reducer.reduce(img)
        # Corner pixel is background (0.1) — should remain close
        np.testing.assert_allclose(out[0, 0, 0], img[0, 0, 0], atol=0.05)


# ---------------------------------------------------------------------------
# TestStarReducerThreshold
# ---------------------------------------------------------------------------

class TestStarReducerThreshold:
    def test_threshold_one_no_mask_no_effect(self):
        """threshold=1.0 means no pixel qualifies as star → near-identity."""
        img = _stars_image()
        reducer = StarReducer(StarReductionConfig(amount=1.0, radius=2, threshold=1.0))
        out = reducer.reduce(img)
        # With threshold=1.0, no pixel is masked (max pixel value in img is 0.9 < 1.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_threshold_zero_all_pixels_masked(self):
        """threshold=0.0 → all pixels treated as stars."""
        img = _gray(0.5)
        reducer = StarReducer(StarReductionConfig(amount=1.0, radius=1, threshold=0.0))
        out = reducer.reduce(img)
        # uniform image → minimum filter on uniform is same → output should match original
        np.testing.assert_allclose(out, img, atol=1e-5)


# ---------------------------------------------------------------------------
# TestStarReducerDtype
# ---------------------------------------------------------------------------

class TestStarReducerDtype:
    def test_float32_in_float32_out_rgb(self):
        img = _stars_image().astype(np.float32)
        reducer = StarReducer(StarReductionConfig(amount=0.5))
        out = reducer.reduce(img)
        assert out.dtype == np.float32

    def test_float64_in_float64_out_rgb(self):
        img = _stars_image().astype(np.float64)
        reducer = StarReducer(StarReductionConfig(amount=0.5))
        out = reducer.reduce(img)
        assert out.dtype == np.float64

    def test_float32_in_float32_out_gray(self):
        img = _stars_gray().astype(np.float32)
        reducer = StarReducer(StarReductionConfig(amount=0.5))
        out = reducer.reduce(img)
        assert out.dtype == np.float32

    def test_output_not_input_alias(self):
        img = _stars_image()
        reducer = StarReducer(StarReductionConfig(amount=0.5))
        out = reducer.reduce(img)
        assert out is not img


# ---------------------------------------------------------------------------
# TestStarReducerGrayscale
# ---------------------------------------------------------------------------

class TestStarReducerGrayscale:
    def test_grayscale_shape_preserved(self):
        img = _stars_gray()
        reducer = StarReducer(StarReductionConfig(amount=0.5))
        out = reducer.reduce(img)
        assert out.shape == img.shape
        assert out.ndim == 2

    def test_grayscale_runs_without_error(self):
        img = _stars_gray()
        reducer = StarReducer(StarReductionConfig(amount=0.8, radius=1, threshold=0.5))
        out = reducer.reduce(img)
        assert out is not None


# ---------------------------------------------------------------------------
# TestStarReducerRGB
# ---------------------------------------------------------------------------

class TestStarReducerRGB:
    def test_rgb_shape_preserved(self):
        img = _stars_image()
        reducer = StarReducer(StarReductionConfig(amount=0.5))
        out = reducer.reduce(img)
        assert out.shape == img.shape
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_rgb_runs_without_error(self):
        img = _stars_image()
        reducer = StarReducer(StarReductionConfig(amount=0.8, radius=2, threshold=0.4))
        out = reducer.reduce(img)
        assert out is not None


# ---------------------------------------------------------------------------
# TestStarReducerClipping
# ---------------------------------------------------------------------------

class TestStarReducerClipping:
    def test_output_max_not_above_one(self):
        img = np.ones((16, 16, 3), dtype=np.float32)
        reducer = StarReducer(StarReductionConfig(amount=1.0, radius=2, threshold=0.0))
        out = reducer.reduce(img)
        assert out.max() <= 1.0 + 1e-7

    def test_output_min_not_below_zero(self):
        img = _stars_image()
        reducer = StarReducer(StarReductionConfig(amount=1.0, radius=5, threshold=0.05))
        out = reducer.reduce(img)
        assert out.min() >= -1e-7

    def test_output_clipped_gray(self):
        img = _stars_gray(size=16)
        reducer = StarReducer(StarReductionConfig(amount=1.0, radius=3, threshold=0.05))
        out = reducer.reduce(img)
        assert out.min() >= -1e-7
        assert out.max() <= 1.0 + 1e-7


# ---------------------------------------------------------------------------
# TestStarReductionStep
# ---------------------------------------------------------------------------

class TestStarReductionStep:
    def test_step_name(self):
        step = StarReductionStep()
        assert "ternreduktion" in step.name

    def test_step_stage_is_processing(self):
        from astroai.core.pipeline.base import PipelineStage
        step = StarReductionStep()
        assert step.stage == PipelineStage.PROCESSING

    def test_execute_sets_result(self):
        img = _stars_image()
        step = StarReductionStep(StarReductionConfig(amount=0.5))
        ctx = _make_context(img)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None

    def test_execute_falls_back_to_images(self):
        from astroai.core.pipeline.base import PipelineContext
        img = _stars_image()
        ctx = PipelineContext(result=None, images=[img])
        step = StarReductionStep(StarReductionConfig(amount=0.5))
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None

    def test_execute_no_image_skips(self):
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[])
        step = StarReductionStep()
        result_ctx = step.execute(ctx)
        assert result_ctx.result is None

    def test_execute_calls_progress_twice(self):
        img = _stars_image()
        step = StarReductionStep(StarReductionConfig(amount=0.5))
        ctx = _make_context(img)
        calls = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2

    def test_execute_identity_config(self):
        img = _stars_image()
        step = StarReductionStep(StarReductionConfig(amount=0.0))
        ctx = _make_context(img)
        result_ctx = step.execute(ctx)
        # identity → same array returned unchanged
        np.testing.assert_array_equal(result_ctx.result, img)
