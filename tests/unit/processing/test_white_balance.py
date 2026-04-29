"""Unit tests for astroai.processing.color.white_balance."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.processing.color.white_balance import (
    WhiteBalanceAdjustment,
    WhiteBalanceConfig,
    WhiteBalanceStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_image(r: float = 0.5, g: float = 0.5, b: float = 0.5, size: int = 4) -> np.ndarray:
    """Create a solid RGB image with given channel values."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def _gray_image(size: int = 4) -> np.ndarray:
    return np.full((size, size), 0.5, dtype=np.float32)


def _gray1_image(size: int = 4) -> np.ndarray:
    return np.full((size, size, 1), 0.5, dtype=np.float32)


# ---------------------------------------------------------------------------
# TestWhiteBalanceConfig
# ---------------------------------------------------------------------------

class TestWhiteBalanceConfig:
    def test_defaults(self):
        cfg = WhiteBalanceConfig()
        assert cfg.red_factor == pytest.approx(1.0)
        assert cfg.green_factor == pytest.approx(1.0)
        assert cfg.blue_factor == pytest.approx(1.0)

    def test_is_identity_defaults(self):
        assert WhiteBalanceConfig().is_identity()

    def test_is_identity_false_when_red_differs(self):
        assert not WhiteBalanceConfig(red_factor=1.1).is_identity()

    def test_is_identity_false_when_green_differs(self):
        assert not WhiteBalanceConfig(green_factor=0.9).is_identity()

    def test_is_identity_false_when_blue_differs(self):
        assert not WhiteBalanceConfig(blue_factor=2.0).is_identity()

    def test_is_identity_within_atol(self):
        cfg = WhiteBalanceConfig(red_factor=1.0 + 9e-5)
        assert cfg.is_identity()

    def test_is_identity_outside_atol(self):
        cfg = WhiteBalanceConfig(red_factor=1.0 + 2e-4)
        assert not cfg.is_identity()

    def test_as_dict_keys(self):
        d = WhiteBalanceConfig().as_dict()
        assert set(d.keys()) == {"red_factor", "green_factor", "blue_factor"}

    def test_as_dict_values(self):
        cfg = WhiteBalanceConfig(red_factor=1.5, green_factor=0.8, blue_factor=2.0)
        d = cfg.as_dict()
        assert d["red_factor"] == pytest.approx(1.5)
        assert d["green_factor"] == pytest.approx(0.8)
        assert d["blue_factor"] == pytest.approx(2.0)

    def test_validation_red_zero_raises(self):
        with pytest.raises(ValueError, match="red_factor"):
            WhiteBalanceConfig(red_factor=0.0)

    def test_validation_red_negative_raises(self):
        with pytest.raises(ValueError):
            WhiteBalanceConfig(red_factor=-1.0)

    def test_validation_green_zero_raises(self):
        with pytest.raises(ValueError, match="green_factor"):
            WhiteBalanceConfig(green_factor=0.0)

    def test_validation_blue_zero_raises(self):
        with pytest.raises(ValueError, match="blue_factor"):
            WhiteBalanceConfig(blue_factor=0.0)

    def test_frozen_immutable(self):
        cfg = WhiteBalanceConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.red_factor = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestWhiteBalanceAdjustmentConstruction
# ---------------------------------------------------------------------------

class TestWhiteBalanceAdjustmentConstruction:
    def test_default_config_is_identity(self):
        adj = WhiteBalanceAdjustment()
        assert adj.config.is_identity()

    def test_custom_config_stored(self):
        cfg = WhiteBalanceConfig(red_factor=1.5)
        adj = WhiteBalanceAdjustment(cfg)
        assert adj.config.red_factor == pytest.approx(1.5)

    def test_none_config_uses_defaults(self):
        adj = WhiteBalanceAdjustment(None)
        assert adj.config == WhiteBalanceConfig()


# ---------------------------------------------------------------------------
# TestApplyIdentity
# ---------------------------------------------------------------------------

class TestApplyIdentity:
    def test_identity_returns_same_values(self):
        img = _rgb_image(0.3, 0.5, 0.7)
        adj = WhiteBalanceAdjustment()
        out = adj.apply(img)
        np.testing.assert_array_equal(out, img)

    def test_identity_returns_same_object(self):
        img = _rgb_image(0.3, 0.5, 0.7)
        adj = WhiteBalanceAdjustment()
        out = adj.apply(img)
        assert out is img

    def test_identity_atol_boundary(self):
        cfg = WhiteBalanceConfig(red_factor=1.0 + 9e-5, green_factor=1.0, blue_factor=1.0)
        img = _rgb_image(0.4, 0.4, 0.4)
        adj = WhiteBalanceAdjustment(cfg)
        out = adj.apply(img)
        assert out is img


# ---------------------------------------------------------------------------
# TestApplyChannels
# ---------------------------------------------------------------------------

class TestApplyChannels:
    def test_red_only_multiplier(self):
        img = _rgb_image(0.4, 0.5, 0.6)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(red_factor=2.0))
        out = adj.apply(img)
        np.testing.assert_allclose(out[:, :, 0], np.clip(0.4 * 2.0, 0, 1), atol=1e-6)
        np.testing.assert_allclose(out[:, :, 1], 0.5, atol=1e-6)
        np.testing.assert_allclose(out[:, :, 2], 0.6, atol=1e-6)

    def test_green_only_multiplier(self):
        img = _rgb_image(0.4, 0.5, 0.6)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(green_factor=1.5))
        out = adj.apply(img)
        np.testing.assert_allclose(out[:, :, 0], 0.4, atol=1e-6)
        np.testing.assert_allclose(out[:, :, 1], np.clip(0.5 * 1.5, 0, 1), atol=1e-6)
        np.testing.assert_allclose(out[:, :, 2], 0.6, atol=1e-6)

    def test_blue_only_multiplier(self):
        img = _rgb_image(0.4, 0.5, 0.6)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(blue_factor=0.5))
        out = adj.apply(img)
        np.testing.assert_allclose(out[:, :, 0], 0.4, atol=1e-6)
        np.testing.assert_allclose(out[:, :, 1], 0.5, atol=1e-6)
        np.testing.assert_allclose(out[:, :, 2], 0.3, atol=1e-6)

    def test_all_channels_multiplied(self):
        img = _rgb_image(0.2, 0.4, 0.6)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(red_factor=1.5, green_factor=0.5, blue_factor=2.0))
        out = adj.apply(img)
        np.testing.assert_allclose(out[:, :, 0], 0.3, atol=1e-6)
        np.testing.assert_allclose(out[:, :, 1], 0.2, atol=1e-6)
        np.testing.assert_allclose(out[:, :, 2], 1.0, atol=1e-6)  # clipped


# ---------------------------------------------------------------------------
# TestApplyGrayscale
# ---------------------------------------------------------------------------

class TestApplyGrayscale:
    def test_2d_grayscale_returned_unchanged(self):
        img = _gray_image()
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(red_factor=2.0))
        out = adj.apply(img)
        assert out is img

    def test_hwc1_grayscale_returned_unchanged(self):
        img = _gray1_image()
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(blue_factor=3.0))
        out = adj.apply(img)
        assert out is img

    def test_2d_grayscale_values_preserved(self):
        img = _gray_image()
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(green_factor=0.1))
        out = adj.apply(img)
        np.testing.assert_array_equal(out, img)


# ---------------------------------------------------------------------------
# TestApplyClipping
# ---------------------------------------------------------------------------

class TestApplyClipping:
    def test_clipping_red_exceeds_1(self):
        img = _rgb_image(0.8, 0.5, 0.5)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(red_factor=2.0))
        out = adj.apply(img)
        assert out[:, :, 0].max() <= 1.0

    def test_clipping_blue_exceeds_1(self):
        img = _rgb_image(0.5, 0.5, 0.9)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(blue_factor=5.0))
        out = adj.apply(img)
        assert out[:, :, 2].max() <= 1.0 + 1e-7

    def test_output_never_below_zero(self):
        img = _rgb_image(0.1, 0.1, 0.1)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(red_factor=0.1, green_factor=0.1, blue_factor=0.1))
        out = adj.apply(img)
        assert out.min() >= 0.0

    def test_bright_image_large_factor_clips(self):
        img = np.ones((4, 4, 3), dtype=np.float32)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(red_factor=5.0, green_factor=5.0, blue_factor=5.0))
        out = adj.apply(img)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# TestApplyDtype
# ---------------------------------------------------------------------------

class TestApplyDtype:
    def test_float32_input_float32_output(self):
        img = _rgb_image(0.3, 0.4, 0.5).astype(np.float32)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(red_factor=1.5))
        out = adj.apply(img)
        assert out.dtype == np.float32

    def test_float64_input_float64_output(self):
        img = _rgb_image(0.3, 0.4, 0.5).astype(np.float64)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(red_factor=1.5))
        out = adj.apply(img)
        assert out.dtype == np.float64

    def test_output_not_input_alias(self):
        img = _rgb_image(0.3, 0.4, 0.5)
        adj = WhiteBalanceAdjustment(WhiteBalanceConfig(red_factor=1.5))
        out = adj.apply(img)
        assert out is not img


# ---------------------------------------------------------------------------
# TestWhiteBalanceStep
# ---------------------------------------------------------------------------

class TestWhiteBalanceStep:
    def _make_context(self, img, images=None):
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(
            result=img,
            images=images or ([img] if img is not None else []),
        )
        return ctx

    def test_step_name(self):
        step = WhiteBalanceStep()
        assert "eißabgleich" in step.name

    def test_step_stage_is_processing(self):
        from astroai.core.pipeline.base import PipelineStage
        step = WhiteBalanceStep()
        assert step.stage == PipelineStage.PROCESSING

    def test_execute_with_context_result(self):
        img = _rgb_image(0.4, 0.5, 0.6)
        cfg = WhiteBalanceConfig(red_factor=2.0)
        step = WhiteBalanceStep(cfg)
        ctx = self._make_context(img)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        np.testing.assert_allclose(result_ctx.result[:, :, 0], np.clip(0.4 * 2.0, 0, 1), atol=1e-6)

    def test_execute_falls_back_to_images(self):
        img = _rgb_image(0.3, 0.3, 0.3)
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[img])
        cfg = WhiteBalanceConfig(blue_factor=2.0)
        step = WhiteBalanceStep(cfg)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        np.testing.assert_allclose(result_ctx.result[:, :, 2], np.clip(0.3 * 2.0, 0, 1), atol=1e-6)

    def test_execute_no_image_skips(self):
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[])
        step = WhiteBalanceStep()
        result_ctx = step.execute(ctx)
        assert result_ctx.result is None

    def test_execute_sets_context_result(self):
        img = _rgb_image(0.2, 0.3, 0.4)
        step = WhiteBalanceStep()
        ctx = self._make_context(img)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None

    def test_execute_calls_progress(self):
        img = _rgb_image(0.2, 0.3, 0.4)
        cfg = WhiteBalanceConfig(red_factor=1.5)
        step = WhiteBalanceStep(cfg)
        ctx = self._make_context(img)
        calls = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2

    def test_execute_identity_config(self):
        img = _rgb_image(0.2, 0.3, 0.4)
        step = WhiteBalanceStep()
        ctx = self._make_context(img)
        result_ctx = step.execute(ctx)
        # identity returns same values
        np.testing.assert_array_equal(result_ctx.result, img)

    def test_default_step_config_is_identity(self):
        step = WhiteBalanceStep()
        assert step._adj.config.is_identity()
