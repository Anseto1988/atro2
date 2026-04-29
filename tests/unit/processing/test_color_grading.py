"""Tests for ColorGrading — shadow/midtone/highlight color correction."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.processing.color.color_grading import (
    ColorGradingConfig,
    ColorGrader,
    ColorGradingStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb(h: int = 32, w: int = 32, value: float = 0.5) -> np.ndarray:
    """Uniform RGB image filled with *value*."""
    return np.full((h, w, 3), value, dtype=np.float32)


def _dark_rgb(h: int = 32, w: int = 32) -> np.ndarray:
    """Very dark image (0.05) — strong shadow mask."""
    return np.full((h, w, 3), 0.05, dtype=np.float32)


def _bright_rgb(h: int = 32, w: int = 32) -> np.ndarray:
    """Very bright image (0.95) — strong highlight mask."""
    return np.full((h, w, 3), 0.95, dtype=np.float32)


def _gradient_rgb(h: int = 32, w: int = 32) -> np.ndarray:
    """RGB image with luminance gradient from 0 to 1 across columns."""
    arr = np.zeros((h, w, 3), dtype=np.float64)
    gradient = np.linspace(0.0, 1.0, w)
    arr[:, :, 0] = gradient
    arr[:, :, 1] = gradient
    arr[:, :, 2] = gradient
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# TestColorGradingConfig
# ---------------------------------------------------------------------------

class TestColorGradingConfig:
    def test_default_fields(self):
        cfg = ColorGradingConfig()
        assert cfg.shadow_r == 0.0
        assert cfg.shadow_g == 0.0
        assert cfg.shadow_b == 0.0
        assert cfg.midtone_r == 0.0
        assert cfg.midtone_g == 0.0
        assert cfg.midtone_b == 0.0
        assert cfg.highlight_r == 0.0
        assert cfg.highlight_g == 0.0
        assert cfg.highlight_b == 0.0

    def test_custom_fields(self):
        cfg = ColorGradingConfig(shadow_r=0.1, midtone_g=-0.2, highlight_b=0.5)
        assert cfg.shadow_r == 0.1
        assert cfg.midtone_g == -0.2
        assert cfg.highlight_b == 0.5

    def test_is_identity_default(self):
        assert ColorGradingConfig().is_identity() is True

    def test_is_identity_nonzero(self):
        cfg = ColorGradingConfig(shadow_r=0.01)
        assert cfg.is_identity() is False

    def test_is_identity_near_zero(self):
        cfg = ColorGradingConfig(shadow_r=1e-6)
        assert cfg.is_identity() is True

    def test_is_identity_exact_atol(self):
        # 1e-5 exactly is the boundary — should still pass (< 1e-5 is False for ==)
        cfg = ColorGradingConfig(shadow_r=1e-5)
        assert cfg.is_identity() is False

    def test_as_dict_keys(self):
        cfg = ColorGradingConfig()
        d = cfg.as_dict()
        expected_keys = {
            "shadow_r", "shadow_g", "shadow_b",
            "midtone_r", "midtone_g", "midtone_b",
            "highlight_r", "highlight_g", "highlight_b",
        }
        assert set(d.keys()) == expected_keys

    def test_as_dict_values(self):
        cfg = ColorGradingConfig(shadow_r=0.1, highlight_b=-0.3)
        d = cfg.as_dict()
        assert d["shadow_r"] == pytest.approx(0.1)
        assert d["highlight_b"] == pytest.approx(-0.3)
        assert d["shadow_g"] == pytest.approx(0.0)

    def test_validation_max_shadow_r(self):
        with pytest.raises(ValueError, match="shadow_r"):
            ColorGradingConfig(shadow_r=0.51)

    def test_validation_min_shadow_r(self):
        with pytest.raises(ValueError, match="shadow_r"):
            ColorGradingConfig(shadow_r=-0.51)

    def test_validation_highlight_b(self):
        with pytest.raises(ValueError, match="highlight_b"):
            ColorGradingConfig(highlight_b=1.0)

    def test_validation_midtone_g(self):
        with pytest.raises(ValueError, match="midtone_g"):
            ColorGradingConfig(midtone_g=-0.6)

    def test_boundary_values_ok(self):
        cfg = ColorGradingConfig(
            shadow_r=-0.5, shadow_g=0.5,
            midtone_r=-0.5, midtone_g=0.5,
            highlight_r=-0.5, highlight_b=0.5,
        )
        assert cfg.shadow_r == -0.5
        assert cfg.shadow_g == 0.5

    def test_frozen(self):
        cfg = ColorGradingConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.shadow_r = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestColorGraderIdentity
# ---------------------------------------------------------------------------

class TestColorGraderIdentity:
    def test_identity_returns_same_object(self):
        img = _rgb()
        grader = ColorGrader(ColorGradingConfig())
        result = grader.grade(img)
        assert result is img

    def test_identity_default_config(self):
        img = _rgb()
        grader = ColorGrader()
        result = grader.grade(img)
        assert result is img

    def test_identity_pixel_values_unchanged(self):
        img = _gradient_rgb()
        grader = ColorGrader(ColorGradingConfig())
        result = grader.grade(img)
        np.testing.assert_array_equal(result, img)


# ---------------------------------------------------------------------------
# TestColorGraderShadowsOnly
# ---------------------------------------------------------------------------

class TestColorGraderShadowsOnly:
    def test_shadow_r_brightens_dark_red(self):
        """shadow_r=0.2 should increase R channel in a very dark image."""
        img = _dark_rgb()
        grader = ColorGrader(ColorGradingConfig(shadow_r=0.2))
        result = grader.grade(img)
        assert result[:, :, 0].mean() > img[:, :, 0].mean()

    def test_shadow_r_barely_affects_bright_pixels(self):
        """shadow_r=0.2 should have minimal effect on very bright pixels."""
        img = _bright_rgb()
        grader = ColorGrader(ColorGradingConfig(shadow_r=0.2))
        result = grader.grade(img)
        # Bright pixels have almost zero shadow mask → nearly no change
        assert abs(result[:, :, 0].mean() - img[:, :, 0].mean()) < 0.02

    def test_shadow_g_channel_separation(self):
        """shadow_g affects G channel but not R or B."""
        img = _dark_rgb()
        grader = ColorGrader(ColorGradingConfig(shadow_g=0.2))
        result = grader.grade(img)
        assert result[:, :, 1].mean() > img[:, :, 1].mean()
        np.testing.assert_array_almost_equal(result[:, :, 0], img[:, :, 0], decimal=6)
        np.testing.assert_array_almost_equal(result[:, :, 2], img[:, :, 2], decimal=6)

    def test_shadow_negative_shift_darkens(self):
        img = _dark_rgb()
        grader = ColorGrader(ColorGradingConfig(shadow_b=-0.04))
        result = grader.grade(img)
        assert result[:, :, 2].mean() < img[:, :, 2].mean()


# ---------------------------------------------------------------------------
# TestColorGraderHighlightsOnly
# ---------------------------------------------------------------------------

class TestColorGraderHighlightsOnly:
    def test_highlight_r_brightens_bright_red(self):
        img = _bright_rgb()
        grader = ColorGrader(ColorGradingConfig(highlight_r=0.02))
        result = grader.grade(img)
        # Bright image is near 1.0, so clipping may keep it at 1.0
        assert result[:, :, 0].mean() >= img[:, :, 0].mean() - 1e-6

    def test_highlight_r_barely_affects_dark_pixels(self):
        img = _dark_rgb()
        grader = ColorGrader(ColorGradingConfig(highlight_r=0.3))
        result = grader.grade(img)
        assert abs(result[:, :, 0].mean() - img[:, :, 0].mean()) < 0.05

    def test_highlight_b_channel_separation(self):
        img = _bright_rgb()
        grader = ColorGrader(ColorGradingConfig(highlight_b=-0.05))
        result = grader.grade(img)
        assert result[:, :, 2].mean() < img[:, :, 2].mean()
        np.testing.assert_array_almost_equal(result[:, :, 0], img[:, :, 0], decimal=5)

    def test_highlight_gradient_effect(self):
        """Bright end of gradient is more affected than dark end."""
        img = _gradient_rgb()
        grader = ColorGrader(ColorGradingConfig(highlight_r=0.2))
        result = grader.grade(img)
        # Right side (bright) gets more shift than left side (dark)
        assert result[0, -1, 0] >= result[0, 0, 0]


# ---------------------------------------------------------------------------
# TestColorGraderMidtonesOnly
# ---------------------------------------------------------------------------

class TestColorGraderMidtonesOnly:
    def test_midtone_r_affects_midgray(self):
        img = _rgb(value=0.5)
        grader = ColorGrader(ColorGradingConfig(midtone_r=0.1))
        result = grader.grade(img)
        assert result[:, :, 0].mean() > img[:, :, 0].mean()

    def test_midtone_g_channel_only(self):
        img = _rgb(value=0.5)
        grader = ColorGrader(ColorGradingConfig(midtone_g=0.1))
        result = grader.grade(img)
        assert result[:, :, 1].mean() > img[:, :, 1].mean()
        np.testing.assert_array_almost_equal(result[:, :, 0], img[:, :, 0], decimal=5)
        np.testing.assert_array_almost_equal(result[:, :, 2], img[:, :, 2], decimal=5)

    def test_midtone_mask_stronger_at_midgray_than_extremes_on_gradient(self):
        """On a gradient image the midtone mask peaks between dark and bright ends.

        We check this by splitting the gradient image into thirds and comparing
        the absolute shift applied to the center third vs. the far-dark/far-bright thirds.
        Uses float64 for precision.
        """
        # gradient: columns go from lum=0 to lum=1
        h, w = 16, 120
        arr = np.zeros((h, w, 3), dtype=np.float64)
        gradient = np.linspace(0.0, 1.0, w)
        arr[:, :, 0] = gradient
        arr[:, :, 1] = gradient
        arr[:, :, 2] = gradient

        cfg = ColorGradingConfig(midtone_r=0.3)
        result = ColorGrader(cfg).grade(arr)
        delta = result[:, :, 0] - arr[:, :, 0]  # H×W

        third = w // 3
        delta_dark = float(delta[:, :third].mean())
        delta_mid = float(delta[:, third:2*third].mean())
        delta_bright = float(delta[:, 2*third:].mean())

        # Center third (midtones) should receive more shift than either extreme
        assert delta_mid > delta_dark
        assert delta_mid > delta_bright


# ---------------------------------------------------------------------------
# TestColorGraderGrayscale
# ---------------------------------------------------------------------------

class TestColorGraderGrayscale:
    def test_2d_grayscale_unchanged(self):
        img = np.random.rand(32, 32).astype(np.float32)
        grader = ColorGrader(ColorGradingConfig(shadow_r=0.2))
        result = grader.grade(img)
        assert result is img

    def test_hw1_grayscale_unchanged(self):
        img = np.random.rand(32, 32, 1).astype(np.float32)
        grader = ColorGrader(ColorGradingConfig(highlight_g=0.1))
        result = grader.grade(img)
        assert result is img


# ---------------------------------------------------------------------------
# TestColorGraderClipping
# ---------------------------------------------------------------------------

class TestColorGraderClipping:
    def test_output_never_exceeds_1(self):
        img = _bright_rgb()
        grader = ColorGrader(ColorGradingConfig(highlight_r=0.5, highlight_g=0.5, highlight_b=0.5))
        result = grader.grade(img)
        assert result.max() <= 1.0

    def test_output_never_below_0(self):
        img = _dark_rgb()
        grader = ColorGrader(ColorGradingConfig(shadow_r=-0.5, shadow_g=-0.5, shadow_b=-0.5))
        result = grader.grade(img)
        assert result.min() >= 0.0

    def test_clipping_with_all_extreme_shifts(self):
        img = _gradient_rgb()
        cfg = ColorGradingConfig(
            shadow_r=0.5, midtone_r=0.5, highlight_r=0.5,
            shadow_b=-0.5, midtone_b=-0.5, highlight_b=-0.5,
        )
        result = ColorGrader(cfg).grade(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# TestColorGraderDtype
# ---------------------------------------------------------------------------

class TestColorGraderDtype:
    def test_float32_in_float32_out(self):
        img = _rgb().astype(np.float32)
        grader = ColorGrader(ColorGradingConfig(shadow_r=0.1))
        result = grader.grade(img)
        assert result.dtype == np.float32

    def test_float64_in_float64_out(self):
        img = _rgb().astype(np.float64)
        grader = ColorGrader(ColorGradingConfig(shadow_g=0.1))
        result = grader.grade(img)
        assert result.dtype == np.float64

    def test_shape_preserved(self):
        img = _gradient_rgb(h=16, w=48)
        grader = ColorGrader(ColorGradingConfig(midtone_r=0.1))
        result = grader.grade(img)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# TestColorGradingStep
# ---------------------------------------------------------------------------

class TestColorGradingStep:
    def test_name(self):
        step = ColorGradingStep()
        assert step.name == "Farbabstufung"

    def test_stage(self):
        from astroai.core.pipeline.base import PipelineStage
        step = ColorGradingStep()
        assert step.stage == PipelineStage.PROCESSING

    def test_execute_with_result(self):
        from astroai.core.pipeline.base import PipelineContext
        img = _dark_rgb()
        ctx = PipelineContext(images=[], result=img)
        cfg = ColorGradingConfig(shadow_r=0.2)
        step = ColorGradingStep(config=cfg)
        out_ctx = step.execute(ctx)
        assert out_ctx.result is not None
        assert out_ctx.result.shape == img.shape

    def test_execute_with_images_fallback(self):
        from astroai.core.pipeline.base import PipelineContext
        img = _rgb()
        ctx = PipelineContext(images=[img], result=None)
        step = ColorGradingStep(config=ColorGradingConfig(highlight_g=0.1))
        out_ctx = step.execute(ctx)
        assert out_ctx.result is not None

    def test_execute_no_image_returns_unchanged(self):
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(images=[], result=None)
        step = ColorGradingStep()
        out_ctx = step.execute(ctx)
        assert out_ctx.result is None

    def test_execute_identity_config(self):
        from astroai.core.pipeline.base import PipelineContext
        img = _rgb()
        ctx = PipelineContext(images=[], result=img)
        step = ColorGradingStep(config=ColorGradingConfig())
        out_ctx = step.execute(ctx)
        # identity → same object returned
        assert out_ctx.result is img

    def test_execute_progress_called(self):
        from astroai.core.pipeline.base import PipelineContext
        img = _rgb()
        ctx = PipelineContext(images=[], result=img)
        step = ColorGradingStep(config=ColorGradingConfig(shadow_r=0.1))
        calls = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2
