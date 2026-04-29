"""Unit tests for astroai.processing.color.saturation."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.processing.color.saturation import (
    SaturationAdjustment,
    SaturationConfig,
    SaturationStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(h_deg: float, s: float = 0.8, v: float = 0.8, size: int = 4) -> np.ndarray:
    """Create a solid-colour image from HSV."""
    h = h_deg / 360.0
    f = h * 6.0
    i = int(f) % 6
    fi = f - int(f)
    p = v * (1.0 - s)
    q = v * (1.0 - fi * s)
    t = v * (1.0 - (1.0 - fi) * s)
    lut = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    r, g, b = lut[i]
    img = np.full((size, size, 3), [r, g, b], dtype=np.float32)
    return img


def _identity_config() -> SaturationConfig:
    return SaturationConfig()


# ---------------------------------------------------------------------------
# SaturationConfig
# ---------------------------------------------------------------------------

class TestSaturationConfig:
    def test_defaults_are_identity(self):
        cfg = SaturationConfig()
        assert cfg.is_identity()

    def test_non_identity_global(self):
        cfg = SaturationConfig(global_saturation=1.5)
        assert not cfg.is_identity()

    def test_non_identity_per_range(self):
        cfg = SaturationConfig(reds=0.5)
        assert not cfg.is_identity()

    def test_as_dict_has_all_ranges(self):
        cfg = SaturationConfig()
        d = cfg.as_dict()
        for key in ("reds", "oranges", "yellows", "greens", "cyans", "blues", "purples"):
            assert key in d

    def test_as_dict_values_match(self):
        cfg = SaturationConfig(reds=0.3, blues=2.0)
        assert cfg.as_dict()["reds"] == pytest.approx(0.3)
        assert cfg.as_dict()["blues"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# SaturationAdjustment — construction and validation
# ---------------------------------------------------------------------------

class TestSaturationAdjustmentConstruction:
    def test_default_config(self):
        adj = SaturationAdjustment()
        assert adj.config.is_identity()

    def test_custom_config(self):
        cfg = SaturationConfig(global_saturation=1.5)
        adj = SaturationAdjustment(cfg)
        assert adj.config.global_saturation == pytest.approx(1.5)

    def test_wrong_shape_raises(self):
        adj = SaturationAdjustment()
        with pytest.raises(ValueError, match="shape"):
            adj.apply(np.zeros((4, 4)))

    def test_wrong_channels_raises(self):
        adj = SaturationAdjustment()
        with pytest.raises(ValueError, match="shape"):
            adj.apply(np.zeros((4, 4, 1)))

    def test_four_channels_raises(self):
        adj = SaturationAdjustment()
        with pytest.raises(ValueError):
            adj.apply(np.zeros((4, 4, 4)))


# ---------------------------------------------------------------------------
# Identity behaviour
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_identity_config_returns_copy(self):
        img = _solid(120, 0.6)
        adj = SaturationAdjustment()
        out = adj.apply(img)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_output_is_copy_not_alias(self):
        img = _solid(120, 0.6)
        adj = SaturationAdjustment()
        out = adj.apply(img)
        assert out is not img

    def test_identity_preserves_dtype_range(self):
        img = _solid(200, 0.5, 0.9)
        out = SaturationAdjustment().apply(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# Global saturation
# ---------------------------------------------------------------------------

class TestGlobalSaturation:
    def test_global_zero_desaturates(self):
        img = _solid(120, 0.8)
        adj = SaturationAdjustment(SaturationConfig(global_saturation=0.0))
        out = adj.apply(img)
        # All channels should be equal (grey) for a saturated input
        np.testing.assert_allclose(out[:, :, 0], out[:, :, 1], atol=1e-5)
        np.testing.assert_allclose(out[:, :, 1], out[:, :, 2], atol=1e-5)

    def test_global_two_increases_saturation(self):
        img = _solid(120, 0.3)  # low saturation
        adj = SaturationAdjustment(SaturationConfig(global_saturation=2.0))
        out = adj.apply(img)
        # Increased saturation → larger spread between channels
        spread_in = float(img.max() - img.min())
        spread_out = float(out.max() - out.min())
        assert spread_out > spread_in

    def test_global_saturation_output_clamped(self):
        img = _solid(120, 0.9)
        adj = SaturationAdjustment(SaturationConfig(global_saturation=10.0))
        out = adj.apply(img)
        assert out.max() <= 1.0 + 1e-6
        assert out.min() >= -1e-6

    def test_grey_unaffected_by_global(self):
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        adj = SaturationAdjustment(SaturationConfig(global_saturation=2.0))
        out = adj.apply(img)
        np.testing.assert_allclose(out, img, atol=1e-5)


# ---------------------------------------------------------------------------
# Per-range saturation (selective)
# ---------------------------------------------------------------------------

class TestPerRangeSaturation:
    @pytest.mark.parametrize("hue,range_name", [
        (0,   "reds"),
        (30,  "oranges"),
        (60,  "yellows"),
        (120, "greens"),
        (180, "cyans"),
        (240, "blues"),
        (300, "purples"),
    ])
    def test_range_desaturates_matching_hue(self, hue, range_name):
        img = _solid(hue, s=0.8, size=8)
        cfg = SaturationConfig(**{range_name: 0.0})
        adj = SaturationAdjustment(cfg)
        out = adj.apply(img)
        # Desaturated: channels converge toward equal
        out_spread = float(out.max() - out.min())
        in_spread = float(img.max() - img.min())
        assert out_spread < in_spread

    def test_reds_zero_does_not_desaturate_green(self):
        img_green = _solid(120, s=0.8, size=8)
        cfg = SaturationConfig(reds=0.0)
        adj = SaturationAdjustment(cfg)
        out = adj.apply(img_green)
        # Green should be mostly unaffected
        in_spread = float(img_green.max() - img_green.min())
        out_spread = float(out.max() - out.min())
        assert out_spread > in_spread * 0.7

    def test_boost_reds_increases_red_saturation(self):
        img = _solid(0, s=0.4, size=8)
        cfg = SaturationConfig(reds=2.5)
        adj = SaturationAdjustment(cfg)
        out = adj.apply(img)
        out_spread = float(out.max() - out.min())
        in_spread = float(img.max() - img.min())
        assert out_spread > in_spread

    def test_boost_blues_only_affects_blue_hue(self):
        img_blue = _solid(240, s=0.5, size=8)
        img_red = _solid(0, s=0.5, size=8)
        cfg = SaturationConfig(blues=3.0)
        adj = SaturationAdjustment(cfg)
        out_blue = adj.apply(img_blue)
        out_red = adj.apply(img_red)
        # Blue should be more affected than red
        blue_change = abs(float(out_blue.max() - out_blue.min()) - float(img_blue.max() - img_blue.min()))
        red_change = abs(float(out_red.max() - out_red.min()) - float(img_red.max() - img_red.min()))
        assert blue_change > red_change

    def test_all_ranges_zero_fully_desaturates(self):
        img = _solid(120, s=0.8)
        cfg = SaturationConfig(
            global_saturation=1.0,
            reds=0.0, oranges=0.0, yellows=0.0,
            greens=0.0, cyans=0.0, blues=0.0, purples=0.0,
        )
        adj = SaturationAdjustment(cfg)
        out = adj.apply(img)
        np.testing.assert_allclose(out[:, :, 0], out[:, :, 1], atol=0.05)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_black_image(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        adj = SaturationAdjustment(SaturationConfig(global_saturation=2.0))
        out = adj.apply(img)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_all_white_image(self):
        img = np.ones((4, 4, 3), dtype=np.float32)
        adj = SaturationAdjustment(SaturationConfig(global_saturation=2.0))
        out = adj.apply(img)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_float64_accepted(self):
        img = _solid(120, 0.7).astype(np.float64)
        adj = SaturationAdjustment()
        out = adj.apply(img)
        assert out.shape == img.shape

    def test_large_image_runs(self):
        img = _solid(200, 0.6, size=256)
        adj = SaturationAdjustment(SaturationConfig(cyans=1.5))
        out = adj.apply(img)
        assert out.shape == img.shape

    def test_output_always_in_range(self):
        rng = np.random.default_rng(99)
        img = rng.random((32, 32, 3)).astype(np.float32)
        adj = SaturationAdjustment(SaturationConfig(
            global_saturation=3.0, reds=2.0, blues=0.1,
        ))
        out = adj.apply(img)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_red_hue_wraps_correctly(self):
        # Hue near 360° (350°) should be treated as red
        img = _solid(350, s=0.8, size=8)
        cfg = SaturationConfig(reds=0.0)
        adj = SaturationAdjustment(cfg)
        out = adj.apply(img)
        out_spread = float(out.max() - out.min())
        in_spread = float(img.max() - img.min())
        assert out_spread < in_spread


# ---------------------------------------------------------------------------
# HSV roundtrip
# ---------------------------------------------------------------------------

class TestHsvRoundtrip:
    def test_rgb_hsv_rgb_roundtrip(self):
        rng = np.random.default_rng(42)
        img = rng.random((16, 16, 3)).astype(np.float64)
        adj = SaturationAdjustment()
        hsv = adj._rgb_to_hsv(img)
        out = adj._hsv_to_rgb(hsv)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_hsv_v_equals_rgb_max(self):
        img = _solid(120, 0.7, 0.8).astype(np.float64)
        adj = SaturationAdjustment()
        hsv = adj._rgb_to_hsv(img)
        np.testing.assert_allclose(hsv[:, :, 2], img.max(axis=2), atol=1e-6)

    def test_grey_has_zero_saturation(self):
        img = np.full((4, 4, 3), 0.6, dtype=np.float64)
        adj = SaturationAdjustment()
        hsv = adj._rgb_to_hsv(img)
        np.testing.assert_allclose(hsv[:, :, 1], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# SaturationStep pipeline integration
# ---------------------------------------------------------------------------

class TestSaturationStep:
    def _make_context(self, img: np.ndarray):
        from unittest.mock import MagicMock
        ctx = MagicMock()
        ctx.result = img
        return ctx

    def test_step_name(self):
        step = SaturationStep()
        assert "ättigung" in step.name

    def test_step_stage_is_processing(self):
        from astroai.core.pipeline.base import PipelineStage
        step = SaturationStep()
        assert step.stage == PipelineStage.PROCESSING

    def test_step_applies_adjustment(self):
        img = _solid(120, s=0.8, size=8).astype(np.float64)
        cfg = SaturationConfig(greens=0.0)
        step = SaturationStep(cfg)
        ctx = self._make_context(img)
        result_ctx = step.execute(ctx)
        out_spread = float(result_ctx.result.max() - result_ctx.result.min())
        in_spread = float(img.max() - img.min())
        assert out_spread < in_spread

    def test_step_skips_none_result(self):
        step = SaturationStep()
        ctx = self._make_context(None)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is None

    def test_step_calls_progress(self):
        img = _solid(120, 0.6).astype(np.float64)
        step = SaturationStep()
        ctx = self._make_context(img)
        calls = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2

    def test_step_default_config(self):
        img = _solid(120, 0.6, size=8).astype(np.float64)
        step = SaturationStep()
        ctx = self._make_context(img)
        result_ctx = step.execute(ctx)
        np.testing.assert_allclose(result_ctx.result, img, atol=1e-5)
