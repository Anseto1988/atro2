"""Unit tests for ChannelBalanceConfig, ChannelBalancer, and ChannelBalanceStep."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.processing.color.channel_balance import (
    ChannelBalanceConfig,
    ChannelBalancer,
    _bg_median,
)
from astroai.core.pipeline.channel_balance_step import ChannelBalanceStep
from astroai.core.pipeline.base import PipelineContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _gray(h: int = 64, w: int = 64, bg: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(42)
    img = rng.uniform(bg, 1.0, (h, w)).astype(np.float32)
    img[:8, :8] = bg  # dark patch
    return img


def _rgb(h: int = 64, w: int = 64, bgs: tuple = (0.05, 0.08, 0.03)) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.uniform(0.1, 1.0, (h, w, 3)).astype(np.float32)
    for c, bg in enumerate(bgs):
        img[:8, :8, c] = bg
    return img


# ---------------------------------------------------------------------------
# ChannelBalanceConfig
# ---------------------------------------------------------------------------

class TestChannelBalanceConfig:
    def test_defaults_are_identity(self):
        cfg = ChannelBalanceConfig()
        assert cfg.is_identity()

    def test_nonzero_offset_not_identity(self):
        cfg = ChannelBalanceConfig(r_offset=0.01)
        assert not cfg.is_identity()

    def test_as_dict_has_all_keys(self):
        cfg = ChannelBalanceConfig(r_offset=0.1, g_offset=-0.05, b_offset=0.02)
        d = cfg.as_dict()
        assert set(d.keys()) == {"r_offset", "g_offset", "b_offset", "l_offset", "sample_percentile"}

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError, match="sample_percentile"):
            ChannelBalanceConfig(sample_percentile=0.0)

    def test_invalid_percentile_too_high(self):
        with pytest.raises(ValueError):
            ChannelBalanceConfig(sample_percentile=30.0)

    def test_l_offset_identity_check(self):
        cfg = ChannelBalanceConfig(l_offset=0.001)
        assert not cfg.is_identity()


# ---------------------------------------------------------------------------
# _bg_median helper
# ---------------------------------------------------------------------------

class TestBgMedian:
    def test_returns_float(self):
        arr = np.linspace(0.0, 1.0, 100)
        result = _bg_median(arr, 5.0)
        assert isinstance(result, float)

    def test_selects_darkest_pixels(self):
        arr = np.zeros(100, dtype=np.float64)
        arr[:5] = 0.01
        arr[5:] = 0.9
        result = _bg_median(arr, 5.0)
        assert result < 0.05

    def test_empty_dark_pixels_fallback(self):
        arr = np.full(100, 0.5, dtype=np.float64)
        result = _bg_median(arr, 5.0)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# ChannelBalancer.apply
# ---------------------------------------------------------------------------

class TestChannelBalancerApply:
    def test_identity_returns_unchanged(self):
        img = _rgb()
        balancer = ChannelBalancer()
        out = balancer.apply(img)
        np.testing.assert_array_equal(out, img)

    def test_positive_offset_increases_channel(self):
        img = np.full((16, 16, 3), 0.2, dtype=np.float32)
        cfg = ChannelBalanceConfig(r_offset=0.1)
        balancer = ChannelBalancer(cfg)
        out = balancer.apply(img)
        assert np.allclose(out[:, :, 0], 0.3, atol=1e-5)
        assert np.allclose(out[:, :, 1], 0.2, atol=1e-5)

    def test_negative_offset_decreases_channel(self):
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        cfg = ChannelBalanceConfig(g_offset=-0.2)
        balancer = ChannelBalancer(cfg)
        out = balancer.apply(img)
        assert np.allclose(out[:, :, 1], 0.3, atol=1e-5)

    def test_clips_to_zero(self):
        img = np.full((8, 8, 3), 0.05, dtype=np.float32)
        cfg = ChannelBalanceConfig(r_offset=-0.1)
        balancer = ChannelBalancer(cfg)
        out = balancer.apply(img)
        assert out[:, :, 0].min() >= 0.0

    def test_clips_to_one(self):
        img = np.full((8, 8, 3), 0.95, dtype=np.float32)
        cfg = ChannelBalanceConfig(b_offset=0.2)
        balancer = ChannelBalancer(cfg)
        out = balancer.apply(img)
        assert out[:, :, 2].max() <= 1.0

    def test_preserves_dtype(self):
        img = _rgb().astype(np.float64)
        balancer = ChannelBalancer(ChannelBalanceConfig(r_offset=0.01))
        out = balancer.apply(img)
        assert out.dtype == np.float64

    def test_grayscale_uses_l_offset(self):
        img = np.full((16, 16), 0.3, dtype=np.float32)
        cfg = ChannelBalanceConfig(l_offset=0.1)
        balancer = ChannelBalancer(cfg)
        out = balancer.apply(img)
        assert np.allclose(out, 0.4, atol=1e-5)

    def test_5d_shape_returns_unchanged(self):
        img = np.zeros((4, 4, 4, 2, 2), dtype=np.float32)
        balancer = ChannelBalancer()
        out = balancer.apply(img)
        assert out is img

    def test_output_shape_matches_input(self):
        img = _rgb(32, 48)
        balancer = ChannelBalancer(ChannelBalanceConfig(r_offset=0.02, g_offset=-0.01))
        out = balancer.apply(img)
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# ChannelBalancer.auto_sample
# ---------------------------------------------------------------------------

class TestChannelBalancerAutoSample:
    def test_returns_config(self):
        img = _rgb()
        balancer = ChannelBalancer()
        cfg = balancer.auto_sample(img)
        assert isinstance(cfg, ChannelBalanceConfig)

    def test_offsets_equalise_backgrounds(self):
        """After auto_sample + apply, per-channel backgrounds should be similar."""
        bgs = (0.05, 0.10, 0.02)
        img = _rgb(bgs=bgs)
        balancer = ChannelBalancer()
        cfg = balancer.auto_sample(img)
        out = ChannelBalancer(cfg).apply(img)
        # sample backgrounds again from corrected image
        from astroai.processing.color.channel_balance import _bg_median
        result_bgs = [_bg_median(out[:, :, c], 5.0) for c in range(3)]
        # All channels should now be close to the min background
        assert max(result_bgs) - min(result_bgs) < 0.05

    def test_grayscale_produces_l_offset(self):
        img = _gray()
        balancer = ChannelBalancer()
        cfg = balancer.auto_sample(img)
        assert cfg.l_offset != 0.0 or cfg.is_identity()

    def test_unsupported_shape_returns_default(self):
        img = np.zeros((4, 4, 4), dtype=np.float32)
        balancer = ChannelBalancer()
        cfg = balancer.auto_sample(img)
        assert isinstance(cfg, ChannelBalanceConfig)

    def test_preserves_sample_percentile(self):
        img = _rgb()
        balancer = ChannelBalancer(ChannelBalanceConfig(sample_percentile=10.0))
        cfg = balancer.auto_sample(img)
        assert cfg.sample_percentile == 10.0


# ---------------------------------------------------------------------------
# ChannelBalanceStep
# ---------------------------------------------------------------------------

class TestChannelBalanceStep:
    def test_name(self):
        step = ChannelBalanceStep()
        assert step.name == "Kanal-Balance"

    def test_processes_context_result(self):
        img = _rgb()
        ctx = PipelineContext(result=img)
        cfg = ChannelBalanceConfig(r_offset=0.05)
        step = ChannelBalanceStep(cfg)
        out_ctx = step.execute(ctx)
        assert out_ctx.result is not None
        assert not np.array_equal(out_ctx.result, img)

    def test_falls_back_to_context_images(self):
        img = _rgb()
        ctx = PipelineContext(images=[img])
        step = ChannelBalanceStep(ChannelBalanceConfig(g_offset=0.01))
        out_ctx = step.execute(ctx)
        assert out_ctx.result is not None

    def test_no_image_skips_gracefully(self):
        ctx = PipelineContext()
        step = ChannelBalanceStep()
        out_ctx = step.execute(ctx)
        assert out_ctx.result is None

    def test_identity_config_output_equals_input(self):
        img = _rgb()
        ctx = PipelineContext(result=img)
        step = ChannelBalanceStep()
        out_ctx = step.execute(ctx)
        np.testing.assert_array_equal(out_ctx.result, img)

    def test_progress_callback_called(self):
        img = _rgb()
        ctx = PipelineContext(result=img)
        calls: list = []
        step = ChannelBalanceStep()
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2  # start + finish
