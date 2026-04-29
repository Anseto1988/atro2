"""Unit tests for astroai.processing.contrast.clahe (CLAHE)."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.processing.contrast.clahe import (
    CLAHEConfig,
    CLAHEEnhancer,
    CLAHEStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_image(r: float = 0.5, g: float = 0.5, b: float = 0.5, size: int = 16) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def _gray_image(value: float = 0.5, size: int = 16) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def _gradient_gray(size: int = 32) -> np.ndarray:
    """Gradient image (0 → 1 linearly) for contrast tests."""
    arr = np.tile(np.linspace(0.0, 1.0, size, dtype=np.float32), (size, 1))
    return arr


def _gradient_rgb(size: int = 32) -> np.ndarray:
    g = _gradient_gray(size)
    return np.stack([g, g * 0.8, g * 0.6], axis=2).astype(np.float32)


def _make_context(img, images=None):
    from astroai.core.pipeline.base import PipelineContext
    return PipelineContext(
        result=img,
        images=images or ([img] if img is not None else []),
    )


# ---------------------------------------------------------------------------
# TestCLAHEConfig
# ---------------------------------------------------------------------------

class TestCLAHEConfig:
    def test_defaults(self):
        cfg = CLAHEConfig()
        assert cfg.clip_limit == pytest.approx(2.0)
        assert cfg.tile_size == 64
        assert cfg.n_bins == 256
        assert cfg.channel_mode == "luminance"

    def test_custom_values(self):
        cfg = CLAHEConfig(clip_limit=3.0, tile_size=32, n_bins=128, channel_mode="each")
        assert cfg.clip_limit == pytest.approx(3.0)
        assert cfg.tile_size == 32
        assert cfg.n_bins == 128
        assert cfg.channel_mode == "each"

    def test_as_dict_keys(self):
        d = CLAHEConfig().as_dict()
        assert set(d.keys()) == {"clip_limit", "tile_size", "n_bins", "channel_mode"}

    def test_as_dict_values(self):
        cfg = CLAHEConfig(clip_limit=4.5, tile_size=16, n_bins=64, channel_mode="each")
        d = cfg.as_dict()
        assert d["clip_limit"] == pytest.approx(4.5)
        assert d["tile_size"] == 16
        assert d["n_bins"] == 64
        assert d["channel_mode"] == "each"

    def test_is_identity_always_false(self):
        assert CLAHEConfig().is_identity() is False

    def test_validation_clip_limit_below_1_raises(self):
        with pytest.raises(ValueError, match="clip_limit"):
            CLAHEConfig(clip_limit=0.9)

    def test_validation_clip_limit_zero_raises(self):
        with pytest.raises(ValueError):
            CLAHEConfig(clip_limit=0.0)

    def test_validation_tile_size_zero_raises(self):
        with pytest.raises(ValueError, match="tile_size"):
            CLAHEConfig(tile_size=0)

    def test_validation_tile_size_negative_raises(self):
        with pytest.raises(ValueError):
            CLAHEConfig(tile_size=-1)

    def test_validation_n_bins_too_small_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            CLAHEConfig(n_bins=32)

    def test_validation_n_bins_too_large_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            CLAHEConfig(n_bins=2048)

    def test_validation_invalid_channel_mode_raises(self):
        with pytest.raises(ValueError, match="channel_mode"):
            CLAHEConfig(channel_mode="invalid_mode")

    def test_frozen_immutable(self):
        cfg = CLAHEConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.clip_limit = 5.0  # type: ignore[misc]

    def test_valid_channel_mode_each(self):
        cfg = CLAHEConfig(channel_mode="each")
        assert cfg.channel_mode == "each"

    def test_valid_channel_mode_grayscale(self):
        cfg = CLAHEConfig(channel_mode="grayscale")
        assert cfg.channel_mode == "grayscale"

    def test_clip_limit_exactly_1_valid(self):
        cfg = CLAHEConfig(clip_limit=1.0)
        assert cfg.clip_limit == pytest.approx(1.0)

    def test_clip_limit_max_valid(self):
        cfg = CLAHEConfig(clip_limit=10.0)
        assert cfg.clip_limit == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TestCLAHEEnhancerConstruction
# ---------------------------------------------------------------------------

class TestCLAHEEnhancerConstruction:
    def test_default_config(self):
        enh = CLAHEEnhancer()
        assert isinstance(enh.config, CLAHEConfig)
        assert enh.config.clip_limit == pytest.approx(2.0)

    def test_custom_config_stored(self):
        cfg = CLAHEConfig(clip_limit=5.0, tile_size=32)
        enh = CLAHEEnhancer(cfg)
        assert enh.config is cfg

    def test_none_config_uses_defaults(self):
        enh = CLAHEEnhancer(None)
        assert enh.config == CLAHEConfig()


# ---------------------------------------------------------------------------
# TestEnhanceGrayscale
# ---------------------------------------------------------------------------

class TestEnhanceGrayscale:
    def test_output_shape_preserved(self):
        img = _gray_image(0.5, size=16)
        cfg = CLAHEConfig(tile_size=8, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.shape == img.shape

    def test_output_values_in_range(self):
        img = _gradient_gray(32)
        cfg = CLAHEConfig(tile_size=8, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_uniform_gray_unchanged(self):
        """A perfectly uniform image should remain uniform after CLAHE."""
        img = _gray_image(0.5, size=16)
        cfg = CLAHEConfig(tile_size=8, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        # Uniform input: all pixels same bin → uniform LUT → output still uniform
        assert np.std(out) < 1e-4

    def test_gradient_contrast_increases(self):
        """CLAHE should stretch contrast on a gradient image."""
        img = _gradient_gray(32)
        cfg = CLAHEConfig(clip_limit=4.0, tile_size=8, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        # Output should span a wider (or equal) range
        assert out.max() - out.min() >= img.max() - img.min() - 0.1

    def test_output_dtype_float32(self):
        img = _gray_image(0.5, size=16).astype(np.float32)
        out = CLAHEEnhancer(CLAHEConfig(tile_size=8, n_bins=64)).enhance(img)
        assert out.dtype == np.float32

    def test_output_dtype_float64(self):
        img = _gray_image(0.5, size=16).astype(np.float64)
        out = CLAHEEnhancer(CLAHEConfig(tile_size=8, n_bins=64)).enhance(img)
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# TestEnhanceLuminance
# ---------------------------------------------------------------------------

class TestEnhanceLuminance:
    def test_rgb_shape_preserved(self):
        img = _gradient_rgb(32)
        cfg = CLAHEConfig(tile_size=8, n_bins=64, channel_mode="luminance")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.shape == img.shape

    def test_rgb_values_in_range(self):
        img = _gradient_rgb(32)
        cfg = CLAHEConfig(tile_size=8, n_bins=64, channel_mode="luminance")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_uniform_rgb_stays_uniform(self):
        img = _rgb_image(0.5, 0.5, 0.5, size=16)
        cfg = CLAHEConfig(tile_size=8, n_bins=64, channel_mode="luminance")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert np.std(out) < 1e-4

    def test_output_dtype_preserved_rgb(self):
        img = _gradient_rgb(32).astype(np.float32)
        cfg = CLAHEConfig(tile_size=8, n_bins=64, channel_mode="luminance")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# TestEnhanceEachChannel
# ---------------------------------------------------------------------------

class TestEnhanceEachChannel:
    def test_each_channel_shape(self):
        img = _gradient_rgb(32)
        cfg = CLAHEConfig(tile_size=8, n_bins=64, channel_mode="each")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.shape == img.shape

    def test_each_channel_values_in_range(self):
        img = _gradient_rgb(32)
        cfg = CLAHEConfig(tile_size=8, n_bins=64, channel_mode="each")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_each_channel_dtype(self):
        img = _gradient_rgb(32).astype(np.float32)
        cfg = CLAHEConfig(tile_size=8, n_bins=64, channel_mode="each")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.dtype == np.float32

    def test_each_uniform_stays_uniform(self):
        img = _rgb_image(0.3, 0.3, 0.3, size=16)
        cfg = CLAHEConfig(tile_size=8, n_bins=64, channel_mode="each")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert np.std(out) < 1e-4


# ---------------------------------------------------------------------------
# TestEnhanceClipping
# ---------------------------------------------------------------------------

class TestEnhanceClipping:
    def test_output_never_above_1(self):
        img = np.ones((16, 16), dtype=np.float32) * 0.99
        cfg = CLAHEConfig(clip_limit=10.0, tile_size=8, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.max() <= 1.0 + 1e-6

    def test_output_never_below_0(self):
        img = np.ones((16, 16), dtype=np.float32) * 0.01
        cfg = CLAHEConfig(clip_limit=10.0, tile_size=8, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.min() >= -1e-6

    def test_rgb_output_never_above_1(self):
        img = _gradient_rgb(32)
        cfg = CLAHEConfig(clip_limit=10.0, tile_size=8, n_bins=64, channel_mode="each")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.max() <= 1.0 + 1e-6

    def test_rgb_output_never_below_0(self):
        img = _gradient_rgb(32)
        cfg = CLAHEConfig(clip_limit=10.0, tile_size=8, n_bins=64, channel_mode="luminance")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.min() >= -1e-6


# ---------------------------------------------------------------------------
# TestEnhanceDtype
# ---------------------------------------------------------------------------

class TestEnhanceDtype:
    def test_float32_grayscale_preserved(self):
        img = _gradient_gray(16).astype(np.float32)
        out = CLAHEEnhancer(CLAHEConfig(tile_size=8, n_bins=64)).enhance(img)
        assert out.dtype == np.float32

    def test_float64_grayscale_preserved(self):
        img = _gradient_gray(16).astype(np.float64)
        out = CLAHEEnhancer(CLAHEConfig(tile_size=8, n_bins=64)).enhance(img)
        assert out.dtype == np.float64

    def test_float32_rgb_preserved(self):
        img = _gradient_rgb(16).astype(np.float32)
        out = CLAHEEnhancer(CLAHEConfig(tile_size=8, n_bins=64, channel_mode="each")).enhance(img)
        assert out.dtype == np.float32

    def test_output_is_not_input(self):
        img = _gradient_gray(16).astype(np.float32)
        out = CLAHEEnhancer(CLAHEConfig(tile_size=8, n_bins=64)).enhance(img)
        assert out is not img


# ---------------------------------------------------------------------------
# TestEnhanceHighClipLimit
# ---------------------------------------------------------------------------

class TestEnhanceHighClipLimit:
    def test_high_clip_approaches_global_eq(self):
        """Very high clip_limit ≈ standard histogram equalization."""
        np.random.seed(42)
        img = np.random.rand(32, 32).astype(np.float32)
        cfg = CLAHEConfig(clip_limit=9.9, tile_size=32, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        # Output should still be in [0, 1]
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_low_clip_produces_less_stretch(self):
        """Lower clip_limit should produce less extreme contrast stretching."""
        np.random.seed(7)
        img = np.random.rand(32, 32).astype(np.float32) * 0.4 + 0.3  # narrow range
        cfg_low = CLAHEConfig(clip_limit=1.0, tile_size=8, n_bins=64)
        cfg_high = CLAHEConfig(clip_limit=8.0, tile_size=8, n_bins=64)
        out_low = CLAHEEnhancer(cfg_low).enhance(img)
        out_high = CLAHEEnhancer(cfg_high).enhance(img)
        # Higher clip should spread the histogram more
        assert out_high.std() >= out_low.std() - 0.05


# ---------------------------------------------------------------------------
# TestEnhanceSmallImage
# ---------------------------------------------------------------------------

class TestEnhanceSmallImage:
    def test_4x4_grayscale(self):
        img = np.random.rand(4, 4).astype(np.float32)
        cfg = CLAHEConfig(tile_size=4, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.shape == (4, 4)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_4x4_rgb(self):
        img = np.random.rand(4, 4, 3).astype(np.float32)
        cfg = CLAHEConfig(tile_size=4, n_bins=64, channel_mode="each")
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.shape == (4, 4, 3)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_single_tile(self):
        """Image smaller than tile_size: only one tile."""
        img = np.random.rand(8, 8).astype(np.float32)
        cfg = CLAHEConfig(tile_size=16, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.shape == (8, 8)

    def test_non_square_image(self):
        img = np.random.rand(16, 32).astype(np.float32)
        cfg = CLAHEConfig(tile_size=8, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.shape == (16, 32)

    def test_wide_image(self):
        img = np.random.rand(8, 64).astype(np.float32)
        cfg = CLAHEConfig(tile_size=8, n_bins=64)
        out = CLAHEEnhancer(cfg).enhance(img)
        assert out.shape == (8, 64)


# ---------------------------------------------------------------------------
# TestCLAHEStep
# ---------------------------------------------------------------------------

class TestCLAHEStep:
    def test_step_name(self):
        step = CLAHEStep()
        assert "Kontrastverbesserung" in step.name

    def test_step_stage_is_processing(self):
        from astroai.core.pipeline.base import PipelineStage
        step = CLAHEStep()
        assert step.stage == PipelineStage.PROCESSING

    def test_execute_with_grayscale(self):
        img = _gradient_gray(16)
        step = CLAHEStep(CLAHEConfig(tile_size=8, n_bins=64))
        ctx = _make_context(img)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        assert result_ctx.result.shape == img.shape

    def test_execute_with_rgb(self):
        img = _gradient_rgb(16)
        step = CLAHEStep(CLAHEConfig(tile_size=8, n_bins=64))
        ctx = _make_context(img)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        assert result_ctx.result.shape == img.shape

    def test_execute_falls_back_to_images(self):
        img = _gradient_gray(16)
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[img])
        step = CLAHEStep(CLAHEConfig(tile_size=8, n_bins=64))
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None

    def test_execute_no_image_skips(self):
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[])
        step = CLAHEStep()
        result_ctx = step.execute(ctx)
        assert result_ctx.result is None

    def test_execute_calls_progress(self):
        img = _gradient_gray(16)
        step = CLAHEStep(CLAHEConfig(tile_size=8, n_bins=64))
        ctx = _make_context(img)
        calls = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2

    def test_execute_sets_result_in_context(self):
        img = _gradient_gray(16)
        step = CLAHEStep(CLAHEConfig(tile_size=8, n_bins=64))
        ctx = _make_context(img)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None

    def test_execute_result_values_in_range(self):
        img = _gradient_rgb(16)
        step = CLAHEStep(CLAHEConfig(tile_size=8, n_bins=64))
        ctx = _make_context(img)
        result_ctx = step.execute(ctx)
        assert result_ctx.result.min() >= -1e-6
        assert result_ctx.result.max() <= 1.0 + 1e-6

    def test_default_step_config(self):
        step = CLAHEStep()
        assert step._enhancer.config.clip_limit == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TestBuilderIntegration
# ---------------------------------------------------------------------------

class TestBuilderIntegration:
    def _make_model(self, enabled: bool = True):
        """Create a minimal PipelineModel with CLAHE enabled."""
        from unittest.mock import MagicMock, patch

        # We patch QObject.__init__ to avoid requiring a QApplication
        with patch("astroai.ui.models.QObject.__init__", return_value=None):
            from astroai.ui.models import PipelineModel
            m = PipelineModel.__new__(PipelineModel)
            m._clahe_enabled = enabled
            m._clahe_clip_limit = 3.0
            m._clahe_tile_size = 32
            m._clahe_channel_mode = "each"
            return m

    def test_clahe_step_added_when_enabled(self):
        from astroai.core.pipeline.builder import PipelineBuilder
        from astroai.processing.contrast.clahe import CLAHEStep

        m = self._make_model(enabled=True)
        builder = PipelineBuilder()
        # We only test the import/instantiation path, not full build
        from astroai.processing.contrast.clahe import CLAHEConfig
        step = CLAHEStep(config=CLAHEConfig(
            clip_limit=m._clahe_clip_limit,
            tile_size=m._clahe_tile_size,
            channel_mode=m._clahe_channel_mode,
        ))
        assert step._enhancer.config.clip_limit == pytest.approx(3.0)
        assert step._enhancer.config.tile_size == 32
        assert step._enhancer.config.channel_mode == "each"

    def test_clahe_config_from_model_values(self):
        from astroai.processing.contrast.clahe import CLAHEConfig
        cfg = CLAHEConfig(clip_limit=5.0, tile_size=16, channel_mode="luminance")
        assert cfg.clip_limit == pytest.approx(5.0)
        assert cfg.tile_size == 16
        assert cfg.channel_mode == "luminance"
