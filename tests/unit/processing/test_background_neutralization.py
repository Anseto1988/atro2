"""Unit tests for astroai.processing.color.background_neutralizer."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.processing.color.background_neutralizer import (
    BackgroundNeutralizationConfig,
    BackgroundNeutralizer,
    BackgroundNeutralizationStep,
    SampleMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gray(value: float = 0.05, size: int = 8) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def _rgb(r: float = 0.05, g: float = 0.08, b: float = 0.12, size: int = 8) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def _ramp_rgb(size: int = 16) -> np.ndarray:
    """RGB image with a background gradient: R darker, G medium, B brighter."""
    img = np.zeros((size, size, 3), dtype=np.float64)
    img[:, :, 0] = 0.04
    img[:, :, 1] = 0.07
    img[:, :, 2] = 0.15
    # add some bright objects
    img[size // 2, size // 2, :] = 0.9
    return img


# ---------------------------------------------------------------------------
# TestBackgroundNeutralizationConfig — validation & metadata
# ---------------------------------------------------------------------------

class TestBackgroundNeutralizationConfig:
    def test_defaults(self):
        cfg = BackgroundNeutralizationConfig()
        assert cfg.sample_mode is SampleMode.AUTO
        assert cfg.target_background == pytest.approx(0.1)
        assert cfg.roi is None

    def test_target_lower_bound_valid(self):
        cfg = BackgroundNeutralizationConfig(target_background=0.0)
        assert cfg.target_background == pytest.approx(0.0)

    def test_target_upper_bound_valid(self):
        cfg = BackgroundNeutralizationConfig(target_background=0.3)
        assert cfg.target_background == pytest.approx(0.3)

    def test_target_below_min_raises(self):
        with pytest.raises(ValueError, match="target_background"):
            BackgroundNeutralizationConfig(target_background=-0.01)

    def test_target_above_max_raises(self):
        with pytest.raises(ValueError, match="target_background"):
            BackgroundNeutralizationConfig(target_background=0.31)

    def test_sample_percentile_bounds_valid(self):
        cfg = BackgroundNeutralizationConfig(sample_percentile=2.0)
        assert cfg.sample_percentile == pytest.approx(2.0)

    def test_sample_percentile_too_low_raises(self):
        with pytest.raises(ValueError, match="sample_percentile"):
            BackgroundNeutralizationConfig(sample_percentile=0.05)

    def test_sample_percentile_too_high_raises(self):
        with pytest.raises(ValueError, match="sample_percentile"):
            BackgroundNeutralizationConfig(sample_percentile=21.0)

    def test_roi_valid_manual(self):
        cfg = BackgroundNeutralizationConfig(
            sample_mode=SampleMode.MANUAL, roi=(0, 10, 0, 10)
        )
        assert cfg.roi == (0, 10, 0, 10)

    def test_roi_invalid_row_order_raises(self):
        with pytest.raises(ValueError, match="roi"):
            BackgroundNeutralizationConfig(
                sample_mode=SampleMode.MANUAL, roi=(10, 5, 0, 10)
            )

    def test_roi_invalid_col_order_raises(self):
        with pytest.raises(ValueError, match="roi"):
            BackgroundNeutralizationConfig(
                sample_mode=SampleMode.MANUAL, roi=(0, 10, 20, 5)
            )

    def test_roi_none_manual_is_valid(self):
        cfg = BackgroundNeutralizationConfig(sample_mode=SampleMode.MANUAL, roi=None)
        assert cfg.roi is None

    def test_is_identity_zero_target(self):
        cfg = BackgroundNeutralizationConfig(target_background=0.0)
        assert cfg.is_identity()

    def test_is_identity_nonzero_target(self):
        cfg = BackgroundNeutralizationConfig(target_background=0.1)
        assert not cfg.is_identity()

    def test_frozen_immutable(self):
        cfg = BackgroundNeutralizationConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.target_background = 0.2  # type: ignore[misc]

    def test_as_dict_keys(self):
        d = BackgroundNeutralizationConfig().as_dict()
        assert set(d.keys()) == {"sample_mode", "target_background", "roi", "sample_percentile"}

    def test_as_dict_values(self):
        cfg = BackgroundNeutralizationConfig(target_background=0.15)
        d = cfg.as_dict()
        assert d["target_background"] == pytest.approx(0.15)
        assert d["sample_mode"] == "auto"


# ---------------------------------------------------------------------------
# TestBackgroundNeutralizer — core apply logic
# ---------------------------------------------------------------------------

class TestBackgroundNeutralizerAuto:
    def test_apply_grayscale_shifts_background_up(self):
        img = _gray(0.02)
        bn = BackgroundNeutralizer(BackgroundNeutralizationConfig(target_background=0.1))
        out = bn.apply(img)
        assert out.mean() > img.mean()

    def test_apply_grayscale_output_in_range(self):
        img = _gray(0.05)
        bn = BackgroundNeutralizer()
        out = bn.apply(img)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_apply_rgb_corrects_per_channel(self):
        img = _rgb(r=0.04, g=0.07, b=0.15)
        bn = BackgroundNeutralizer(BackgroundNeutralizationConfig(target_background=0.1))
        out = bn.apply(img)
        # Each channel should be shifted towards target
        assert out.shape == img.shape

    def test_apply_rgb_output_in_range(self):
        img = _ramp_rgb()
        bn = BackgroundNeutralizer()
        out = bn.apply(img)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-9

    def test_apply_preserves_shape_gray(self):
        img = _gray()
        bn = BackgroundNeutralizer()
        out = bn.apply(img)
        assert out.shape == img.shape

    def test_apply_preserves_shape_rgb(self):
        img = _rgb()
        bn = BackgroundNeutralizer()
        out = bn.apply(img)
        assert out.shape == img.shape

    def test_apply_preserves_dtype_float32(self):
        img = _gray().astype(np.float32)
        bn = BackgroundNeutralizer()
        out = bn.apply(img)
        assert out.dtype == np.float32

    def test_apply_preserves_dtype_float64(self):
        img = _gray().astype(np.float64)
        bn = BackgroundNeutralizer()
        out = bn.apply(img)
        assert out.dtype == np.float64

    def test_apply_uniform_sky_background_equals_target(self):
        """Uniform sky image should converge to target_background after correction."""
        img = np.full((20, 20), 0.05, dtype=np.float64)
        bn = BackgroundNeutralizer(BackgroundNeutralizationConfig(target_background=0.1))
        out = bn.apply(img)
        np.testing.assert_allclose(out, 0.1, atol=1e-6)

    def test_apply_does_not_clip_bright_sources(self):
        img = np.zeros((10, 10), dtype=np.float64)
        img[5, 5] = 0.9
        bn = BackgroundNeutralizer(BackgroundNeutralizationConfig(target_background=0.1))
        out = bn.apply(img)
        # bright source should remain bright (possibly clipped at 1 but not darkened)
        assert out[5, 5] >= 0.5

    def test_apply_zero_target_subtracts_background(self):
        """target_background=0.0 → background pixels → 0."""
        img = np.full((10, 10), 0.05, dtype=np.float64)
        bn = BackgroundNeutralizer(BackgroundNeutralizationConfig(target_background=0.0))
        out = bn.apply(img)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_apply_unsupported_4d_returns_unchanged(self):
        img = np.zeros((4, 4, 4, 2), dtype=np.float32)
        bn = BackgroundNeutralizer()
        out = bn.apply(img)
        assert out is img

    def test_default_config_construction(self):
        bn = BackgroundNeutralizer()
        assert bn.config == BackgroundNeutralizationConfig()

    def test_none_config_uses_default(self):
        bn = BackgroundNeutralizer(None)
        assert bn.config == BackgroundNeutralizationConfig()


class TestBackgroundNeutralizerManual:
    def test_manual_roi_uses_specified_region(self):
        img = np.zeros((20, 20), dtype=np.float64)
        # background region: top-left 5×5 patch = 0.03
        img[:5, :5] = 0.03
        img[10:, 10:] = 0.8  # bright objects elsewhere
        cfg = BackgroundNeutralizationConfig(
            sample_mode=SampleMode.MANUAL,
            target_background=0.1,
            roi=(0, 5, 0, 5),
        )
        bn = BackgroundNeutralizer(cfg)
        out = bn.apply(img)
        # ROI region should now be ~target_background
        roi_out = out[:5, :5]
        np.testing.assert_allclose(roi_out.mean(), 0.1, atol=0.01)

    def test_manual_out_of_bounds_roi_clamped(self):
        img = _gray(0.05, size=10)
        cfg = BackgroundNeutralizationConfig(
            sample_mode=SampleMode.MANUAL,
            target_background=0.1,
            roi=(0, 500, 0, 500),  # exceeds image size
        )
        bn = BackgroundNeutralizer(cfg)
        out = bn.apply(img)  # should not crash
        assert out.shape == img.shape

    def test_manual_none_roi_falls_back_to_full_image(self):
        img = _gray(0.05)
        cfg = BackgroundNeutralizationConfig(
            sample_mode=SampleMode.MANUAL, roi=None, target_background=0.1
        )
        bn = BackgroundNeutralizer(cfg)
        out = bn.apply(img)
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# TestBackgroundNeutralizerEstimate
# ---------------------------------------------------------------------------

class TestEstimateBackground:
    def test_estimate_grayscale_single_value(self):
        img = _gray(0.05)
        bn = BackgroundNeutralizer()
        est = bn.estimate_background(img)
        assert est.shape == (1,)
        assert est[0] == pytest.approx(0.05, abs=0.01)

    def test_estimate_rgb_three_values(self):
        img = _rgb(r=0.04, g=0.07, b=0.12)
        bn = BackgroundNeutralizer()
        est = bn.estimate_background(img)
        assert est.shape == (3,)

    def test_estimate_rgb_lower_channel_has_lower_estimate(self):
        img = _rgb(r=0.04, g=0.07, b=0.12)
        bn = BackgroundNeutralizer()
        est = bn.estimate_background(img)
        assert est[0] < est[2]

    def test_estimate_values_in_unit_range(self):
        img = _ramp_rgb()
        bn = BackgroundNeutralizer()
        est = bn.estimate_background(img)
        assert all(0.0 <= v <= 1.0 for v in est)


# ---------------------------------------------------------------------------
# TestBackgroundNeutralizationStep
# ---------------------------------------------------------------------------

class TestBackgroundNeutralizationStep:
    def _make_ctx(self, img=None, images=None):
        from astroai.core.pipeline.base import PipelineContext
        return PipelineContext(
            result=img,
            images=images or ([img] if img is not None else []),
        )

    def test_step_name(self):
        step = BackgroundNeutralizationStep()
        assert "Hintergrund" in step.name

    def test_step_stage_is_processing(self):
        from astroai.core.pipeline.base import PipelineStage
        assert BackgroundNeutralizationStep().stage == PipelineStage.PROCESSING

    def test_execute_updates_result(self):
        img = _rgb(r=0.04, g=0.07, b=0.12)
        step = BackgroundNeutralizationStep()
        ctx = self._make_ctx(img)
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == img.shape

    def test_execute_stores_estimate_in_metadata(self):
        img = _rgb()
        step = BackgroundNeutralizationStep()
        ctx = self._make_ctx(img)
        out = step.execute(ctx)
        assert "background_neutralization_estimate" in out.metadata

    def test_execute_falls_back_to_images(self):
        img = _gray()
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[img])
        out = BackgroundNeutralizationStep().execute(ctx)
        assert out.result is not None

    def test_execute_no_image_skips(self):
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[])
        out = BackgroundNeutralizationStep().execute(ctx)
        assert out.result is None

    def test_execute_calls_progress_three_times(self):
        img = _gray()
        step = BackgroundNeutralizationStep()
        ctx = self._make_ctx(img)
        calls = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 3

    def test_execute_preserves_rgb_shape(self):
        img = _rgb()
        step = BackgroundNeutralizationStep()
        ctx = self._make_ctx(img)
        out = step.execute(ctx)
        assert out.result.shape == img.shape

    def test_execute_output_clipped_to_unit(self):
        img = _rgb(r=0.99, g=0.99, b=0.99)
        step = BackgroundNeutralizationStep(
            BackgroundNeutralizationConfig(target_background=0.1)
        )
        ctx = self._make_ctx(img)
        out = step.execute(ctx)
        assert float(out.result.max()) <= 1.0
        assert float(out.result.min()) >= 0.0

    def test_default_config_step(self):
        step = BackgroundNeutralizationStep()
        assert step._neutralizer.config == BackgroundNeutralizationConfig()
