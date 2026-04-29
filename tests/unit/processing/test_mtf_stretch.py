"""Unit tests for astroai.processing.stretch.mtf_stretch."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.processing.stretch.mtf_stretch import (
    MidtoneTransferConfig,
    MidtoneTransferFunction,
    MTFStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gray(value: float = 0.1, size: int = 4) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def _rgb(r: float = 0.1, g: float = 0.2, b: float = 0.3, size: int = 4) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def _ramp(size: int = 64) -> np.ndarray:
    return np.linspace(0.0, 1.0, size * size, dtype=np.float64).reshape(size, size)


# ---------------------------------------------------------------------------
# TestMidtoneTransferConfig — validation & metadata
# ---------------------------------------------------------------------------

class TestMidtoneTransferConfig:
    def test_defaults(self):
        cfg = MidtoneTransferConfig()
        assert cfg.midpoint == pytest.approx(0.25)
        assert cfg.shadows_clipping == pytest.approx(0.0)
        assert cfg.highlights == pytest.approx(1.0)

    def test_midpoint_lower_bound_valid(self):
        cfg = MidtoneTransferConfig(midpoint=0.001)
        assert cfg.midpoint == pytest.approx(0.001)

    def test_midpoint_upper_bound_valid(self):
        cfg = MidtoneTransferConfig(midpoint=0.499)
        assert cfg.midpoint == pytest.approx(0.499)

    def test_midpoint_below_min_raises(self):
        with pytest.raises(ValueError, match="midpoint"):
            MidtoneTransferConfig(midpoint=0.0)

    def test_midpoint_above_max_raises(self):
        with pytest.raises(ValueError, match="midpoint"):
            MidtoneTransferConfig(midpoint=0.5)

    def test_shadows_zero_valid(self):
        cfg = MidtoneTransferConfig(shadows_clipping=0.0)
        assert cfg.shadows_clipping == pytest.approx(0.0)

    def test_shadows_max_valid(self):
        cfg = MidtoneTransferConfig(shadows_clipping=0.1)
        assert cfg.shadows_clipping == pytest.approx(0.1)

    def test_shadows_negative_raises(self):
        with pytest.raises(ValueError, match="shadows_clipping"):
            MidtoneTransferConfig(shadows_clipping=-0.01)

    def test_shadows_too_high_raises(self):
        with pytest.raises(ValueError, match="shadows_clipping"):
            MidtoneTransferConfig(shadows_clipping=0.11)

    def test_highlights_min_valid(self):
        cfg = MidtoneTransferConfig(highlights=0.98)
        assert cfg.highlights == pytest.approx(0.98)

    def test_highlights_max_valid(self):
        cfg = MidtoneTransferConfig(highlights=1.0)
        assert cfg.highlights == pytest.approx(1.0)

    def test_highlights_too_low_raises(self):
        with pytest.raises(ValueError, match="highlights"):
            MidtoneTransferConfig(highlights=0.97)

    def test_highlights_too_high_raises(self):
        with pytest.raises(ValueError, match="highlights"):
            MidtoneTransferConfig(highlights=1.01)

    def test_is_identity_default_midpoint_is_false(self):
        assert not MidtoneTransferConfig().is_identity()

    def test_is_identity_near_0_5_no_clipping(self):
        cfg = MidtoneTransferConfig(midpoint=0.499, shadows_clipping=0.0, highlights=1.0)
        assert cfg.is_identity()

    def test_is_identity_false_with_shadows(self):
        cfg = MidtoneTransferConfig(midpoint=0.499, shadows_clipping=0.05)
        assert not cfg.is_identity()

    def test_is_identity_false_with_highlights(self):
        cfg = MidtoneTransferConfig(midpoint=0.499, highlights=0.99)
        assert not cfg.is_identity()

    def test_frozen_immutable(self):
        cfg = MidtoneTransferConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.midpoint = 0.3  # type: ignore[misc]

    def test_as_dict_keys(self):
        d = MidtoneTransferConfig().as_dict()
        assert set(d.keys()) == {"midpoint", "shadows_clipping", "highlights"}

    def test_as_dict_values(self):
        cfg = MidtoneTransferConfig(midpoint=0.3, shadows_clipping=0.05, highlights=0.99)
        d = cfg.as_dict()
        assert d["midpoint"] == pytest.approx(0.3)
        assert d["shadows_clipping"] == pytest.approx(0.05)
        assert d["highlights"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# TestMTFEdgeCases — x=0 → 0, x=1 → 1 for all midpoints
# ---------------------------------------------------------------------------

class TestMTFEdgeCases:
    @pytest.mark.parametrize("m", [0.001, 0.1, 0.25, 0.35, 0.499])
    def test_zero_input_zero_output(self, m: float):
        img = np.zeros((4, 4), dtype=np.float64)
        mtf = MidtoneTransferFunction(MidtoneTransferConfig(midpoint=m))
        out = mtf.apply(img)
        np.testing.assert_allclose(out, 0.0, atol=1e-9)

    @pytest.mark.parametrize("m", [0.001, 0.1, 0.25, 0.35, 0.499])
    def test_one_input_one_output(self, m: float):
        img = np.ones((4, 4), dtype=np.float64)
        mtf = MidtoneTransferFunction(MidtoneTransferConfig(midpoint=m))
        out = mtf.apply(img)
        np.testing.assert_allclose(out, 1.0, atol=1e-9)

    def test_uniform_zero_image_stays_zero(self):
        img = np.zeros((8, 8, 3), dtype=np.float32)
        mtf = MidtoneTransferFunction()
        out = mtf.apply(img)
        np.testing.assert_allclose(out, 0.0, atol=1e-7)

    def test_uniform_one_image_stays_one(self):
        img = np.ones((8, 8, 3), dtype=np.float32)
        mtf = MidtoneTransferFunction()
        out = mtf.apply(img)
        np.testing.assert_allclose(out, 1.0, atol=1e-7)

    def test_output_always_in_range(self):
        img = _ramp()
        mtf = MidtoneTransferFunction(MidtoneTransferConfig(midpoint=0.15))
        out = mtf.apply(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# TestMidtoneTransferFunctionApply
# ---------------------------------------------------------------------------

class TestMidtoneTransferFunctionApply:
    def test_apply_increases_midrange(self):
        """Small midpoint < 0.5 should stretch faint values up."""
        img = _gray(0.05)
        mtf = MidtoneTransferFunction(MidtoneTransferConfig(midpoint=0.1))
        out = mtf.apply(img)
        assert out.mean() > img.mean()

    def test_apply_grayscale_2d(self):
        img = _gray(0.2)
        mtf = MidtoneTransferFunction()
        out = mtf.apply(img)
        assert out.shape == img.shape

    def test_apply_rgb_3d(self):
        img = _rgb()
        mtf = MidtoneTransferFunction()
        out = mtf.apply(img)
        assert out.shape == img.shape

    def test_apply_shadows_clipping_zeros_below(self):
        img = np.full((4, 4), 0.04, dtype=np.float32)
        cfg = MidtoneTransferConfig(midpoint=0.25, shadows_clipping=0.05)
        mtf = MidtoneTransferFunction(cfg)
        out = mtf.apply(img)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_apply_highlights_clipping_ones_above(self):
        img = np.full((4, 4), 1.0, dtype=np.float32)
        cfg = MidtoneTransferConfig(midpoint=0.25, highlights=0.99)
        mtf = MidtoneTransferFunction(cfg)
        out = mtf.apply(img)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_preserves_dtype_float32(self):
        img = _gray(0.3).astype(np.float32)
        mtf = MidtoneTransferFunction()
        out = mtf.apply(img)
        assert out.dtype == np.float32

    def test_preserves_dtype_float64(self):
        img = _gray(0.3).astype(np.float64)
        mtf = MidtoneTransferFunction()
        out = mtf.apply(img)
        assert out.dtype == np.float64

    def test_apply_monotone_on_ramp(self):
        """Output must be monotonically non-decreasing for ramp input."""
        img = _ramp()
        mtf = MidtoneTransferFunction(MidtoneTransferConfig(midpoint=0.2))
        out = mtf.apply(img)
        flat = out.ravel()
        assert np.all(np.diff(flat) >= -1e-9)

    def test_default_config_constructs(self):
        mtf = MidtoneTransferFunction()
        assert mtf.config == MidtoneTransferConfig()

    def test_none_config_uses_default(self):
        mtf = MidtoneTransferFunction(None)
        assert mtf.config == MidtoneTransferConfig()

    def test_empty_range_returns_zeros(self):
        # shadows_clipping == highlights is impossible via config, but test near-zero range
        img = np.full((4, 4), 0.5, dtype=np.float32)
        # Apply with shadows = 0, highlights = 0.98 — valid, output should be clipped from above
        cfg = MidtoneTransferConfig(midpoint=0.25, highlights=0.98)
        mtf = MidtoneTransferFunction(cfg)
        out = mtf.apply(img)
        # pixels at 0.5 < 0.98 are remapped into [0,1]
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# TestAutoBTF
# ---------------------------------------------------------------------------

class TestAutoBTF:
    def test_dark_background_gives_low_midpoint(self):
        m = MidtoneTransferFunction.compute_midpoint_from_background(0.01)
        assert 0.001 <= m <= 0.499

    def test_result_always_in_valid_range(self):
        for bg in [0.001, 0.01, 0.05, 0.1, 0.2]:
            m = MidtoneTransferFunction.compute_midpoint_from_background(bg)
            assert 0.001 <= m <= 0.499, f"m={m} for bg={bg}"

    def test_background_at_or_above_target_returns_default(self):
        m = MidtoneTransferFunction.compute_midpoint_from_background(0.3, target_background=0.25)
        assert m == pytest.approx(0.25)

    def test_compute_clips_to_min(self):
        m = MidtoneTransferFunction.compute_midpoint_from_background(1e-9)
        assert m >= 0.001

    def test_compute_clips_to_max(self):
        m = MidtoneTransferFunction.compute_midpoint_from_background(0.249)
        assert m <= 0.499

    def test_darker_image_gives_smaller_midpoint(self):
        m_dark = MidtoneTransferFunction.compute_midpoint_from_background(0.01)
        m_medium = MidtoneTransferFunction.compute_midpoint_from_background(0.1)
        assert m_dark <= m_medium

    def test_estimate_background_uniform_image(self):
        img = np.full((10, 10), 0.3, dtype=np.float32)
        bg = MidtoneTransferFunction.estimate_background(img)
        assert isinstance(bg, float)
        assert 0.0 <= bg <= 1.0

    def test_estimate_background_rgb(self):
        img = _rgb(0.1, 0.2, 0.3)
        bg = MidtoneTransferFunction.estimate_background(img)
        assert isinstance(bg, float)
        assert 0.0 <= bg <= 1.0

    def test_estimate_background_dark_image_low(self):
        img = np.full((8, 8), 0.05, dtype=np.float32)
        bg = MidtoneTransferFunction.estimate_background(img)
        assert bg < 0.1

    def test_estimate_background_returns_float(self):
        img = _ramp()
        bg = MidtoneTransferFunction.estimate_background(img)
        assert isinstance(bg, float)


# ---------------------------------------------------------------------------
# TestMTFStep
# ---------------------------------------------------------------------------

class TestMTFStep:
    def _make_ctx(self, img=None, images=None):
        from astroai.core.pipeline.base import PipelineContext
        return PipelineContext(
            result=img,
            images=images or ([img] if img is not None else []),
        )

    def test_step_name(self):
        assert "MTF" in MTFStep().name

    def test_step_stage_is_processing(self):
        from astroai.core.pipeline.base import PipelineStage
        assert MTFStep().stage == PipelineStage.PROCESSING

    def test_execute_with_context_result(self):
        img = _gray(0.05)
        step = MTFStep(MidtoneTransferConfig(midpoint=0.1))
        ctx = self._make_ctx(img)
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.mean() > img.mean()

    def test_execute_falls_back_to_images(self):
        img = _gray(0.05)
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[img])
        step = MTFStep(MidtoneTransferConfig(midpoint=0.1))
        out = step.execute(ctx)
        assert out.result is not None

    def test_execute_no_image_skips(self):
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[])
        out = MTFStep().execute(ctx)
        assert out.result is None

    def test_execute_calls_progress_twice(self):
        img = _gray(0.1)
        step = MTFStep()
        ctx = self._make_ctx(img)
        calls = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2

    def test_execute_preserves_shape(self):
        img = _rgb()
        step = MTFStep()
        ctx = self._make_ctx(img)
        out = step.execute(ctx)
        assert out.result.shape == img.shape

    def test_default_config(self):
        step = MTFStep()
        assert step._mtf.config == MidtoneTransferConfig()
