"""Unit tests for astroai.processing.stretch.asinh_stretcher."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.processing.stretch.asinh_stretcher import (
    AsinHConfig,
    AsinHStep,
    AsinHStretcher,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_image(r: float = 0.5, g: float = 0.5, b: float = 0.5, size: int = 4) -> np.ndarray:
    """Create a solid-colour RGB image."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def _gray_image(value: float = 0.5, size: int = 4) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def _asinh_ref(x: float, beta: float) -> float:
    """Reference single-value arcsinh stretch."""
    return float(np.arcsinh(beta * x) / np.arcsinh(beta))


# ---------------------------------------------------------------------------
# TestAsinHConfig
# ---------------------------------------------------------------------------

class TestAsinHConfig:
    def test_defaults(self):
        cfg = AsinHConfig()
        assert cfg.stretch_factor == pytest.approx(1.0)
        assert cfg.black_point == pytest.approx(0.0)
        assert cfg.linked_channels is True

    def test_custom_values_stored(self):
        cfg = AsinHConfig(stretch_factor=10.0, black_point=0.05, linked_channels=False)
        assert cfg.stretch_factor == pytest.approx(10.0)
        assert cfg.black_point == pytest.approx(0.05)
        assert cfg.linked_channels is False

    def test_is_identity_defaults(self):
        assert AsinHConfig().is_identity()

    def test_is_identity_true_near_one(self):
        # Within atol=1e-6
        cfg = AsinHConfig(stretch_factor=1.0 + 5e-7, black_point=3e-7)
        assert cfg.is_identity()

    def test_is_identity_false_stretch_factor(self):
        assert not AsinHConfig(stretch_factor=2.0).is_identity()

    def test_is_identity_false_black_point(self):
        assert not AsinHConfig(black_point=0.01).is_identity()

    def test_is_identity_false_outside_atol(self):
        cfg = AsinHConfig(stretch_factor=1.0 + 2e-6)
        assert not cfg.is_identity()

    def test_as_dict_keys(self):
        d = AsinHConfig().as_dict()
        assert set(d.keys()) == {"stretch_factor", "black_point", "linked_channels"}

    def test_as_dict_values(self):
        cfg = AsinHConfig(stretch_factor=5.0, black_point=0.1, linked_channels=False)
        d = cfg.as_dict()
        assert d["stretch_factor"] == pytest.approx(5.0)
        assert d["black_point"] == pytest.approx(0.1)
        assert d["linked_channels"] is False

    def test_validation_stretch_factor_zero_raises(self):
        with pytest.raises(ValueError, match="stretch_factor"):
            AsinHConfig(stretch_factor=0.0)

    def test_validation_stretch_factor_negative_raises(self):
        with pytest.raises(ValueError, match="stretch_factor"):
            AsinHConfig(stretch_factor=-1.0)

    def test_validation_black_point_negative_raises(self):
        with pytest.raises(ValueError, match="black_point"):
            AsinHConfig(black_point=-0.01)

    def test_validation_black_point_above_half_raises(self):
        with pytest.raises(ValueError, match="black_point"):
            AsinHConfig(black_point=0.51)

    def test_validation_black_point_exactly_half_ok(self):
        cfg = AsinHConfig(black_point=0.5)
        assert cfg.black_point == pytest.approx(0.5)

    def test_frozen_immutable(self):
        cfg = AsinHConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.stretch_factor = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestAsinHStretcherIdentity
# ---------------------------------------------------------------------------

class TestAsinHStretcherIdentity:
    def test_identity_grayscale_approx_equal(self):
        img = _gray_image(0.5)
        stretcher = AsinHStretcher(AsinHConfig(stretch_factor=1.0, black_point=0.0))
        out = stretcher.stretch(img)
        # arcsinh(1.0 * 0.5) / arcsinh(1.0) is not exactly 0.5 but close
        expected = _asinh_ref(0.5, 1.0)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_identity_rgb_approx_equal(self):
        img = _rgb_image(0.3, 0.5, 0.7)
        stretcher = AsinHStretcher(AsinHConfig(stretch_factor=1.0, black_point=0.0))
        out = stretcher.stretch(img)
        np.testing.assert_allclose(
            out[:, :, 0], _asinh_ref(0.3, 1.0), atol=1e-5
        )
        np.testing.assert_allclose(
            out[:, :, 1], _asinh_ref(0.5, 1.0), atol=1e-5
        )
        np.testing.assert_allclose(
            out[:, :, 2], _asinh_ref(0.7, 1.0), atol=1e-5
        )

    def test_zero_input_gives_zero_output(self):
        img = np.zeros((4, 4), dtype=np.float32)
        out = AsinHStretcher().stretch(img)
        np.testing.assert_array_equal(out, 0.0)

    def test_one_input_gives_one_output(self):
        img = np.ones((4, 4), dtype=np.float32)
        out = AsinHStretcher().stretch(img)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# TestAsinHStretcherApply
# ---------------------------------------------------------------------------

class TestAsinHStretcherApply:
    def test_known_value_grayscale(self):
        """arcsinh(5 * 0.5) / arcsinh(5) for a single pixel."""
        beta = 5.0
        x = 0.5
        expected = _asinh_ref(x, beta)
        img = np.full((4, 4), x, dtype=np.float32)
        out = AsinHStretcher(AsinHConfig(stretch_factor=beta)).stretch(img)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_known_value_rgb_linked(self):
        """arcsinh(5 * 0.5) / arcsinh(5) applied uniformly for linked RGB."""
        beta = 5.0
        x = 0.5
        expected = _asinh_ref(x, beta)
        img = _rgb_image(x, x, x)
        out = AsinHStretcher(AsinHConfig(stretch_factor=beta, linked_channels=True)).stretch(img)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_large_beta_compresses_highlights(self):
        """With large β, bright values are compressed closer to 1.0."""
        beta = 500.0
        img_bright = np.full((4, 4), 0.9, dtype=np.float32)
        img_medium = np.full((4, 4), 0.5, dtype=np.float32)
        out_bright = AsinHStretcher(AsinHConfig(stretch_factor=beta)).stretch(img_bright)
        out_medium = AsinHStretcher(AsinHConfig(stretch_factor=beta)).stretch(img_medium)
        # Both should be close to 1.0 with large beta, bright > medium
        assert out_bright.mean() > out_medium.mean()

    def test_output_bounded_01(self):
        """All output values must lie in [0, 1]."""
        img = np.random.default_rng(42).uniform(0.0, 1.0, (8, 8, 3)).astype(np.float32)
        out = AsinHStretcher(AsinHConfig(stretch_factor=50.0)).stretch(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-7

    def test_monotone_increasing(self):
        """Higher input pixel values must map to higher output values."""
        xs = np.linspace(0.0, 1.0, 20, dtype=np.float64).reshape(1, 20)
        out = AsinHStretcher(AsinHConfig(stretch_factor=10.0)).stretch(xs)
        diffs = np.diff(out[0])
        assert np.all(diffs >= 0)


# ---------------------------------------------------------------------------
# TestAsinHBlackPoint
# ---------------------------------------------------------------------------

class TestAsinHBlackPoint:
    def test_black_point_shifts_output(self):
        """With black_point > 0 the mid-value pixel should map to a lower output."""
        img = np.full((4, 4), 0.3, dtype=np.float32)
        out_no_bp = AsinHStretcher(AsinHConfig(stretch_factor=1.0, black_point=0.0)).stretch(img)
        out_with_bp = AsinHStretcher(AsinHConfig(stretch_factor=1.0, black_point=0.1)).stretch(img)
        assert out_with_bp.mean() < out_no_bp.mean()

    def test_black_point_below_offset_clips_to_zero(self):
        """Pixels below the black point are clipped to 0 after subtraction."""
        img = np.full((4, 4), 0.05, dtype=np.float32)
        out = AsinHStretcher(AsinHConfig(stretch_factor=5.0, black_point=0.1)).stretch(img)
        np.testing.assert_array_equal(out, 0.0)

    def test_black_point_grayscale(self):
        bp = 0.2
        x = 0.5
        expected_input = max(0.0, x - bp)
        expected_out = _asinh_ref(expected_input, 1.0)
        img = np.full((4, 4), x, dtype=np.float32)
        out = AsinHStretcher(AsinHConfig(stretch_factor=1.0, black_point=bp)).stretch(img)
        np.testing.assert_allclose(out, expected_out, atol=1e-5)

    def test_black_point_never_negative_output(self):
        img = np.full((4, 4, 3), 0.1, dtype=np.float32)
        out = AsinHStretcher(AsinHConfig(stretch_factor=10.0, black_point=0.3)).stretch(img)
        assert out.min() >= 0.0


# ---------------------------------------------------------------------------
# TestAsinHLinked
# ---------------------------------------------------------------------------

class TestAsinHLinked:
    def test_linked_same_result_all_channels(self):
        """When linked=True, all channels get the same stretch."""
        img = _rgb_image(0.3, 0.5, 0.7)
        out = AsinHStretcher(AsinHConfig(stretch_factor=5.0, linked_channels=True)).stretch(img)
        # Each channel uses the same beta, so values should match arcsinh of respective channel
        np.testing.assert_allclose(out[:, :, 0], _asinh_ref(0.3, 5.0), atol=1e-5)
        np.testing.assert_allclose(out[:, :, 1], _asinh_ref(0.5, 5.0), atol=1e-5)
        np.testing.assert_allclose(out[:, :, 2], _asinh_ref(0.7, 5.0), atol=1e-5)

    def test_unlinked_same_formula_per_channel(self):
        """When linked=False, each channel is processed independently with same beta."""
        img = _rgb_image(0.3, 0.5, 0.7)
        out = AsinHStretcher(AsinHConfig(stretch_factor=5.0, linked_channels=False)).stretch(img)
        # Same beta for all channels → same formula → should match linked result
        np.testing.assert_allclose(out[:, :, 0], _asinh_ref(0.3, 5.0), atol=1e-5)
        np.testing.assert_allclose(out[:, :, 1], _asinh_ref(0.5, 5.0), atol=1e-5)
        np.testing.assert_allclose(out[:, :, 2], _asinh_ref(0.7, 5.0), atol=1e-5)

    def test_linked_vs_unlinked_same_uniform_image(self):
        """For uniform colour images, linked and unlinked should yield identical output."""
        img = _rgb_image(0.4, 0.4, 0.4)
        beta = 8.0
        out_linked = AsinHStretcher(AsinHConfig(stretch_factor=beta, linked_channels=True)).stretch(img)
        out_unlinked = AsinHStretcher(AsinHConfig(stretch_factor=beta, linked_channels=False)).stretch(img)
        np.testing.assert_allclose(out_linked, out_unlinked, atol=1e-6)


# ---------------------------------------------------------------------------
# TestAsinHEdgeCases
# ---------------------------------------------------------------------------

class TestAsinHEdgeCases:
    def test_black_image_stays_black(self):
        img = np.zeros((8, 8, 3), dtype=np.float32)
        out = AsinHStretcher(AsinHConfig(stretch_factor=100.0)).stretch(img)
        np.testing.assert_array_equal(out, 0.0)

    def test_white_image_stays_white(self):
        img = np.ones((8, 8, 3), dtype=np.float32)
        out = AsinHStretcher(AsinHConfig(stretch_factor=100.0)).stretch(img)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_very_small_beta_linear_passthrough_grayscale(self):
        """For β < 1e-9 the output should equal the input (linear passthrough)."""
        img = _gray_image(0.4)
        out = AsinHStretcher(AsinHConfig(stretch_factor=1e-12)).stretch(img)
        np.testing.assert_allclose(out, 0.4, atol=1e-5)

    def test_very_small_beta_linear_passthrough_rgb(self):
        img = _rgb_image(0.2, 0.5, 0.8)
        out = AsinHStretcher(AsinHConfig(stretch_factor=1e-12)).stretch(img)
        np.testing.assert_allclose(out[:, :, 0], 0.2, atol=1e-5)
        np.testing.assert_allclose(out[:, :, 1], 0.5, atol=1e-5)
        np.testing.assert_allclose(out[:, :, 2], 0.8, atol=1e-5)

    def test_dtype_float32_preserved(self):
        img = _rgb_image(0.3, 0.5, 0.7).astype(np.float32)
        out = AsinHStretcher(AsinHConfig(stretch_factor=5.0)).stretch(img)
        assert out.dtype == np.float32

    def test_dtype_float64_preserved(self):
        img = _rgb_image(0.3, 0.5, 0.7).astype(np.float64)
        out = AsinHStretcher(AsinHConfig(stretch_factor=5.0)).stretch(img)
        assert out.dtype == np.float64

    def test_grayscale_2d_supported(self):
        img = _gray_image(0.6)
        out = AsinHStretcher(AsinHConfig(stretch_factor=3.0)).stretch(img)
        assert out.ndim == 2
        np.testing.assert_allclose(out, _asinh_ref(0.6, 3.0), atol=1e-5)

    def test_output_clipped_upper(self):
        """Any intermediate value > 1.0 must be clipped."""
        img = np.full((4, 4, 3), 0.99, dtype=np.float32)
        out = AsinHStretcher(AsinHConfig(stretch_factor=1000.0)).stretch(img)
        assert out.max() <= 1.0 + 1e-7

    def test_output_clipped_lower(self):
        """Any intermediate value < 0.0 must be clipped."""
        img = np.zeros((4, 4, 3), dtype=np.float32)
        out = AsinHStretcher(AsinHConfig(stretch_factor=5.0, black_point=0.3)).stretch(img)
        assert out.min() >= 0.0


# ---------------------------------------------------------------------------
# TestAsinHStep
# ---------------------------------------------------------------------------

class TestAsinHStep:
    def _make_context(self, img, images=None):
        from astroai.core.pipeline.base import PipelineContext
        return PipelineContext(
            result=img,
            images=images or ([img] if img is not None else []),
        )

    def test_step_name(self):
        step = AsinHStep()
        assert step.name == "Arcsinh Stretch"

    def test_step_stage_is_processing(self):
        from astroai.core.pipeline.base import PipelineStage
        step = AsinHStep()
        assert step.stage == PipelineStage.PROCESSING

    def test_execute_uses_context_result(self):
        img = _rgb_image(0.4, 0.5, 0.6)
        step = AsinHStep(AsinHConfig(stretch_factor=5.0))
        ctx = self._make_context(img)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        np.testing.assert_allclose(
            result_ctx.result[:, :, 0], _asinh_ref(0.4, 5.0), atol=1e-5
        )

    def test_execute_falls_back_to_images(self):
        img = _rgb_image(0.3, 0.3, 0.3)
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[img])
        step = AsinHStep(AsinHConfig(stretch_factor=3.0))
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        np.testing.assert_allclose(
            result_ctx.result[:, :, 0], _asinh_ref(0.3, 3.0), atol=1e-5
        )

    def test_execute_no_image_skips(self):
        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext(result=None, images=[])
        step = AsinHStep()
        result_ctx = step.execute(ctx)
        assert result_ctx.result is None

    def test_execute_sets_context_result(self):
        img = _rgb_image(0.2, 0.3, 0.4)
        step = AsinHStep()
        ctx = self._make_context(img)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None

    def test_execute_calls_progress_twice(self):
        img = _rgb_image(0.2, 0.3, 0.4)
        step = AsinHStep(AsinHConfig(stretch_factor=2.0))
        ctx = self._make_context(img)
        calls = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2

    def test_execute_grayscale(self):
        img = _gray_image(0.5)
        step = AsinHStep(AsinHConfig(stretch_factor=10.0))
        ctx = self._make_context(img)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        assert result_ctx.result.ndim == 2

    def test_default_step_uses_default_config(self):
        step = AsinHStep()
        assert step._stretcher.config == AsinHConfig()
