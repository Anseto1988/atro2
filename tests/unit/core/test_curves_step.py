"""Unit tests for CurvesStep tone-curve processing."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext
from astroai.processing.curves.pipeline_step import (
    CurvesStep,
    _apply_lut,
    _build_lut,
    _is_identity,
)

_IDENTITY = [(0.0, 0.0), (1.0, 1.0)]


class TestIsIdentity:
    def test_exact_identity(self) -> None:
        assert _is_identity([(0.0, 0.0), (1.0, 1.0)]) is True

    def test_non_identity_three_points(self) -> None:
        assert _is_identity([(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]) is False

    def test_shifted_output(self) -> None:
        assert _is_identity([(0.0, 0.1), (1.0, 1.0)]) is False


class TestBuildLut:
    def test_identity_lut(self) -> None:
        lut = _build_lut(_IDENTITY)
        assert lut.shape == (65536,)
        assert float(lut[0]) == pytest.approx(0.0, abs=0.001)
        assert float(lut[-1]) == pytest.approx(1.0, abs=0.001)

    def test_lut_clipped_to_unit(self) -> None:
        # S-curve that could overshoot
        pts = [(0.0, 0.0), (0.25, 0.1), (0.75, 0.9), (1.0, 1.0)]
        lut = _build_lut(pts)
        assert float(lut.min()) >= 0.0
        assert float(lut.max()) <= 1.0

    def test_deduplicate_x(self) -> None:
        # Duplicate x should not crash
        pts = [(0.0, 0.0), (0.5, 0.5), (0.5, 0.6), (1.0, 1.0)]
        lut = _build_lut(pts)
        assert lut.shape == (65536,)


class TestApplyLut:
    def test_identity_lut_preserves_values(self) -> None:
        lut = _build_lut(_IDENTITY)
        arr = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        result = _apply_lut(arr, lut)
        np.testing.assert_allclose(result, arr, atol=0.001)

    def test_dtype_preserved(self) -> None:
        lut = _build_lut(_IDENTITY)
        arr = np.ones((4, 4), dtype=np.float64) * 0.5
        result = _apply_lut(arr, lut)
        assert result.dtype == np.float64


class TestCurvesStep:
    def _make_rgb_image(self, value: float = 0.5) -> np.ndarray:
        return np.full((8, 8, 3), value, dtype=np.float32)

    def _make_gray_image(self, value: float = 0.5) -> np.ndarray:
        return np.full((8, 8), value, dtype=np.float32)

    def test_name(self) -> None:
        step = CurvesStep()
        assert step.name == "Kurven"

    def test_identity_curve_noop_rgb(self) -> None:
        image = self._make_rgb_image(0.6)
        ctx = PipelineContext(result=image.copy())
        step = CurvesStep(rgb_points=_IDENTITY)
        result_ctx = step.execute(ctx)
        np.testing.assert_allclose(result_ctx.result, image, atol=0.002)

    def test_identity_curve_noop_gray(self) -> None:
        image = self._make_gray_image(0.4)
        ctx = PipelineContext(result=image.copy())
        step = CurvesStep()
        result_ctx = step.execute(ctx)
        np.testing.assert_allclose(result_ctx.result, image, atol=0.002)

    def test_rgb_curve_brightens_all_channels(self) -> None:
        image = self._make_rgb_image(0.3)
        ctx = PipelineContext(result=image.copy())
        # Curve that lifts midtones
        pts = [(0.0, 0.0), (0.3, 0.5), (1.0, 1.0)]
        step = CurvesStep(rgb_points=pts)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        assert float(result_ctx.result.mean()) > 0.3

    def test_r_channel_curve_only_affects_red(self) -> None:
        image = np.full((8, 8, 3), 0.5, dtype=np.float32)
        ctx = PipelineContext(result=image.copy())
        # Curve that darkens (below diagonal)
        pts = [(0.0, 0.0), (0.5, 0.2), (1.0, 1.0)]
        step = CurvesStep(r_points=pts)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        r = result_ctx.result[..., 0].mean()
        g = result_ctx.result[..., 1].mean()
        b = result_ctx.result[..., 2].mean()
        assert r < 0.5
        assert g == pytest.approx(0.5, abs=0.002)
        assert b == pytest.approx(0.5, abs=0.002)

    def test_no_result_returns_context_unchanged(self) -> None:
        ctx = PipelineContext()
        step = CurvesStep()
        result_ctx = step.execute(ctx)
        assert result_ctx.result is None

    def test_output_clipped_to_unit(self) -> None:
        image = np.full((4, 4, 3), 0.9, dtype=np.float32)
        ctx = PipelineContext(result=image)
        # Aggressive brightening curve
        pts = [(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)]
        step = CurvesStep(rgb_points=pts)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        assert float(result_ctx.result.max()) <= 1.0

    def test_min_two_control_points(self) -> None:
        step = CurvesStep(rgb_points=[(0.0, 0.0), (1.0, 1.0)])
        image = self._make_rgb_image(0.5)
        ctx = PipelineContext(result=image)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None

    def test_max_ten_control_points(self) -> None:
        pts = [(i / 9, i / 9) for i in range(10)]
        step = CurvesStep(rgb_points=pts)
        image = self._make_rgb_image(0.5)
        ctx = PipelineContext(result=image)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None

    def test_grayscale_image_with_rgb_curve(self) -> None:
        """Line 93: 2D image + non-identity rgb_points → _apply_lut on 2D array."""
        image = self._make_gray_image(0.3)
        ctx = PipelineContext(result=image.copy())
        pts = [(0.0, 0.0), (0.3, 0.6), (1.0, 1.0)]
        step = CurvesStep(rgb_points=pts)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is not None
        assert result_ctx.result.ndim == 2
        assert float(result_ctx.result.mean()) > 0.3


class TestBuildLutEdgeCases:
    def test_single_unique_x_after_dedup_returns_identity(self) -> None:
        """Line 45: all points share same x → len(xs) < 2 after dedup → identity LUT."""
        pts = [(0.5, 0.0), (0.5, 0.5), (0.5, 1.0)]
        lut = _build_lut(pts)
        assert lut.shape == (65536,)
        assert float(lut[0]) == pytest.approx(0.0, abs=0.001)
        assert float(lut[-1]) == pytest.approx(1.0, abs=0.001)
