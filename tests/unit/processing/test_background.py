"""Unit tests for background extraction and gradient removal."""

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.processing.background import (
    BackgroundExtractor,
    BackgroundRemovalStep,
    GradientRemover,
    ModelMethod,
)


np.random.seed(42)


def _make_gradient_frame(h: int = 128, w: int = 128) -> NDArray[np.float64]:
    """Frame with a strong linear gradient (simulating light pollution)."""
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = 500.0 + 2.0 * xx + 1.5 * yy
    return gradient.astype(np.float64)


def _make_signal_with_gradient(h: int = 128, w: int = 128) -> NDArray[np.float64]:
    """Nebula signal on top of a gradient background."""
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = 300.0 + 3.0 * xx + 2.0 * yy
    nebula = 800.0 * np.exp(
        -((yy - h // 2) ** 2 + (xx - w // 2) ** 2) / (2 * 20 ** 2)
    )
    return (gradient + nebula).astype(np.float64)


def _make_uniform_frame(h: int = 64, w: int = 64) -> NDArray[np.float64]:
    return np.full((h, w), 1000.0, dtype=np.float64)


def _make_quadratic_gradient(h: int = 128, w: int = 128) -> NDArray[np.float64]:
    yy, xx = np.mgrid[0:h, 0:w]
    ny = yy.astype(np.float64) / h
    nx = xx.astype(np.float64) / w
    return (200.0 + 500.0 * ny**2 + 300.0 * nx**2).astype(np.float64)


# --- BackgroundExtractor tests ---


class TestBackgroundExtractor:
    def test_extract_returns_same_shape(self):
        frame = _make_gradient_frame()
        extractor = BackgroundExtractor(tile_size=32)
        bg = extractor.extract(frame)
        assert bg.shape == frame.shape

    def test_extract_polynomial_smooth(self):
        frame = _make_gradient_frame()
        extractor = BackgroundExtractor(
            tile_size=32, method=ModelMethod.POLYNOMIAL, poly_degree=1
        )
        bg = extractor.extract(frame)
        residual = np.abs(frame - bg)
        assert residual.mean() < 50.0

    def test_extract_rbf_smooth(self):
        frame = _make_gradient_frame()
        extractor = BackgroundExtractor(tile_size=32, method=ModelMethod.RBF)
        bg = extractor.extract(frame)
        residual = np.abs(frame - bg)
        assert residual.mean() < 50.0

    def test_extract_rgb_frame(self):
        h, w = 64, 64
        rgb = np.random.rand(h, w, 3).astype(np.float64) * 1000.0 + 500.0
        extractor = BackgroundExtractor(tile_size=16)
        bg = extractor.extract(rgb)
        assert bg.shape == (h, w, 3)

    def test_extract_uniform_returns_constant(self):
        frame = _make_uniform_frame()
        extractor = BackgroundExtractor(tile_size=16)
        bg = extractor.extract(frame)
        assert np.std(bg) < 10.0

    def test_extract_quadratic_with_higher_degree(self):
        frame = _make_quadratic_gradient()
        extractor = BackgroundExtractor(
            tile_size=32, method=ModelMethod.POLYNOMIAL, poly_degree=3
        )
        bg = extractor.extract(frame)
        residual = np.abs(frame - bg)
        assert residual.mean() < 30.0


# --- GradientRemover tests ---


class TestGradientRemover:
    def test_remove_returns_same_shape_dtype(self):
        frame = _make_gradient_frame()
        remover = GradientRemover()
        result = remover.remove(frame)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_remove_reduces_gradient(self):
        frame = _make_gradient_frame()
        remover = GradientRemover(
            extractor=BackgroundExtractor(tile_size=32, method=ModelMethod.POLYNOMIAL)
        )
        result = remover.remove(frame)
        original_std = float(np.std(frame))
        corrected_std = float(np.std(result))
        assert corrected_std < original_std * 0.5

    def test_remove_preserves_median(self):
        frame = _make_signal_with_gradient()
        remover = GradientRemover(preserve_median=True)
        result = remover.remove(frame)
        median_diff = abs(float(np.median(result)) - float(np.median(frame)))
        assert median_diff < 50.0

    def test_remove_no_negative_with_clip(self):
        frame = _make_gradient_frame()
        remover = GradientRemover(clip_negative=True)
        result = remover.remove(frame)
        assert result.min() >= 0.0

    def test_remove_batch_processes_all(self):
        frames = [_make_gradient_frame(64, 64) for _ in range(3)]
        remover = GradientRemover()
        results = remover.remove_batch(frames)
        assert len(results) == 3
        for r in results:
            assert r.shape == (64, 64)

    def test_extract_background_preview(self):
        frame = _make_signal_with_gradient()
        remover = GradientRemover()
        bg = remover.extract_background(frame)
        assert bg.shape == frame.shape
        assert bg.dtype == frame.dtype


# --- BackgroundRemovalStep (Pipeline integration) tests ---


class TestBackgroundRemovalStep:
    def test_step_name(self):
        step = BackgroundRemovalStep()
        assert step.name == "Background Removal"

    def test_step_processes_result(self):
        from astroai.core.pipeline.base import PipelineContext

        frame = _make_signal_with_gradient()
        ctx = PipelineContext(result=frame)
        step = BackgroundRemovalStep(tile_size=32)
        ctx = step.execute(ctx)
        assert ctx.result is not None
        assert ctx.result.shape == frame.shape
        corrected_std = float(np.std(ctx.result))
        assert corrected_std < float(np.std(frame))

    def test_step_processes_image_list(self):
        from astroai.core.pipeline.base import PipelineContext

        frames = [_make_gradient_frame(64, 64) for _ in range(2)]
        ctx = PipelineContext(images=frames)
        step = BackgroundRemovalStep(tile_size=16)
        ctx = step.execute(ctx)
        assert len(ctx.images) == 2
        for img in ctx.images:
            assert img.shape == (64, 64)
