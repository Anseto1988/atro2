"""Tests for DeconvolutionStep pipeline integration and Deconvolver algorithm."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.processing.deconvolution import Deconvolver, DeconvolutionStep, gaussian_psf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blurred_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Synthetic blurred grayscale frame."""
    rng = np.random.RandomState(0)
    base = rng.uniform(0.2, 0.8, size=(h, w)).astype(np.float64)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(base, sigma=1.5)


def _make_rgb_frame(h: int = 64, w: int = 64) -> np.ndarray:
    mono = _make_blurred_frame(h, w)
    return np.stack([mono, mono * 0.9, mono * 0.7], axis=-1)


# ---------------------------------------------------------------------------
# gaussian_psf
# ---------------------------------------------------------------------------

class TestGaussianPsf:
    def test_shape(self):
        psf = gaussian_psf(size=5)
        assert psf.shape == (5, 5)

    def test_normalised(self):
        psf = gaussian_psf(size=7, sigma=2.0)
        assert abs(psf.sum() - 1.0) < 1e-12

    def test_symmetric(self):
        psf = gaussian_psf(size=5, sigma=1.0)
        np.testing.assert_allclose(psf, psf.T)


# ---------------------------------------------------------------------------
# Deconvolver (core algorithm)
# ---------------------------------------------------------------------------

class TestDeconvolver:
    def test_output_shape_grayscale(self):
        d = Deconvolver(iterations=3)
        img = _make_blurred_frame()
        out = d.deconvolve(img)
        assert out.shape == img.shape

    def test_output_shape_rgb(self):
        d = Deconvolver(iterations=3)
        img = _make_rgb_frame()
        out = d.deconvolve(img)
        assert out.shape == img.shape

    def test_output_dtype_preserved(self):
        d = Deconvolver(iterations=3)
        img = _make_blurred_frame().astype(np.float32)
        out = d.deconvolve(img)
        assert out.dtype == np.float32

    def test_clip_output_respects_input_range(self):
        d = Deconvolver(iterations=3, clip_output=True)
        img = _make_blurred_frame()
        out = d.deconvolve(img)
        assert out.max() <= img.max() + 1e-9

    def test_no_clip_can_exceed_input(self):
        d = Deconvolver(iterations=5, clip_output=False)
        img = _make_blurred_frame()
        out = d.deconvolve(img)
        assert out is not None  # just verify it runs

    def test_single_iteration_does_not_crash(self):
        d = Deconvolver(iterations=1)
        out = d.deconvolve(_make_blurred_frame())
        assert out is not None


# ---------------------------------------------------------------------------
# DeconvolutionStep — properties
# ---------------------------------------------------------------------------

@pytest.fixture()
def step() -> DeconvolutionStep:
    return DeconvolutionStep(iterations=3)


@pytest.fixture()
def context_with_result() -> PipelineContext:
    ctx = PipelineContext()
    ctx.result = _make_blurred_frame()
    return ctx


@pytest.fixture()
def context_with_images() -> PipelineContext:
    ctx = PipelineContext()
    ctx.images = [_make_blurred_frame(), _make_blurred_frame()]
    return ctx


class TestProperties:
    def test_name(self, step):
        assert step.name == "Deconvolution"

    def test_stage(self, step):
        assert step.stage == PipelineStage.PROCESSING


# ---------------------------------------------------------------------------
# DeconvolutionStep — basic execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    def test_result_shape_preserved(self, step, context_with_result):
        original_shape = context_with_result.result.shape
        ctx = step.execute(context_with_result)
        assert ctx.result.shape == original_shape

    def test_result_is_modified(self, step, context_with_result):
        original = context_with_result.result.copy()
        ctx = step.execute(context_with_result)
        assert not np.array_equal(ctx.result, original)

    def test_result_stays_finite(self, step, context_with_result):
        ctx = step.execute(context_with_result)
        assert np.all(np.isfinite(ctx.result))

    def test_processes_image_list(self, step, context_with_images):
        ctx = step.execute(context_with_images)
        assert len(ctx.images) == 2
        for img in ctx.images:
            assert np.all(np.isfinite(img))

    def test_empty_context_returns_unchanged(self, step):
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.result is None
        assert out.images == []

    def test_rgb_input(self, step):
        ctx = PipelineContext(result=_make_rgb_frame())
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.ndim == 3
        assert out.result.shape[2] == 3


# ---------------------------------------------------------------------------
# DeconvolutionStep — progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_progress_called_for_result(self, step, context_with_result):
        calls = []
        step.execute(context_with_result, progress=lambda p: calls.append(p))
        assert len(calls) == 2

    def test_progress_messages_for_result(self, step, context_with_result):
        calls = []
        step.execute(context_with_result, progress=lambda p: calls.append(p))
        assert "läuft" in calls[0].message or "Deconvolution" in calls[0].message
        assert "abgeschlossen" in calls[1].message or "complete" in calls[1].message.lower()

    def test_progress_called_per_image(self, step, context_with_images):
        calls = []
        step.execute(context_with_images, progress=lambda p: calls.append(p))
        assert len(calls) == 2  # one per image


# ---------------------------------------------------------------------------
# DeconvolutionStep — ONNX fallback
# ---------------------------------------------------------------------------

class TestOnnxFallback:
    def test_fallback_when_model_unavailable(self):
        s = DeconvolutionStep(iterations=3)
        ctx = PipelineContext(result=_make_blurred_frame())
        out = s.execute(ctx)
        assert out.result is not None

    def test_nonexistent_onnx_path_falls_back(self):
        s = DeconvolutionStep(iterations=3, onnx_model_path="/nonexistent/deconv.onnx")
        ctx = PipelineContext(result=_make_blurred_frame())
        out = s.execute(ctx)
        assert out.result is not None


# ---------------------------------------------------------------------------
# DeconvolutionStep — custom parameters
# ---------------------------------------------------------------------------

class TestCustomParameters:
    def test_custom_iterations(self):
        s = DeconvolutionStep(iterations=20)
        ctx = PipelineContext(result=_make_blurred_frame())
        out = s.execute(ctx)
        assert out.result is not None

    def test_custom_psf_sigma(self):
        s = DeconvolutionStep(iterations=3, psf_sigma=2.0)
        ctx = PipelineContext(result=_make_blurred_frame())
        out = s.execute(ctx)
        assert out.result is not None

    def test_larger_psf_size(self):
        s = DeconvolutionStep(iterations=3, psf_size=7)
        ctx = PipelineContext(result=_make_blurred_frame())
        out = s.execute(ctx)
        assert out.result is not None
