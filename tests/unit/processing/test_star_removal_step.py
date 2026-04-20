"""Tests for StarRemovalStep pipeline integration."""

from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.processing.stars import StarRemovalStep


def _make_starfield(h: int = 128, w: int = 128) -> np.ndarray:
    rng = np.random.RandomState(42)
    yy, xx = np.mgrid[0:h, 0:w]
    bg = 200.0 * np.exp(-((yy - h // 2) ** 2 + (xx - w // 2) ** 2) / (2 * 30**2))
    stars = np.zeros((h, w), dtype=np.float64)
    for _ in range(6):
        cy, cx = rng.randint(10, h - 10), rng.randint(10, w - 10)
        sigma = rng.uniform(1.5, 3.0)
        flux = rng.uniform(3000, 8000)
        stars += flux * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    return np.clip(bg + stars + 50, 0, 65535).astype(np.float64)


def _make_rgb_starfield(h: int = 128, w: int = 128) -> np.ndarray:
    mono = _make_starfield(h, w)
    return np.stack([mono, mono * 0.8, mono * 0.6], axis=-1)


@pytest.fixture()
def step() -> StarRemovalStep:
    return StarRemovalStep()


@pytest.fixture()
def context_with_result() -> PipelineContext:
    ctx = PipelineContext()
    ctx.result = _make_starfield()
    return ctx


@pytest.fixture()
def context_with_images() -> PipelineContext:
    ctx = PipelineContext()
    ctx.images = [_make_starfield()]
    return ctx


class TestBasicExecution:
    def test_execute_sets_starless(self, step, context_with_result):
        ctx = step.execute(context_with_result)
        assert ctx.starless_image is not None
        assert ctx.starless_image.shape == context_with_result.result.shape

    def test_execute_sets_star_mask(self, step, context_with_result):
        ctx = step.execute(context_with_result)
        assert ctx.star_mask is not None
        assert ctx.star_mask.dtype == np.float32

    def test_star_mask_is_2d(self, step, context_with_result):
        ctx = step.execute(context_with_result)
        assert ctx.star_mask.ndim == 2

    def test_star_mask_is_binary(self, step, context_with_result):
        ctx = step.execute(context_with_result)
        unique = np.unique(ctx.star_mask)
        assert all(v in (0.0, 1.0) for v in unique)

    def test_result_replaced_with_starless(self, step, context_with_result):
        original_max = context_with_result.result.max()
        ctx = step.execute(context_with_result)
        assert ctx.result.max() <= original_max

    def test_fallback_to_images(self, step, context_with_images):
        ctx = step.execute(context_with_images)
        assert ctx.starless_image is not None
        assert ctx.result is not None

    def test_empty_context_returns_unchanged(self, step):
        ctx = PipelineContext()
        result = step.execute(ctx)
        assert result.starless_image is None
        assert result.star_mask is None

    def test_rgb_input(self, step):
        ctx = PipelineContext(result=_make_rgb_starfield())
        out = step.execute(ctx)
        assert out.starless_image.ndim == 3
        assert out.star_mask.ndim == 2


class TestProperties:
    def test_name(self, step):
        assert step.name == "Star Removal"

    def test_stage(self, step):
        assert step.stage == PipelineStage.PROCESSING


class TestProgressCallback:
    def test_progress_callback_called(self, step, context_with_result):
        calls = []
        step.execute(context_with_result, progress=lambda p: calls.append(p))
        assert len(calls) == 2

    def test_progress_messages(self, step, context_with_result):
        calls = []
        step.execute(context_with_result, progress=lambda p: calls.append(p))
        assert "Removing" in calls[0].message
        assert "complete" in calls[1].message


class TestOnnxFallback:
    def test_fallback_when_model_unavailable(self):
        step = StarRemovalStep()
        ctx = PipelineContext(result=_make_starfield())
        out = step.execute(ctx)
        assert out.starless_image is not None
        assert out.star_mask is not None

    def test_onnx_path_nonexistent_falls_back(self):
        step = StarRemovalStep(onnx_model_path="/nonexistent/starnet.onnx")
        ctx = PipelineContext(result=_make_starfield())
        out = step.execute(ctx)
        assert out.starless_image is not None

    def test_onnx_path_accepted(self):
        step = StarRemovalStep(onnx_model_path="/some/path/starnet.onnx")
        ctx = PipelineContext(result=_make_starfield())
        out = step.execute(ctx)
        assert out.starless_image is not None


class TestCustomParameters:
    def test_custom_sigma(self):
        step = StarRemovalStep(detection_sigma=5.0)
        ctx = PipelineContext(result=_make_starfield())
        out = step.execute(ctx)
        assert out.starless_image is not None

    def test_custom_dilation(self):
        step = StarRemovalStep(mask_dilation=5)
        ctx = PipelineContext(result=_make_starfield())
        out = step.execute(ctx)
        assert out.star_mask is not None
