"""Tests for DenoiseStep pipeline integration."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.processing.denoise import DenoiseStep


def _make_noisy(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.RandomState(7)
    return rng.uniform(0.1, 0.9, (h, w)).astype(np.float64)


def _make_rgb(h: int = 64, w: int = 64) -> np.ndarray:
    mono = _make_noisy(h, w)
    return np.stack([mono, mono * 0.9, mono * 0.8], axis=-1)


@pytest.fixture()
def step() -> DenoiseStep:
    return DenoiseStep(strength=1.0)


class TestProperties:
    def test_name(self, step):
        assert step.name == "Denoise"

    def test_stage(self, step):
        assert step.stage == PipelineStage.PROCESSING


class TestExecution:
    def test_result_shape_preserved(self, step):
        ctx = PipelineContext(result=_make_noisy())
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (64, 64)

    def test_result_finite(self, step):
        ctx = PipelineContext(result=_make_noisy())
        out = step.execute(ctx)
        assert np.all(np.isfinite(out.result))

    def test_dtype_preserved(self, step):
        arr = _make_noisy().astype(np.float32)
        ctx = PipelineContext(result=arr)
        out = step.execute(ctx)
        assert out.result.dtype == np.float32

    def test_rgb_input(self, step):
        ctx = PipelineContext(result=_make_rgb())
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (64, 64, 3)

    def test_image_list(self, step):
        ctx = PipelineContext(images=[_make_noisy(), _make_noisy()])
        out = step.execute(ctx)
        assert len(out.images) == 2
        for img in out.images:
            assert np.all(np.isfinite(img))

    def test_empty_context_unchanged(self, step):
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.result is None
        assert out.images == []

    def test_strength_zero_returns_original(self):
        s = DenoiseStep(strength=0.0)
        arr = _make_noisy()
        ctx = PipelineContext(result=arr.copy())
        out = s.execute(ctx)
        np.testing.assert_allclose(out.result, arr, atol=1e-10)


class TestProgress:
    def test_progress_called_for_result(self, step):
        calls = []
        ctx = PipelineContext(result=_make_noisy())
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2

    def test_progress_called_per_image(self, step):
        calls = []
        ctx = PipelineContext(images=[_make_noisy(), _make_noisy()])
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2
