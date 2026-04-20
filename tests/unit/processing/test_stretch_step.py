"""Tests for StretchStep pipeline integration."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.processing.stretch import StretchStep


def _make_linear(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.RandomState(3)
    return (rng.uniform(0.03, 0.12, (h, w))).astype(np.float64)


def _make_rgb(h: int = 64, w: int = 64) -> np.ndarray:
    mono = _make_linear(h, w)
    return np.stack([mono, mono * 0.95, mono * 0.85], axis=-1)


@pytest.fixture()
def step() -> StretchStep:
    return StretchStep()


class TestProperties:
    def test_name(self, step):
        assert step.name == "Stretch"

    def test_stage(self, step):
        assert step.stage == PipelineStage.PROCESSING


class TestExecution:
    def test_result_shape_preserved(self, step):
        ctx = PipelineContext(result=_make_linear())
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (64, 64)

    def test_result_in_0_1(self, step):
        ctx = PipelineContext(result=_make_linear())
        out = step.execute(ctx)
        assert out.result.min() >= 0.0
        assert out.result.max() <= 1.0

    def test_result_finite(self, step):
        ctx = PipelineContext(result=_make_linear())
        out = step.execute(ctx)
        assert np.all(np.isfinite(out.result))

    def test_dtype_preserved(self, step):
        arr = _make_linear().astype(np.float32)
        ctx = PipelineContext(result=arr)
        out = step.execute(ctx)
        assert out.result.dtype == np.float32

    def test_rgb_input(self, step):
        ctx = PipelineContext(result=_make_rgb())
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (64, 64, 3)

    def test_image_list(self, step):
        ctx = PipelineContext(images=[_make_linear(), _make_linear()])
        out = step.execute(ctx)
        assert len(out.images) == 2
        for img in out.images:
            assert np.all(np.isfinite(img))

    def test_empty_context_unchanged(self, step):
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.result is None
        assert out.images == []

    def test_custom_target_background(self):
        s = StretchStep(target_background=0.15)
        ctx = PipelineContext(result=_make_linear())
        out = s.execute(ctx)
        assert out.result is not None
        assert np.all(np.isfinite(out.result))

    def test_independent_channels(self):
        s = StretchStep(linked_channels=False)
        ctx = PipelineContext(result=_make_rgb())
        out = s.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (64, 64, 3)


class TestProgress:
    def test_progress_called_for_result(self, step):
        calls = []
        ctx = PipelineContext(result=_make_linear())
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2

    def test_progress_called_per_image(self, step):
        calls = []
        ctx = PipelineContext(images=[_make_linear(), _make_linear()])
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 2
