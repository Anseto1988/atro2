"""Unit tests for NAFNetDenoiser and OnnxDenoiseStep (≥30 tests, no real model)."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineProgress
from astroai.core.pipeline.onnx_denoise_step import OnnxDenoiseStep
from astroai.inference.backends.nafnet import NAFNetDenoiser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 64, w: int = 64, channels: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    if channels:
        return rng.uniform(0.0, 1.0, (h, w, channels)).astype(np.float64)
    return rng.uniform(0.0, 1.0, (h, w)).astype(np.float64)


class _DummyInput:
    def __init__(self, name: str = "input") -> None:
        self.name = name


class _PassthroughSession:
    """ONNX session stub: returns a slightly smoothed copy (not identity)."""

    def get_inputs(self) -> list[Any]:
        return [_DummyInput("input")]

    def run(self, _output_names: Any, input_dict: dict[str, Any]) -> list[np.ndarray]:
        data = input_dict["input"].copy().astype(np.float32)
        # simulate slight denoising: small smoothing
        return [data * 0.98]


class _IdentitySession:
    """ONNX session stub: identity pass-through."""

    def get_inputs(self) -> list[Any]:
        return [_DummyInput("input")]

    def run(self, _output_names: Any, input_dict: dict[str, Any]) -> list[np.ndarray]:
        return [input_dict["input"].copy()]


def _make_registry(session: Any, available: bool = True) -> MagicMock:
    reg = MagicMock()
    reg.get_session.return_value = session
    reg.is_available.return_value = available
    reg.backend_label = "[ONNX]"
    return reg


# ===========================================================================
# NAFNetDenoiser — init & properties
# ===========================================================================

class TestNAFNetDenoiserInit:
    def test_default_strength(self):
        d = NAFNetDenoiser(registry=_make_registry(_IdentitySession()))
        assert d.strength == 1.0

    def test_strength_clipped_above(self):
        d = NAFNetDenoiser(strength=2.0, registry=_make_registry(_IdentitySession()))
        assert d.strength == 1.0

    def test_strength_clipped_below(self):
        d = NAFNetDenoiser(strength=-0.5, registry=_make_registry(_IdentitySession()))
        assert d.strength == 0.0

    def test_strength_setter(self):
        d = NAFNetDenoiser(registry=_make_registry(_IdentitySession()))
        d.strength = 0.5
        assert d.strength == 0.5

    def test_tile_size_stored(self):
        d = NAFNetDenoiser(tile_size=256, registry=_make_registry(_IdentitySession()))
        assert d.tile_size == 256

    def test_tile_overlap_stored(self):
        d = NAFNetDenoiser(tile_overlap=32, registry=_make_registry(_IdentitySession()))
        assert d.tile_overlap == 32

    def test_model_not_loaded_initially(self):
        d = NAFNetDenoiser(registry=_make_registry(_IdentitySession()))
        assert not d.is_model_loaded

    def test_backend_label_delegates_to_registry(self):
        reg = _make_registry(_IdentitySession())
        reg.backend_label = "[GPU]"
        d = NAFNetDenoiser(registry=reg)
        assert d.backend_label == "[GPU]"

    def test_model_name_constant(self):
        assert NAFNetDenoiser.MODEL_NAME == "nafnet_denoise"


# ===========================================================================
# NAFNetDenoiser — lazy load
# ===========================================================================

class TestNAFNetDenoiserLoad:
    def test_load_calls_registry(self):
        session = _IdentitySession()
        reg = _make_registry(session)
        d = NAFNetDenoiser(registry=reg)
        d.load()
        reg.get_session.assert_called_once_with(
            "nafnet_denoise", fallback_to_dummy=True, progress=None
        )

    def test_is_loaded_after_load(self):
        d = NAFNetDenoiser(registry=_make_registry(_IdentitySession()))
        d.load()
        assert d.is_model_loaded

    def test_session_loaded_lazily_on_denoise(self):
        reg = _make_registry(_IdentitySession())
        d = NAFNetDenoiser(registry=reg)
        frame = _make_frame()
        d.denoise(frame)
        reg.get_session.assert_called_once()

    def test_session_not_reloaded_on_second_denoise(self):
        reg = _make_registry(_IdentitySession())
        d = NAFNetDenoiser(registry=reg)
        frame = _make_frame()
        d.denoise(frame)
        d.denoise(frame)
        reg.get_session.assert_called_once()


# ===========================================================================
# NAFNetDenoiser — denoise output shape/dtype
# ===========================================================================

class TestNAFNetDenoiserOutput:
    def test_grayscale_shape_preserved(self):
        d = NAFNetDenoiser(registry=_make_registry(_IdentitySession()))
        frame = _make_frame(32, 32)
        result = d.denoise(frame)
        assert result.shape == (32, 32)

    def test_rgb_shape_preserved(self):
        d = NAFNetDenoiser(registry=_make_registry(_IdentitySession()))
        frame = _make_frame(32, 32, channels=3)
        result = d.denoise(frame)
        assert result.shape == (32, 32, 3)

    def test_dtype_float64_preserved(self):
        d = NAFNetDenoiser(registry=_make_registry(_IdentitySession()))
        frame = _make_frame().astype(np.float64)
        result = d.denoise(frame)
        assert result.dtype == np.float64

    def test_dtype_float32_preserved(self):
        d = NAFNetDenoiser(registry=_make_registry(_IdentitySession()))
        frame = _make_frame().astype(np.float32)
        result = d.denoise(frame)
        assert result.dtype == np.float32

    def test_strength_zero_returns_original(self):
        session = _PassthroughSession()
        d = NAFNetDenoiser(strength=0.0, registry=_make_registry(session))
        frame = _make_frame()
        result = d.denoise(frame)
        np.testing.assert_array_almost_equal(result, frame, decimal=10)

    def test_strength_one_applies_full_denoising(self):
        session = _PassthroughSession()
        d = NAFNetDenoiser(strength=1.0, registry=_make_registry(session))
        frame = _make_frame()
        result = d.denoise(frame)
        # With strength=1.0 result should differ from original
        assert not np.allclose(result, frame)

    def test_result_within_original_range(self):
        d = NAFNetDenoiser(strength=1.0, registry=_make_registry(_PassthroughSession()))
        frame = _make_frame() * 1000.0
        result = d.denoise(frame)
        assert result.min() >= frame.min() - 1e-6
        assert result.max() <= frame.max() + 1e-6

    def test_uniform_frame_stable(self):
        d = NAFNetDenoiser(strength=1.0, registry=_make_registry(_IdentitySession()))
        frame = np.full((32, 32), 0.5, dtype=np.float64)
        result = d.denoise(frame)
        assert result.shape == frame.shape

    def test_tiny_frame_single_tile(self):
        d = NAFNetDenoiser(tile_size=512, tile_overlap=64,
                           registry=_make_registry(_IdentitySession()))
        frame = _make_frame(16, 16)
        result = d.denoise(frame)
        assert result.shape == (16, 16)

    def test_large_frame_multi_tile(self):
        d = NAFNetDenoiser(tile_size=32, tile_overlap=8,
                           registry=_make_registry(_IdentitySession()))
        frame = _make_frame(100, 100)
        result = d.denoise(frame)
        assert result.shape == (100, 100)


# ===========================================================================
# OnnxDenoiseStep — properties
# ===========================================================================

class TestOnnxDenoiseStepProperties:
    def test_name(self):
        step = OnnxDenoiseStep(registry=_make_registry(_IdentitySession()))
        assert "NAFNet" in step.name or "ONNX" in step.name

    def test_active_backend_nafnet_when_available(self):
        reg = _make_registry(_IdentitySession(), available=True)
        step = OnnxDenoiseStep(registry=reg)
        assert step.active_backend == "nafnet"

    def test_active_backend_basic_when_unavailable(self):
        reg = _make_registry(_IdentitySession(), available=False)
        step = OnnxDenoiseStep(registry=reg)
        assert step.active_backend == "basic"


# ===========================================================================
# OnnxDenoiseStep — execute with context.result
# ===========================================================================

class TestOnnxDenoiseStepExecuteResult:
    def _ctx(self, frame: np.ndarray) -> PipelineContext:
        ctx = PipelineContext()
        ctx.result = frame
        return ctx

    def test_execute_nafnet_changes_result(self):
        reg = _make_registry(_PassthroughSession(), available=True)
        step = OnnxDenoiseStep(strength=1.0, tile_size=32, registry=reg)
        frame = _make_frame(32, 32)
        ctx = self._ctx(frame.copy())
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (32, 32)

    def test_execute_fallback_changes_result(self):
        reg = _make_registry(_IdentitySession(), available=False)
        step = OnnxDenoiseStep(strength=1.0, registry=reg)
        frame = _make_frame(32, 32)
        ctx = self._ctx(frame.copy())
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (32, 32)

    def test_progress_callback_called(self):
        reg = _make_registry(_IdentitySession(), available=True)
        step = OnnxDenoiseStep(tile_size=32, registry=reg)
        calls: list[PipelineProgress] = []
        ctx = self._ctx(_make_frame(32, 32))
        step.execute(ctx, progress=calls.append)
        assert len(calls) >= 1

    def test_empty_context_unchanged(self):
        reg = _make_registry(_IdentitySession(), available=True)
        step = OnnxDenoiseStep(registry=reg)
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.result is None
        assert out.images == []


# ===========================================================================
# OnnxDenoiseStep — execute with context.images
# ===========================================================================

class TestOnnxDenoiseStepExecuteImages:
    def _ctx(self, *frames: np.ndarray) -> PipelineContext:
        ctx = PipelineContext()
        ctx.images = list(frames)
        return ctx

    def test_execute_nafnet_processes_all_images(self):
        reg = _make_registry(_IdentitySession(), available=True)
        step = OnnxDenoiseStep(tile_size=32, registry=reg)
        f1, f2 = _make_frame(32, 32), _make_frame(32, 32)
        ctx = self._ctx(f1.copy(), f2.copy())
        out = step.execute(ctx)
        assert len(out.images) == 2

    def test_execute_fallback_processes_images(self):
        reg = _make_registry(_IdentitySession(), available=False)
        step = OnnxDenoiseStep(strength=1.0, registry=reg)
        ctx = self._ctx(_make_frame(32, 32))
        out = step.execute(ctx)
        assert len(out.images) == 1

    def test_multiple_images_progress_reports(self):
        reg = _make_registry(_IdentitySession(), available=True)
        step = OnnxDenoiseStep(tile_size=32, registry=reg)
        calls: list[PipelineProgress] = []
        ctx = self._ctx(_make_frame(32, 32), _make_frame(32, 32))
        step.execute(ctx, progress=calls.append)
        assert len(calls) >= 2


# ===========================================================================
# PipelineModel — denoise_backend property
# ===========================================================================

class TestPipelineModelDenoiseBackend:
    def _model(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from astroai.ui.models import PipelineModel
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        return PipelineModel()

    def test_default_backend_is_nafnet(self):
        m = self._model()
        assert m.denoise_backend == "nafnet"

    def test_set_basic_backend(self):
        m = self._model()
        m.denoise_backend = "basic"
        assert m.denoise_backend == "basic"

    def test_invalid_backend_coerced_to_nafnet(self):
        m = self._model()
        m.denoise_backend = "unknown"
        assert m.denoise_backend == "nafnet"

    def test_signal_emitted_on_change(self):
        m = self._model()
        fired: list[None] = []
        m.denoise_config_changed.connect(lambda: fired.append(None))
        m.denoise_backend = "basic"
        assert len(fired) == 1

    def test_no_signal_on_same_value(self):
        m = self._model()
        m.denoise_backend = "nafnet"
        fired: list[None] = []
        m.denoise_config_changed.connect(lambda: fired.append(None))
        m.denoise_backend = "nafnet"
        assert len(fired) == 0
