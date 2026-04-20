from __future__ import annotations

import numpy as np
import pytest
import torch

from astroai.core.calibration.calibrate import apply_dark, apply_flat, calibrate_frame
from astroai.core.calibration.gpu_engine import GPUCalibrationEngine
from astroai.core.calibration.matcher import (
    CalibrationFrame,
    CalibrationLibrary,
)
from astroai.core.io.fits_io import ImageMetadata
from pathlib import Path

from tests.conftest import HAS_GPU, requires_gpu


def _meta(width: int = 64, height: int = 64) -> ImageMetadata:
    return ImageMetadata(exposure=120.0, gain_iso=800, temperature=-10.0, width=width, height=height)


def _lib(dark: np.ndarray, flat: np.ndarray, meta: ImageMetadata) -> CalibrationLibrary:
    df = CalibrationFrame(path=Path("dark.fits"), metadata=meta, data=dark)
    ff = CalibrationFrame(path=Path("flat.fits"), metadata=meta, data=flat)
    return CalibrationLibrary(darks=[df], flats=[ff], bias=[])


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def engine() -> GPUCalibrationEngine:
    return GPUCalibrationEngine()


# ---------------------------------------------------------------------------
# GPU Engine low-level ops
# ---------------------------------------------------------------------------
class TestGPUEngineOps:
    def test_apply_dark_gpu_clamps_negative(self, engine: GPUCalibrationEngine) -> None:
        light = torch.tensor([[0.1, 0.2], [0.05, 0.3]])
        dark = torch.tensor([[0.15, 0.1], [0.1, 0.05]])
        result = engine.apply_dark_gpu(light, dark)
        assert result.min().item() >= 0.0

    def test_apply_flat_gpu_normalises(self, engine: GPUCalibrationEngine) -> None:
        light = torch.ones(4, 4) * 0.5
        flat = torch.ones(4, 4) * 0.8
        result = engine.apply_flat_gpu(light, flat)
        np.testing.assert_allclose(result.numpy(), np.full((4, 4), 0.5), atol=1e-5)

    def test_flat_near_zero_no_div_zero(self, engine: GPUCalibrationEngine) -> None:
        light = torch.ones(4, 4)
        flat = torch.zeros(4, 4)
        result = engine.apply_flat_gpu(light, flat)
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# GPU ↔ CPU Parity Tests
# ---------------------------------------------------------------------------
class TestGPUCPUParity:
    """GPU results must match CPU within tolerance.

    torch.median() returns lower middle value for even-length arrays;
    np.median() averages the two middle values → small but acceptable divergence.
    """

    ATOL = 5e-4

    @pytest.mark.parametrize("size", [128, 1024], ids=["128x128", "1024x1024"])
    def test_apply_dark_parity(
        self, rng: np.random.Generator, engine: GPUCalibrationEngine, size: int,
    ) -> None:
        light = rng.random((size, size), dtype=np.float32)
        dark = rng.random((size, size), dtype=np.float32) * 0.1
        cpu = apply_dark(light, dark)
        light_t = torch.from_numpy(light).to(engine._device)
        dark_t = torch.from_numpy(dark).to(engine._device)
        gpu = engine.apply_dark_gpu(light_t, dark_t).cpu().numpy()
        np.testing.assert_allclose(gpu, cpu, atol=self.ATOL)

    @pytest.mark.parametrize("size", [128, 1024], ids=["128x128", "1024x1024"])
    def test_apply_flat_parity(
        self, rng: np.random.Generator, engine: GPUCalibrationEngine, size: int,
    ) -> None:
        light = rng.random((size, size), dtype=np.float32)
        flat = rng.random((size, size), dtype=np.float32) + 0.5
        cpu = apply_flat(light, flat)
        light_t = torch.from_numpy(light).to(engine._device)
        flat_t = torch.from_numpy(flat).to(engine._device)
        gpu = engine.apply_flat_gpu(light_t, flat_t).cpu().numpy()
        np.testing.assert_allclose(gpu, cpu, atol=self.ATOL)

    @pytest.mark.parametrize("size", [64, 512], ids=["64x64", "512x512"])
    def test_calibrate_frame_parity(
        self, rng: np.random.Generator, engine: GPUCalibrationEngine, size: int,
    ) -> None:
        meta = _meta(width=size, height=size)
        light = rng.random((size, size), dtype=np.float32)
        dark = rng.random((size, size), dtype=np.float32) * 0.1
        flat = rng.random((size, size), dtype=np.float32) + 0.5
        lib = _lib(dark, flat, meta)

        cpu = calibrate_frame(light, meta, lib, use_gpu=False)
        gpu = engine.calibrate_frame_gpu(light, meta, lib)
        np.testing.assert_allclose(gpu, cpu, atol=self.ATOL)


@pytest.mark.slow
class TestGPUCPUParityLargeArrays:
    """Full-spec parity tests with 4096×4096 arrays (VER-242 requirement)."""

    ATOL = 5e-4

    @pytest.fixture()
    def rng_large(self) -> np.random.Generator:
        return np.random.default_rng(99)

    def test_apply_dark_parity_4k(
        self, rng_large: np.random.Generator, engine: GPUCalibrationEngine,
    ) -> None:
        light = rng_large.random((4096, 4096), dtype=np.float32)
        dark = rng_large.random((4096, 4096), dtype=np.float32) * 0.1
        cpu = apply_dark(light, dark)
        light_t = torch.from_numpy(light).to(engine._device)
        dark_t = torch.from_numpy(dark).to(engine._device)
        gpu = engine.apply_dark_gpu(light_t, dark_t).cpu().numpy()
        np.testing.assert_allclose(gpu, cpu, atol=self.ATOL)

    def test_apply_flat_parity_4k(
        self, rng_large: np.random.Generator, engine: GPUCalibrationEngine,
    ) -> None:
        light = rng_large.random((4096, 4096), dtype=np.float32)
        flat = rng_large.random((4096, 4096), dtype=np.float32) + 0.5
        cpu = apply_flat(light, flat)
        light_t = torch.from_numpy(light).to(engine._device)
        flat_t = torch.from_numpy(flat).to(engine._device)
        gpu = engine.apply_flat_gpu(light_t, flat_t).cpu().numpy()
        np.testing.assert_allclose(gpu, cpu, atol=self.ATOL)

    def test_calibrate_frame_parity_4k(
        self, rng_large: np.random.Generator, engine: GPUCalibrationEngine,
    ) -> None:
        meta = _meta(width=4096, height=4096)
        light = rng_large.random((4096, 4096), dtype=np.float32)
        dark = rng_large.random((4096, 4096), dtype=np.float32) * 0.1
        flat = rng_large.random((4096, 4096), dtype=np.float32) + 0.5
        lib = _lib(dark, flat, meta)

        cpu = calibrate_frame(light, meta, lib, use_gpu=False)
        gpu = engine.calibrate_frame_gpu(light, meta, lib)
        np.testing.assert_allclose(gpu, cpu, atol=self.ATOL)


# ---------------------------------------------------------------------------
# GPU-only parity (skipped without real GPU hardware)
# ---------------------------------------------------------------------------
@requires_gpu
@pytest.mark.gpu
class TestGPUHardwareParity:
    """Parity tests that only run on real GPU hardware."""

    ATOL = 5e-4

    def test_gpu_device_not_cpu(self, engine: GPUCalibrationEngine) -> None:
        assert engine.device_type in ("cuda", "mps")

    def test_apply_dark_gpu_hw(self, rng: np.random.Generator, engine: GPUCalibrationEngine) -> None:
        light = rng.random((1024, 1024), dtype=np.float32)
        dark = rng.random((1024, 1024), dtype=np.float32) * 0.1
        cpu = apply_dark(light, dark)
        light_t = torch.from_numpy(light).to(engine._device)
        dark_t = torch.from_numpy(dark).to(engine._device)
        gpu = engine.apply_dark_gpu(light_t, dark_t).cpu().numpy()
        np.testing.assert_allclose(gpu, cpu, atol=self.ATOL)

    def test_apply_flat_gpu_hw(self, rng: np.random.Generator, engine: GPUCalibrationEngine) -> None:
        light = rng.random((1024, 1024), dtype=np.float32)
        flat = rng.random((1024, 1024), dtype=np.float32) + 0.5
        cpu = apply_flat(light, flat)
        light_t = torch.from_numpy(light).to(engine._device)
        flat_t = torch.from_numpy(flat).to(engine._device)
        gpu = engine.apply_flat_gpu(light_t, flat_t).cpu().numpy()
        np.testing.assert_allclose(gpu, cpu, atol=self.ATOL)


# ---------------------------------------------------------------------------
# Batch calibration
# ---------------------------------------------------------------------------
class TestBatchCalibration:
    def test_batch_matches_single(self, rng: np.random.Generator, engine: GPUCalibrationEngine) -> None:
        meta = _meta()
        dark = rng.random((64, 64), dtype=np.float32) * 0.1
        flat = rng.random((64, 64), dtype=np.float32) + 0.5
        lib = _lib(dark, flat, meta)
        frames = [rng.random((64, 64), dtype=np.float32) for _ in range(5)]

        batch_results = engine.calibrate_batch_gpu(frames, meta, lib)
        for frame, batch_result in zip(frames, batch_results):
            single = engine.calibrate_frame_gpu(frame, meta, lib)
            np.testing.assert_allclose(batch_result, single, atol=1e-5)

    def test_batch_returns_correct_count(self, rng: np.random.Generator, engine: GPUCalibrationEngine) -> None:
        meta = _meta()
        lib = CalibrationLibrary.empty()
        frames = [rng.random((64, 64), dtype=np.float32) for _ in range(10)]
        results = engine.calibrate_batch_gpu(frames, meta, lib)
        assert len(results) == 10

    def test_batch_empty_library(self, rng: np.random.Generator, engine: GPUCalibrationEngine) -> None:
        meta = _meta()
        lib = CalibrationLibrary.empty()
        light = rng.random((64, 64), dtype=np.float32)
        results = engine.calibrate_batch_gpu([light], meta, lib)
        np.testing.assert_allclose(results[0], light, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration: calibrate_frame GPU/CPU paths
# ---------------------------------------------------------------------------
class TestCalibrationFrameIntegration:
    def test_calibrate_frame_uses_gpu_when_available(self, rng: np.random.Generator) -> None:
        meta = _meta()
        light = rng.random((64, 64), dtype=np.float32)
        lib = CalibrationLibrary.empty()
        result = calibrate_frame(light, meta, lib, use_gpu=True)
        assert result.shape == (64, 64)

    def test_calibrate_frame_cpu_fallback(self, rng: np.random.Generator) -> None:
        meta = _meta()
        light = rng.random((64, 64), dtype=np.float32)
        lib = CalibrationLibrary.empty()
        result = calibrate_frame(light, meta, lib, use_gpu=False)
        assert result.shape == (64, 64)

    def test_gpu_cpu_results_same_shape(self, rng: np.random.Generator) -> None:
        meta = _meta()
        light = rng.random((64, 64), dtype=np.float32)
        lib = CalibrationLibrary.empty()
        gpu_result = calibrate_frame(light, meta, lib, use_gpu=True)
        cpu_result = calibrate_frame(light, meta, lib, use_gpu=False)
        assert gpu_result.shape == cpu_result.shape
        assert gpu_result.dtype == cpu_result.dtype
