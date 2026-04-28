"""Unit tests for SyntheticFlatGenerator and SyntheticFlatStep."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.processing.background.extractor import ModelMethod
from astroai.processing.flat.synthetic_generator import SyntheticFlatGenerator
from astroai.processing.flat.pipeline_step import SyntheticFlatStep
from astroai.core.pipeline.base import PipelineContext, PipelineProgress


np.random.seed(0)

H, W = 64, 64


def _vignette_frame(
    h: int = H,
    w: int = W,
    center_val: float = 1000.0,
    edge_fraction: float = 0.6,
) -> NDArray[np.float64]:
    """Simulated frame with radial vignetting + faint sky background."""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = np.sqrt(cy ** 2 + cx ** 2)
    flat_model = center_val * (1.0 - (1.0 - edge_fraction) * (r / r_max) ** 2)
    noise = np.random.normal(0, 5.0, (h, w))
    return (flat_model + noise).astype(np.float64)


def _rgb_vignette(h: int = H, w: int = W) -> NDArray[np.float64]:
    r = _vignette_frame(h, w, center_val=1000.0, edge_fraction=0.6)
    g = _vignette_frame(h, w, center_val=1100.0, edge_fraction=0.65)
    b = _vignette_frame(h, w, center_val=900.0, edge_fraction=0.55)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# SyntheticFlatGenerator
# ---------------------------------------------------------------------------

class TestSyntheticFlatGeneratorInit:
    def test_default_properties(self) -> None:
        gen = SyntheticFlatGenerator()
        assert gen.smoothing_sigma == 8.0
        assert gen.min_frames == 1

    def test_custom_properties(self) -> None:
        gen = SyntheticFlatGenerator(smoothing_sigma=5.0, min_frames=3)
        assert gen.smoothing_sigma == 5.0
        assert gen.min_frames == 3


class TestSyntheticFlatGeneratorGenerate:
    def test_output_shape_2d(self) -> None:
        gen = SyntheticFlatGenerator(tile_size=16, smoothing_sigma=2.0)
        frame = _vignette_frame()
        result = gen.generate([frame])
        assert result.shape == (H, W)
        assert result.dtype == np.float32

    def test_output_shape_rgb(self) -> None:
        gen = SyntheticFlatGenerator(tile_size=16, smoothing_sigma=2.0)
        frame = _rgb_vignette()
        result = gen.generate([frame])
        assert result.shape == (H, W, 3)
        assert result.dtype == np.float32

    def test_peak_is_one(self) -> None:
        gen = SyntheticFlatGenerator(tile_size=16, smoothing_sigma=2.0)
        frame = _vignette_frame()
        result = gen.generate([frame])
        assert float(result.max()) == pytest.approx(1.0, abs=1e-4)

    def test_values_in_range(self) -> None:
        gen = SyntheticFlatGenerator(tile_size=16, smoothing_sigma=2.0)
        frame = _vignette_frame()
        result = gen.generate([frame])
        assert float(result.min()) > 0.0
        assert float(result.max()) <= 1.0 + 1e-5

    def test_multi_frame_median(self) -> None:
        gen = SyntheticFlatGenerator(tile_size=16, smoothing_sigma=2.0)
        frames = [_vignette_frame() for _ in range(5)]
        result = gen.generate(frames)
        assert result.shape == (H, W)
        assert float(result.max()) == pytest.approx(1.0, abs=1e-4)

    def test_polynomial_method(self) -> None:
        gen = SyntheticFlatGenerator(
            tile_size=16,
            method=ModelMethod.POLYNOMIAL,
            poly_degree=3,
            smoothing_sigma=2.0,
        )
        result = gen.generate([_vignette_frame()])
        assert result.shape == (H, W)
        assert float(result.max()) == pytest.approx(1.0, abs=1e-4)

    def test_zero_smoothing_sigma(self) -> None:
        gen = SyntheticFlatGenerator(tile_size=16, smoothing_sigma=0.0)
        result = gen.generate([_vignette_frame()])
        assert result.shape == (H, W)

    def test_too_few_frames_raises(self) -> None:
        gen = SyntheticFlatGenerator(min_frames=3)
        with pytest.raises(ValueError, match="at least 3"):
            gen.generate([_vignette_frame(), _vignette_frame()])

    def test_zero_model_returns_uniform(self) -> None:
        gen = SyntheticFlatGenerator(tile_size=16, smoothing_sigma=0.0)
        zero_frame = np.zeros((H, W), dtype=np.float64)
        result = gen.generate([zero_frame])
        assert np.allclose(result, 1.0)

    def test_vignetting_centre_brighter(self) -> None:
        gen = SyntheticFlatGenerator(tile_size=16, smoothing_sigma=2.0)
        frame = _vignette_frame(edge_fraction=0.5)
        result = gen.generate([frame])
        cy, cx = H // 2, W // 2
        centre = float(result[cy, cx])
        corner = float(result[0, 0])
        assert centre > corner


class TestSyntheticFlatStep:
    def _context_with_frames(self, n: int = 3) -> PipelineContext:
        ctx = PipelineContext()
        ctx.images = [_vignette_frame() for _ in range(n)]
        return ctx

    def test_applies_to_all_frames(self) -> None:
        step = SyntheticFlatStep(tile_size=16, smoothing_sigma=2.0)
        ctx = self._context_with_frames(3)
        original_shapes = [f.shape for f in ctx.images]
        result = step.execute(ctx)
        assert len(result.images) == 3
        for s, img in zip(original_shapes, result.images):
            assert img.shape == s

    def test_metadata_flag_set(self) -> None:
        step = SyntheticFlatStep(tile_size=16, smoothing_sigma=2.0)
        ctx = self._context_with_frames(2)
        result = step.execute(ctx)
        assert result.metadata.get("synthetic_flat_applied") is True

    def test_empty_context_skips(self) -> None:
        step = SyntheticFlatStep(tile_size=16, smoothing_sigma=2.0)
        ctx = PipelineContext()
        result = step.execute(ctx)
        assert result.images == []
        assert not result.metadata.get("synthetic_flat_applied", False)

    def test_name_and_stage(self) -> None:
        from astroai.core.pipeline.base import PipelineStage
        step = SyntheticFlatStep()
        assert step.name == "Synthetic Flat"
        assert step.stage == PipelineStage.CALIBRATION

    def test_progress_callback_called(self) -> None:
        step = SyntheticFlatStep(tile_size=16, smoothing_sigma=2.0)
        ctx = self._context_with_frames(2)
        calls: list[PipelineProgress] = []
        step.execute(ctx, progress=calls.append)
        assert len(calls) >= 3

    def test_dtype_preserved(self) -> None:
        step = SyntheticFlatStep(tile_size=16, smoothing_sigma=2.0)
        ctx = PipelineContext()
        ctx.images = [_vignette_frame().astype(np.float32)]
        result = step.execute(ctx)
        assert result.images[0].dtype == np.float32

    def test_rgb_frames(self) -> None:
        step = SyntheticFlatStep(tile_size=16, smoothing_sigma=2.0)
        ctx = PipelineContext()
        ctx.images = [_rgb_vignette() for _ in range(2)]
        result = step.execute(ctx)
        for img in result.images:
            assert img.shape == (H, W, 3)
