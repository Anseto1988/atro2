"""End-to-end integration test for deconvolution pipeline.

Verifies: synthetic blurred FITS -> DenoiseStep -> StretchStep -> correct output shape and pixel range.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.processing.denoise.pipeline_step import DenoiseStep
from astroai.processing.stretch.pipeline_step import StretchStep
from astroai.core.pipeline.base import (
    Pipeline,
    PipelineContext,
    PipelineProgress,
    PipelineStage,
)


def make_blurred_starfield(
    height: int = 64,
    width: int = 64,
    n_stars: int = 10,
    blur_sigma: float = 3.0,
    noise_std: float = 10.0,
    seed: int = 42,
) -> NDArray[np.floating[Any]]:
    """Generate synthetic blurred starfield simulating atmospheric seeing."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width), dtype=np.float32)

    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(n_stars):
        cy = rng.integers(10, height - 10)
        cx = rng.integers(10, width - 10)
        flux = rng.uniform(500, 2000)
        star = flux * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * blur_sigma**2)
        )
        img += star.astype(np.float32)

    img += rng.normal(50, noise_std, (height, width)).astype(np.float32)
    return np.clip(img, 0, None)


@pytest.fixture()
def blurred_frame() -> NDArray[np.floating[Any]]:
    return make_blurred_starfield(64, 64, n_stars=8, blur_sigma=3.0, seed=42)


@pytest.fixture()
def blurred_rgb_frame() -> NDArray[np.floating[Any]]:
    mono = make_blurred_starfield(64, 64, n_stars=8, blur_sigma=3.0, seed=42)
    return np.stack([mono, mono * 0.9, mono * 0.8], axis=-1)


class TestDenoiseStepE2E:
    """Integration tests for DenoiseStep processing."""

    def test_denoise_preserves_shape(
        self, blurred_frame: NDArray[np.floating[Any]]
    ) -> None:
        step = DenoiseStep(strength=1.0, tile_size=64, tile_overlap=8)
        ctx = PipelineContext()
        ctx.result = blurred_frame

        out_ctx = step.execute(ctx)

        assert out_ctx.result is not None
        assert out_ctx.result.shape == blurred_frame.shape

    def test_denoise_preserves_dtype(
        self, blurred_frame: NDArray[np.floating[Any]]
    ) -> None:
        step = DenoiseStep(strength=1.0)
        ctx = PipelineContext()
        ctx.result = blurred_frame.astype(np.float64)

        out_ctx = step.execute(ctx)

        assert out_ctx.result.dtype == np.float64

    def test_denoise_reduces_noise(
        self, blurred_frame: NDArray[np.floating[Any]]
    ) -> None:
        step = DenoiseStep(strength=1.0, tile_size=64, tile_overlap=8)
        ctx = PipelineContext()
        ctx.result = blurred_frame

        out_ctx = step.execute(ctx)

        original_std = np.std(blurred_frame)
        denoised_std = np.std(out_ctx.result)
        assert denoised_std <= original_std

    def test_denoise_rgb(
        self, blurred_rgb_frame: NDArray[np.floating[Any]]
    ) -> None:
        step = DenoiseStep(strength=1.0, tile_size=64, tile_overlap=8)
        ctx = PipelineContext()
        ctx.result = blurred_rgb_frame

        out_ctx = step.execute(ctx)

        assert out_ctx.result.shape == (64, 64, 3)

    def test_denoise_finite_output(
        self, blurred_frame: NDArray[np.floating[Any]]
    ) -> None:
        step = DenoiseStep(strength=1.0, tile_size=64, tile_overlap=8)
        ctx = PipelineContext()
        ctx.result = blurred_frame

        out_ctx = step.execute(ctx)

        assert np.isfinite(out_ctx.result).all()


class TestStretchStepE2E:
    """Integration tests for StretchStep processing."""

    def test_stretch_output_range(
        self, blurred_frame: NDArray[np.floating[Any]]
    ) -> None:
        step = StretchStep(target_background=0.25, linked_channels=True)
        ctx = PipelineContext()
        ctx.result = blurred_frame

        out_ctx = step.execute(ctx)

        assert out_ctx.result is not None
        assert out_ctx.result.min() >= 0.0
        assert out_ctx.result.max() <= 1.0

    def test_stretch_preserves_shape(
        self, blurred_frame: NDArray[np.floating[Any]]
    ) -> None:
        step = StretchStep(target_background=0.25)
        ctx = PipelineContext()
        ctx.result = blurred_frame

        out_ctx = step.execute(ctx)

        assert out_ctx.result.shape == blurred_frame.shape

    def test_stretch_preserves_dtype(
        self, blurred_frame: NDArray[np.floating[Any]]
    ) -> None:
        step = StretchStep()
        ctx = PipelineContext()
        ctx.result = blurred_frame.astype(np.float64)

        out_ctx = step.execute(ctx)

        assert out_ctx.result.dtype == np.float64

    def test_stretch_rgb(
        self, blurred_rgb_frame: NDArray[np.floating[Any]]
    ) -> None:
        step = StretchStep(target_background=0.25, linked_channels=True)
        ctx = PipelineContext()
        ctx.result = blurred_rgb_frame

        out_ctx = step.execute(ctx)

        assert out_ctx.result.shape == (64, 64, 3)
        assert out_ctx.result.min() >= 0.0
        assert out_ctx.result.max() <= 1.0


class TestDenoiseStretchPipelineE2E:
    """Integration tests for combined DenoiseStep -> StretchStep pipeline."""

    def test_full_processing_pipeline(
        self, blurred_frame: NDArray[np.floating[Any]]
    ) -> None:
        pipeline = Pipeline(
            steps=[
                DenoiseStep(strength=1.0, tile_size=64, tile_overlap=8),
                StretchStep(target_background=0.25),
            ]
        )
        ctx = PipelineContext()
        ctx.result = blurred_frame

        out_ctx = pipeline.run(ctx)

        assert out_ctx.result is not None
        assert out_ctx.result.shape == blurred_frame.shape
        assert out_ctx.result.min() >= 0.0
        assert out_ctx.result.max() <= 1.0
        assert np.isfinite(out_ctx.result).all()

    def test_pipeline_progress_tracking(
        self, blurred_frame: NDArray[np.floating[Any]]
    ) -> None:
        pipeline = Pipeline(
            steps=[
                DenoiseStep(strength=1.0, tile_size=64, tile_overlap=8),
                StretchStep(target_background=0.25),
            ]
        )
        ctx = PipelineContext()
        ctx.result = blurred_frame
        progress_calls: list[PipelineProgress] = []

        pipeline.run(ctx, progress=lambda p: progress_calls.append(p))

        assert len(progress_calls) >= 4
        stages = {p.stage for p in progress_calls}
        assert PipelineStage.PROCESSING in stages

    def test_pipeline_rgb_end_to_end(
        self, blurred_rgb_frame: NDArray[np.floating[Any]]
    ) -> None:
        pipeline = Pipeline(
            steps=[
                DenoiseStep(strength=0.8, tile_size=64, tile_overlap=8),
                StretchStep(target_background=0.25, linked_channels=True),
            ]
        )
        ctx = PipelineContext()
        ctx.result = blurred_rgb_frame

        out_ctx = pipeline.run(ctx)

        assert out_ctx.result.shape == (64, 64, 3)
        assert out_ctx.result.min() >= 0.0
        assert out_ctx.result.max() <= 1.0
