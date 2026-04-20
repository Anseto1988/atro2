"""End-to-end integration test for background extraction.

Verifies: synthetic FITS with gradient background -> BackgroundExtractor -> gradient removed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.processing.background.extractor import BackgroundExtractor, ModelMethod
from astroai.processing.background.pipeline_step import BackgroundRemovalStep
from astroai.core.pipeline.base import PipelineContext, PipelineProgress


def make_gradient_frame(
    height: int = 128,
    width: int = 128,
    seed: int = 42,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Generate synthetic frame with known linear gradient + noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    gradient = (yy / height * 500.0 + xx / width * 300.0).astype(np.float32)
    signal = rng.uniform(100.0, 200.0, (height, width)).astype(np.float32)
    frame = signal + gradient
    return frame, gradient


class TestBackgroundExtractorE2E:
    """Integration tests for BackgroundExtractor gradient removal."""

    def test_rbf_removes_gradient(self) -> None:
        frame, gradient = make_gradient_frame(128, 128)
        extractor = BackgroundExtractor(tile_size=32, method=ModelMethod.RBF)

        bg_model = extractor.extract(frame)

        assert bg_model.shape == frame.shape
        assert np.issubdtype(bg_model.dtype, np.floating)
        residual = frame - bg_model
        residual_std = np.std(residual)
        assert residual_std < np.std(frame) * 0.5

    def test_polynomial_removes_gradient(self) -> None:
        frame, gradient = make_gradient_frame(128, 128)
        extractor = BackgroundExtractor(
            tile_size=32, method=ModelMethod.POLYNOMIAL, poly_degree=2
        )

        bg_model = extractor.extract(frame)

        assert bg_model.shape == frame.shape
        correlation = np.corrcoef(bg_model.ravel(), gradient.ravel())[0, 1]
        assert correlation > 0.9

    def test_background_model_finite(self) -> None:
        frame, _ = make_gradient_frame(64, 64)
        extractor = BackgroundExtractor(tile_size=16, method=ModelMethod.RBF)

        bg_model = extractor.extract(frame)

        assert np.isfinite(bg_model).all()

    def test_rgb_frame_gradient_removal(self) -> None:
        frame_mono, _ = make_gradient_frame(64, 64, seed=10)
        frame_rgb = np.stack(
            [frame_mono, frame_mono * 0.9, frame_mono * 0.8], axis=-1
        )
        extractor = BackgroundExtractor(tile_size=32, method=ModelMethod.RBF)

        bg_model = extractor.extract(frame_rgb)

        assert bg_model.shape == (64, 64, 3)
        assert np.isfinite(bg_model).all()


class TestBackgroundRemovalStepE2E:
    """Integration tests for BackgroundRemovalStep in pipeline context."""

    def test_pipeline_step_removes_gradient(self) -> None:
        frame, _ = make_gradient_frame(128, 128)
        ctx = PipelineContext()
        ctx.result = frame

        step = BackgroundRemovalStep(tile_size=32, method=ModelMethod.RBF)
        out_ctx = step.execute(ctx)

        assert out_ctx.result is not None
        assert out_ctx.result.shape == frame.shape
        assert np.std(out_ctx.result) < np.std(frame)

    def test_pipeline_step_preserves_dtype(self) -> None:
        frame, _ = make_gradient_frame(64, 64)
        ctx = PipelineContext()
        ctx.result = frame.astype(np.float64)

        step = BackgroundRemovalStep(tile_size=32, method=ModelMethod.RBF)
        out_ctx = step.execute(ctx)

        assert out_ctx.result.dtype == np.float64

    def test_pipeline_step_finite_output(self) -> None:
        frame, _ = make_gradient_frame(64, 64)
        ctx = PipelineContext()
        ctx.result = frame

        step = BackgroundRemovalStep(tile_size=32)
        out_ctx = step.execute(ctx)

        assert np.isfinite(out_ctx.result).all()

    def test_step_name_and_stage(self) -> None:
        from astroai.core.pipeline.base import PipelineStage

        step = BackgroundRemovalStep()
        assert step.name == "Background Removal"
        assert step.stage == PipelineStage.PROCESSING
