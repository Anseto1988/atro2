"""Integration test for DrizzleStep in the full pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.astrometry.catalog import WcsSolution
from astroai.core.pipeline.base import (
    Pipeline,
    PipelineContext,
    PipelineProgress,
    PipelineStage,
)
from astroai.engine.drizzle.pipeline_step import DrizzleStep


def _make_synthetic_frames(
    n: int = 5, height: int = 128, width: int = 128, seed: int = 42
) -> list[NDArray[np.floating[Any]]]:
    rng = np.random.default_rng(seed)
    frames = []
    yy, xx = np.mgrid[0:height, 0:width]
    for i in range(n):
        bg = rng.normal(100.0, 5.0, (height, width)).astype(np.float32)
        cy, cx = height // 2 + i, width // 2 - i
        star = 2000.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5**2))
        frames.append((bg + star).astype(np.float32))
    return frames


def _make_wcs_list(
    n: int, height: int = 128, width: int = 128
) -> list[WcsSolution]:
    wcs_list = []
    for i in range(n):
        wcs_list.append(WcsSolution(
            ra_center=0.0,
            dec_center=0.0,
            pixel_scale_arcsec=1.0,
            rotation_deg=0.0,
            fov_width_deg=width / 3600.0,
            fov_height_deg=height / 3600.0,
            cd_matrix=(1.0 / 3600.0, 0.0, 0.0, 1.0 / 3600.0),
            crpix1=width / 2.0 + i,
            crpix2=height / 2.0 + i,
        ))
    return wcs_list


@pytest.mark.slow
class TestDrizzleFullPipeline:
    def test_drizzle_full_pipeline(self):
        """Full pipeline with DrizzleStep produces valid output."""
        frames = _make_synthetic_frames(n=5, height=128, width=128)
        wcs_solutions = _make_wcs_list(n=5, height=128, width=128)

        pipeline = Pipeline([
            DrizzleStep(drop_size=1.0, pixfrac=1.0, scale=1.0),
        ])

        progress_log: list[PipelineProgress] = []
        context = PipelineContext(
            images=frames,
            metadata={"wcs_solutions": wcs_solutions},
        )

        result_ctx = pipeline.run(context, progress=progress_log.append)

        assert result_ctx.result is not None
        assert result_ctx.result.shape == (128, 128)
        assert result_ctx.result.dtype == np.float32
        assert np.isfinite(result_ctx.result).all()
        assert result_ctx.result.max() > 0.0

        stages_seen = {p.stage for p in progress_log}
        assert PipelineStage.DRIZZLE in stages_seen
        assert PipelineStage.SAVING in stages_seen

    def test_drizzle_pipeline_scale_2x(self):
        """Pipeline with scale=2.0 produces upsampled output."""
        frames = _make_synthetic_frames(n=3, height=64, width=64, seed=77)
        wcs_solutions = _make_wcs_list(n=3, height=64, width=64)

        pipeline = Pipeline([
            DrizzleStep(drop_size=0.7, pixfrac=0.8, scale=2.0),
        ])

        context = PipelineContext(
            images=frames,
            metadata={"wcs_solutions": wcs_solutions},
        )

        result_ctx = pipeline.run(context)

        assert result_ctx.result is not None
        assert result_ctx.result.shape == (128, 128)
        assert np.isfinite(result_ctx.result).all()

    def test_drizzle_pipeline_without_wcs_fallback(self):
        """Pipeline runs with identity WCS fallback when no WCS in metadata."""
        frames = _make_synthetic_frames(n=3, height=64, width=64, seed=55)

        pipeline = Pipeline([
            DrizzleStep(drop_size=1.0, pixfrac=1.0, scale=1.0),
        ])

        context = PipelineContext(images=frames, metadata={})
        result_ctx = pipeline.run(context)

        assert result_ctx.result is not None
        assert result_ctx.result.shape == (64, 64)
        assert result_ctx.result.dtype == np.float32
        assert np.isfinite(result_ctx.result).all()
