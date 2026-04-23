"""Unit tests for DrizzleStep pipeline integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.astrometry.catalog import WcsSolution
from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.engine.drizzle.pipeline_step import DrizzleStep


@pytest.fixture()
def synthetic_images() -> list[NDArray[np.floating[Any]]]:
    rng = np.random.default_rng(99)
    return [rng.uniform(50.0, 300.0, (64, 64)).astype(np.float32) for _ in range(3)]


@pytest.fixture()
def identity_wcs_list() -> list[WcsSolution]:
    wcs = WcsSolution(
        ra_center=0.0, dec_center=0.0,
        pixel_scale_arcsec=1.0, rotation_deg=0.0,
        fov_width_deg=64 / 3600.0, fov_height_deg=64 / 3600.0,
        cd_matrix=(1.0 / 3600.0, 0.0, 0.0, 1.0 / 3600.0),
        crpix1=32.0, crpix2=32.0,
    )
    return [wcs, wcs, wcs]


class TestDrizzleStep:
    def test_step_stage(self):
        """Stage is PipelineStage.DRIZZLE."""
        step = DrizzleStep()
        assert step.stage == PipelineStage.DRIZZLE

    def test_step_name(self):
        """Name is 'Drizzle'."""
        step = DrizzleStep()
        assert step.name == "Drizzle"

    def test_execute_with_wcs(self, synthetic_images, identity_wcs_list):
        """Step reads WCS from context metadata and calls engine."""
        step = DrizzleStep(drop_size=1.0, pixfrac=1.0, scale=1.0)
        context = PipelineContext(
            images=synthetic_images,
            metadata={"wcs_solutions": identity_wcs_list},
        )

        result_ctx = step.execute(context)

        assert result_ctx.result is not None
        assert result_ctx.result.shape == (64, 64)
        assert result_ctx.result.dtype == np.float32
        assert np.isfinite(result_ctx.result).all()

    def test_execute_without_wcs(self, synthetic_images):
        """Fallback without WCS runs without error."""
        step = DrizzleStep(drop_size=1.0, pixfrac=1.0, scale=1.0)
        context = PipelineContext(images=synthetic_images, metadata={})

        result_ctx = step.execute(context)

        assert result_ctx.result is not None
        assert result_ctx.result.shape == (64, 64)
        assert result_ctx.result.dtype == np.float32
        assert np.isfinite(result_ctx.result).all()

    def test_execute_empty_images(self):
        """Empty image list is handled gracefully."""
        step = DrizzleStep()
        context = PipelineContext(images=[], metadata={})

        result_ctx = step.execute(context)
        assert result_ctx.result is None

    def test_scale_doubles_output(self, synthetic_images, identity_wcs_list):
        """Scale=2.0 produces doubled output dimensions."""
        step = DrizzleStep(drop_size=1.0, pixfrac=1.0, scale=2.0)
        context = PipelineContext(
            images=synthetic_images,
            metadata={"wcs_solutions": identity_wcs_list},
        )

        result_ctx = step.execute(context)

        assert result_ctx.result is not None
        assert result_ctx.result.shape == (128, 128)

    def test_execute_with_single_wcs(self, synthetic_images):
        """A single WcsSolution is broadcast to all frames."""
        single_wcs = WcsSolution(
            ra_center=0.0, dec_center=0.0,
            pixel_scale_arcsec=1.0, rotation_deg=0.0,
            fov_width_deg=64 / 3600.0, fov_height_deg=64 / 3600.0,
            cd_matrix=(1.0 / 3600.0, 0.0, 0.0, 1.0 / 3600.0),
            crpix1=32.0, crpix2=32.0,
        )
        context = PipelineContext(
            images=synthetic_images,
            metadata={"wcs_solution": single_wcs},
        )
        step = DrizzleStep(drop_size=1.0, pixfrac=1.0, scale=1.0)
        result_ctx = step.execute(context)

        assert result_ctx.result is not None
        assert result_ctx.result.shape == (64, 64)
        assert result_ctx.result.dtype == np.float32

    def test_progress_callback(self, synthetic_images, identity_wcs_list):
        """Progress callback is invoked during execution."""
        step = DrizzleStep(drop_size=1.0, pixfrac=1.0, scale=1.0)
        context = PipelineContext(
            images=synthetic_images,
            metadata={"wcs_solutions": identity_wcs_list},
        )

        progress_log = []
        step.execute(context, progress=progress_log.append)

        assert len(progress_log) >= 2
        assert all(p.stage == PipelineStage.DRIZZLE for p in progress_log)
