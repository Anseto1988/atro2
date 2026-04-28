"""Integration test for ColorCalibrationStep in the pipeline."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.core.pipeline.base import (
    Pipeline,
    PipelineContext,
    PipelineProgress,
    PipelineStage,
)
from astroai.processing.color.calibrator import CatalogQueryResult, CatalogSource
from astroai.processing.color.pipeline_step import ColorCalibrationStep


def _make_mock_wcs(width: int = 128, height: int = 128) -> Any:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    wcs = MagicMock()
    scale = 1.0 / 3600.0

    def pixel_to_world(x: Any, y: Any) -> Any:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        ra = 180.0 + (x - width / 2.0) * scale
        dec = 45.0 + (y - height / 2.0) * scale
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    def world_to_pixel(coords: Any) -> tuple[Any, Any]:
        ra = coords.ra.deg
        dec = coords.dec.deg
        x = (ra - 180.0) / scale + width / 2.0
        y = (dec - 45.0) / scale + height / 2.0
        return np.asarray(x), np.asarray(y)

    wcs.pixel_to_world = pixel_to_world
    wcs.world_to_pixel = world_to_pixel
    return wcs


def _make_test_image(
    width: int = 128, height: int = 128,
) -> tuple[NDArray[np.floating[Any]], list[tuple[int, int]]]:
    rng = np.random.default_rng(99)
    img = rng.normal(100.0, 2.0, (height, width, 3)).astype(np.float32)
    positions = [(32, 32), (96, 32), (64, 64), (32, 96), (96, 96), (50, 50), (80, 80)]
    yy, xx = np.mgrid[0:height, 0:width]
    for cx, cy in positions:
        star = 3000.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2))
        img[:, :, 0] += star * 1.2
        img[:, :, 1] += star * 1.0
        img[:, :, 2] += star * 0.8
    return np.clip(img, 0, None), positions


def _make_catalog(
    positions: list[tuple[int, int]], width: int = 128, height: int = 128,
) -> CatalogQueryResult:
    scale = 1.0 / 3600.0
    ra = np.array([180.0 + (cx - width / 2.0) * scale for cx, _ in positions])
    dec = np.array([45.0 + (cy - height / 2.0) * scale for _, cy in positions])
    n = len(positions)
    return CatalogQueryResult(
        ra=ra, dec=dec,
        color_index=np.zeros(n, dtype=np.float64),
        flux_ratio_rg=np.ones(n, dtype=np.float64),
        flux_ratio_bg=np.ones(n, dtype=np.float64),
    )


class TestColorCalibrationPipeline:
    def test_full_pipeline_run(self) -> None:
        image, positions = _make_test_image()
        wcs = _make_mock_wcs()
        catalog = _make_catalog(positions)

        pipeline = Pipeline([
            ColorCalibrationStep(catalog=CatalogSource.GAIA_DR3, sample_radius_px=5),
        ])

        progress_log: list[PipelineProgress] = []
        context = PipelineContext(
            result=image,
            metadata={"wcs": wcs, "color_catalog_data": catalog},
        )

        result_ctx = pipeline.run(context, progress=progress_log.append)

        assert result_ctx.result is not None
        assert result_ctx.result.shape == image.shape
        assert result_ctx.result.dtype == image.dtype
        assert np.isfinite(result_ctx.result).all()
        assert "color_calibration_result" in result_ctx.metadata

        cal_result = result_ctx.metadata["color_calibration_result"]
        assert cal_result.stars_used >= 3

        stages_seen = {p.stage for p in progress_log}
        assert PipelineStage.PROCESSING in stages_seen

    def test_skips_without_wcs(self) -> None:
        image, _ = _make_test_image()

        pipeline = Pipeline([ColorCalibrationStep()])
        context = PipelineContext(result=image, metadata={})

        result_ctx = pipeline.run(context)

        np.testing.assert_array_equal(result_ctx.result, image)
        assert "color_calibration_result" not in result_ctx.metadata

    def test_skips_grayscale(self) -> None:
        gray = np.zeros((128, 128), dtype=np.float32)
        wcs = _make_mock_wcs()

        pipeline = Pipeline([ColorCalibrationStep()])
        context = PipelineContext(result=gray, metadata={"wcs": wcs})

        result_ctx = pipeline.run(context)

        np.testing.assert_array_equal(result_ctx.result, gray)

    def test_skips_without_result(self) -> None:
        wcs = _make_mock_wcs()

        pipeline = Pipeline([ColorCalibrationStep()])
        context = PipelineContext(metadata={"wcs": wcs})

        result_ctx = pipeline.run(context)

        assert result_ctx.result is None
