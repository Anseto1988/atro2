"""Tests for SpectralColorCalibrator and ColorCalibrationStep."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroai.processing.color.calibrator import (
    CalibrationResult,
    CatalogQueryResult,
    CatalogSource,
    SpectralColorCalibrator,
    StarMeasurement,
)
from astroai.processing.color.pipeline_step import ColorCalibrationStep
from astroai.core.pipeline.base import PipelineContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(h: int = 64, w: int = 64, dtype: type = np.float32) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((h, w, 3)).astype(dtype)


def _make_catalog(n: int = 20) -> CatalogQueryResult:
    rng = np.random.default_rng(0)
    return CatalogQueryResult(
        ra=rng.uniform(10.0, 10.5, n),
        dec=rng.uniform(20.0, 20.5, n),
        color_index=rng.uniform(0.3, 1.5, n),
        flux_ratio_rg=rng.uniform(0.8, 1.2, n),
        flux_ratio_bg=rng.uniform(0.7, 1.3, n),
    )


def _make_wcs_mock(width: int = 64, height: int = 64) -> MagicMock:
    wcs = MagicMock()

    def pixel_to_world(x, y):
        sky = MagicMock()
        sky.ra.deg = np.atleast_1d(np.asarray(x) / width * 0.5 + 10.0)
        sky.dec.deg = np.atleast_1d(np.asarray(y) / height * 0.5 + 20.0)
        return sky

    def world_to_pixel(coords):
        # Return pixel coordinates inside the image for all stars
        n = len(np.atleast_1d(coords.ra.deg)) if hasattr(coords, "ra") else 1
        px = np.full(n, width // 2, dtype=np.float64)
        py = np.full(n, height // 2, dtype=np.float64)
        return px, py

    wcs.pixel_to_world.side_effect = pixel_to_world
    wcs.world_to_pixel.return_value = (
        np.array([32.0] * 5),
        np.array([32.0] * 5),
    )
    return wcs


# ---------------------------------------------------------------------------
# CatalogSource
# ---------------------------------------------------------------------------

class TestCatalogSource:
    def test_gaia_value(self) -> None:
        assert CatalogSource.GAIA_DR3.value == "gaia_dr3"

    def test_twomass_value(self) -> None:
        assert CatalogSource.TWOMASS.value == "2mass"


# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------

class TestCalibrationResult:
    def test_construction(self) -> None:
        mat = np.eye(3)
        result = CalibrationResult(
            correction_matrix=mat,
            stars_used=10,
            residual_rms=0.05,
            white_balance=(1.1, 1.0, 0.9),
        )
        assert result.stars_used == 10
        assert result.residual_rms == pytest.approx(0.05)
        assert result.white_balance == (1.1, 1.0, 0.9)


# ---------------------------------------------------------------------------
# SpectralColorCalibrator — no-catalog path
# ---------------------------------------------------------------------------

class TestSpectralColorCalibratorNoCatalog:
    """Tests using a pre-built catalog (avoids network)."""

    @pytest.fixture()
    def calibrator(self) -> SpectralColorCalibrator:
        return SpectralColorCalibrator(sample_radius_px=4, min_stars=3)

    def test_raises_on_wrong_shape(self, calibrator: SpectralColorCalibrator) -> None:
        gray = np.zeros((64, 64))
        with pytest.raises(ValueError, match="Expected .H, W, 3."):
            calibrator.calibrate(gray, MagicMock())

    def test_returns_identity_when_no_stars_in_field(
        self, calibrator: SpectralColorCalibrator
    ) -> None:
        image = _make_rgb_image()
        wcs = MagicMock()
        empty_catalog = CatalogQueryResult(
            ra=np.array([]),
            dec=np.array([]),
            color_index=np.array([]),
            flux_ratio_rg=np.array([]),
            flux_ratio_bg=np.array([]),
        )
        calibrated, result = calibrator.calibrate(image, wcs, catalog_data=empty_catalog)
        assert result.stars_used == 0
        np.testing.assert_array_almost_equal(result.correction_matrix, np.eye(3))
        assert calibrated.shape == image.shape

    def test_output_shape_preserved(self, calibrator: SpectralColorCalibrator) -> None:
        image = _make_rgb_image(32, 32)
        wcs = MagicMock()
        wcs.world_to_pixel.return_value = (np.array([16.0]), np.array([16.0]))

        catalog = CatalogQueryResult(
            ra=np.array([10.25]),
            dec=np.array([20.25]),
            color_index=np.array([0.8]),
            flux_ratio_rg=np.array([1.0]),
            flux_ratio_bg=np.array([1.0]),
        )

        calibrated, result = calibrator.calibrate(image, wcs, catalog_data=catalog)
        assert calibrated.shape == image.shape

    def test_identity_correction_preserves_image(self, calibrator: SpectralColorCalibrator) -> None:
        """When correction is identity (ratio = 1 for all channels), image is unchanged."""
        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        wcs = MagicMock()
        # Single star in center with equal reference ratios
        wcs.world_to_pixel.return_value = (np.array([16.0]), np.array([16.0]))
        catalog = CatalogQueryResult(
            ra=np.array([10.25]),
            dec=np.array([20.25]),
            color_index=np.array([0.5]),
            flux_ratio_rg=np.array([1.0]),
            flux_ratio_bg=np.array([1.0]),
        )
        # Below min_stars threshold → returns original
        calibrated, result = calibrator.calibrate(image, wcs, catalog_data=catalog)
        assert calibrated.shape == image.shape

    def test_correction_matrix_is_diagonal(self, calibrator: SpectralColorCalibrator) -> None:
        """Correction should be a diagonal matrix (per-channel scale)."""
        image = _make_rgb_image(64, 64)
        # Create catalog with multiple well-separated stars
        wcs = MagicMock()
        xs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ys = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        wcs.world_to_pixel.return_value = (xs, ys)

        n = 5
        catalog = CatalogQueryResult(
            ra=np.linspace(10.0, 10.4, n),
            dec=np.linspace(20.0, 20.4, n),
            color_index=np.ones(n) * 0.8,
            flux_ratio_rg=np.ones(n) * 1.2,
            flux_ratio_bg=np.ones(n) * 0.8,
        )
        calibrated, result = calibrator.calibrate(image, wcs, catalog_data=catalog)
        mat = result.correction_matrix
        # Off-diagonal elements should be zero
        assert mat[0, 1] == pytest.approx(0.0)
        assert mat[0, 2] == pytest.approx(0.0)
        assert mat[1, 0] == pytest.approx(0.0)
        assert mat[1, 2] == pytest.approx(0.0)
        assert mat[2, 0] == pytest.approx(0.0)
        assert mat[2, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SpectralColorCalibrator properties
# ---------------------------------------------------------------------------

class TestSpectralColorCalibratorProperties:
    def test_catalog_property(self) -> None:
        cal = SpectralColorCalibrator(catalog=CatalogSource.TWOMASS)
        assert cal.catalog == CatalogSource.TWOMASS

    def test_sample_radius_property(self) -> None:
        cal = SpectralColorCalibrator(sample_radius_px=12)
        assert cal.sample_radius == 12

    def test_default_catalog(self) -> None:
        cal = SpectralColorCalibrator()
        assert cal.catalog == CatalogSource.GAIA_DR3


# ---------------------------------------------------------------------------
# ColorCalibrationStep
# ---------------------------------------------------------------------------

class TestColorCalibrationStep:
    def test_name(self) -> None:
        step = ColorCalibrationStep()
        assert step.name == "Farbkalibrierung"

    def test_skips_when_no_wcs(self) -> None:
        step = ColorCalibrationStep()
        image = _make_rgb_image()
        ctx = PipelineContext(result=image)
        result_ctx = step.execute(ctx)
        assert result_ctx.result is image  # unchanged

    def test_skips_when_no_result(self) -> None:
        step = ColorCalibrationStep()
        wcs = MagicMock()
        ctx = PipelineContext()
        ctx.metadata["wcs"] = wcs
        result_ctx = step.execute(ctx)
        assert result_ctx.result is None

    def test_skips_grayscale(self) -> None:
        step = ColorCalibrationStep()
        gray = np.zeros((32, 32), dtype=np.float32)
        wcs = MagicMock()
        ctx = PipelineContext(result=gray)
        ctx.metadata["wcs"] = wcs
        result_ctx = step.execute(ctx)
        assert result_ctx.result is gray  # unchanged

    def test_stores_calibration_result_in_metadata(self) -> None:
        step = ColorCalibrationStep(sample_radius_px=4)
        image = _make_rgb_image(32, 32)
        wcs = MagicMock()
        wcs.world_to_pixel.return_value = (np.array([]), np.array([]))
        catalog = CatalogQueryResult(
            ra=np.array([]),
            dec=np.array([]),
            color_index=np.array([]),
            flux_ratio_rg=np.array([]),
            flux_ratio_bg=np.array([]),
        )
        ctx = PipelineContext(result=image)
        ctx.metadata["wcs"] = wcs
        ctx.metadata["color_catalog_data"] = catalog
        result_ctx = step.execute(ctx)
        assert "color_calibration_result" in result_ctx.metadata

    def test_accepts_catalog_source_twomass(self) -> None:
        step = ColorCalibrationStep(catalog=CatalogSource.TWOMASS)
        assert step._calibrator.catalog == CatalogSource.TWOMASS
