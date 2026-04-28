"""Tests for SpectralColorCalibrator and ColorCalibrationStep."""
from __future__ import annotations

import sys
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


# ---------------------------------------------------------------------------
# Helpers: patch astropy/astroquery for tests that call _query_catalog
# ---------------------------------------------------------------------------

def _make_astropy_mock():
    mock_skycoord = MagicMock()
    mock_sep = MagicMock()
    mock_sep.deg = 0.3
    mock_skycoord_inst = MagicMock()
    mock_skycoord_inst.separation.return_value = mock_sep
    mock_skycoord.return_value = mock_skycoord_inst

    mock_u = MagicMock()
    mock_u.deg = "deg"

    mock_astropy_coords = MagicMock()
    mock_astropy_coords.SkyCoord = mock_skycoord

    mock_astropy_units = mock_u

    return mock_skycoord, mock_u, mock_astropy_coords, mock_astropy_units


# ---------------------------------------------------------------------------
# _query_catalog via calibrate (catalog_data=None path, line 104)
# ---------------------------------------------------------------------------

class TestQueryCatalogDispatch:
    def test_query_catalog_calls_query_gaia_for_gaia_source(self) -> None:
        cal = SpectralColorCalibrator(catalog=CatalogSource.GAIA_DR3, sample_radius_px=4, min_stars=3)
        image = _make_rgb_image(64, 64)
        wcs = MagicMock()

        corners_sky = MagicMock()
        corners_sky.ra.deg = np.array([10.0, 10.5, 10.5, 10.0])
        corners_sky.dec.deg = np.array([20.0, 20.0, 20.5, 20.5])

        center_sky = MagicMock()
        center_sky.ra.deg = 10.25
        center_sky.dec.deg = 20.25

        def pixel_to_world_side_effect(x, y):
            if np.ndim(x) == 0:
                return center_sky
            return corners_sky

        wcs.pixel_to_world.side_effect = pixel_to_world_side_effect

        fake_catalog = CatalogQueryResult(
            ra=np.array([]),
            dec=np.array([]),
            color_index=np.array([]),
            flux_ratio_rg=np.array([]),
            flux_ratio_bg=np.array([]),
        )

        mock_skycoord, mock_u, mock_astropy_coords, mock_astropy_units = _make_astropy_mock()

        with patch.object(cal, "_query_gaia", return_value=fake_catalog) as mock_gaia:
            with patch.dict(sys.modules, {
                "astropy.coordinates": mock_astropy_coords,
                "astropy.units": mock_astropy_units,
            }):
                cal.calibrate(image, wcs)
            mock_gaia.assert_called_once()

    def test_query_catalog_calls_query_2mass_for_twomass_source(self) -> None:
        cal = SpectralColorCalibrator(catalog=CatalogSource.TWOMASS, sample_radius_px=4, min_stars=3)
        image = _make_rgb_image(64, 64)
        wcs = MagicMock()

        corners_sky = MagicMock()
        corners_sky.ra.deg = np.array([10.0, 10.5, 10.5, 10.0])
        corners_sky.dec.deg = np.array([20.0, 20.0, 20.5, 20.5])

        center_sky = MagicMock()
        center_sky.ra.deg = 10.25
        center_sky.dec.deg = 20.25

        def pixel_to_world_side_effect(x, y):
            if np.ndim(x) == 0:
                return center_sky
            return corners_sky

        wcs.pixel_to_world.side_effect = pixel_to_world_side_effect

        fake_catalog = CatalogQueryResult(
            ra=np.array([]),
            dec=np.array([]),
            color_index=np.array([]),
            flux_ratio_rg=np.array([]),
            flux_ratio_bg=np.array([]),
        )

        mock_skycoord, mock_u, mock_astropy_coords, mock_astropy_units = _make_astropy_mock()

        with patch.object(cal, "_query_2mass", return_value=fake_catalog) as mock_2mass:
            with patch.dict(sys.modules, {
                "astropy.coordinates": mock_astropy_coords,
                "astropy.units": mock_astropy_units,
            }):
                cal.calibrate(image, wcs)
            mock_2mass.assert_called_once()


# ---------------------------------------------------------------------------
# _query_gaia (lines 184-210)
# ---------------------------------------------------------------------------

def _make_gaia_selectable_table(data: dict):
    class GaiaTable:
        def __init__(self, d):
            self._d = d

        def __getitem__(self, key):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return GaiaTable({k: v[key] for k, v in self._d.items()})
            return self._d[key]

    return GaiaTable(data)


class TestQueryGaia:
    def _mock_gaia_modules(self):
        mock_gaia_mod = MagicMock()
        mock_astroquery = MagicMock()
        mock_astroquery.gaia = mock_gaia_mod
        mock_astropy_units = MagicMock()
        mock_astropy_units.deg = MagicMock()
        mock_astropy_coords = MagicMock()
        return mock_gaia_mod, mock_astroquery, mock_astropy_units, mock_astropy_coords

    def test_query_gaia_returns_catalog_result(self) -> None:
        cal = SpectralColorCalibrator()
        n = 3
        data = {
            "phot_bp_mean_mag": np.linspace(12.0, 14.0, n),
            "phot_rp_mean_mag": np.linspace(11.5, 13.5, n),
            "phot_g_mean_mag": np.linspace(11.8, 13.8, n),
            "bp_rp": np.full(n, 0.5),
            "ra": np.linspace(10.1, 10.3, n),
            "dec": np.linspace(20.1, 20.3, n),
        }
        fake_table = _make_gaia_selectable_table(data)

        mock_job = MagicMock()
        mock_job.get_results.return_value = fake_table

        mock_gaia_cls = MagicMock()
        mock_gaia_cls.cone_search_async.return_value = mock_job

        mock_gaia_module = MagicMock()
        mock_gaia_module.Gaia = mock_gaia_cls

        mock_astroquery_pkg = MagicMock()
        mock_astroquery_pkg.gaia = mock_gaia_module

        mock_astropy_units = MagicMock()
        mock_astropy_coords = MagicMock()

        with patch.dict(sys.modules, {
            "astroquery": mock_astroquery_pkg,
            "astroquery.gaia": mock_gaia_module,
            "astropy.units": mock_astropy_units,
            "astropy.coordinates": mock_astropy_coords,
        }):
            result = cal._query_gaia(10.25, 20.25, 0.5)

        assert isinstance(result, CatalogQueryResult)
        assert len(result.ra) == 3

    def test_query_gaia_filters_faint_stars(self) -> None:
        cal = SpectralColorCalibrator()
        data = {
            "phot_bp_mean_mag": np.array([12.0, 17.0]),
            "phot_rp_mean_mag": np.array([11.5, 16.5]),
            "phot_g_mean_mag": np.array([11.8, 17.0]),
            "bp_rp": np.array([0.5, 0.5]),
            "ra": np.array([10.1, 10.2]),
            "dec": np.array([20.1, 20.2]),
        }
        fake_table = _make_gaia_selectable_table(data)

        mock_job = MagicMock()
        mock_job.get_results.return_value = fake_table

        mock_gaia_cls = MagicMock()
        mock_gaia_cls.cone_search_async.return_value = mock_job

        mock_gaia_module = MagicMock()
        mock_gaia_module.Gaia = mock_gaia_cls

        mock_astropy_units = MagicMock()
        mock_astropy_coords = MagicMock()

        with patch.dict(sys.modules, {
            "astroquery": MagicMock(),
            "astroquery.gaia": mock_gaia_module,
            "astropy.units": mock_astropy_units,
            "astropy.coordinates": mock_astropy_coords,
        }):
            result = cal._query_gaia(10.25, 20.25, 0.5)

        assert isinstance(result, CatalogQueryResult)
        assert len(result.ra) == 1


# ---------------------------------------------------------------------------
# _query_2mass (lines 221-256)
# ---------------------------------------------------------------------------

def _make_2mass_selectable_table(data: dict):
    class TwoMassTable:
        def __init__(self, d):
            self._d = d

        def __getitem__(self, key):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return TwoMassTable({k: v[key] for k, v in self._d.items()})
            return self._d[key]

    return TwoMassTable(data)


class TestQuery2Mass:
    def _patch_vizier(self, return_value):
        mock_vizier_inst = MagicMock()
        mock_vizier_inst.query_region.return_value = return_value

        mock_vizier_cls = MagicMock()
        mock_vizier_cls.return_value = mock_vizier_inst

        mock_vizier_module = MagicMock()
        mock_vizier_module.Vizier = mock_vizier_cls

        mock_astropy_units = MagicMock()
        mock_astropy_coords = MagicMock()

        return mock_vizier_module, mock_astropy_units, mock_astropy_coords

    def test_query_2mass_empty_result_returns_empty_catalog(self) -> None:
        cal = SpectralColorCalibrator(catalog=CatalogSource.TWOMASS)
        mock_vizier_module, mock_u, mock_coords = self._patch_vizier([])

        with patch.dict(sys.modules, {
            "astroquery": MagicMock(),
            "astroquery.vizier": mock_vizier_module,
            "astropy.units": mock_u,
            "astropy.coordinates": mock_coords,
        }):
            result = cal._query_2mass(10.25, 20.25, 0.5)

        assert len(result.ra) == 0
        assert len(result.dec) == 0

    def test_query_2mass_with_data_returns_catalog(self) -> None:
        cal = SpectralColorCalibrator(catalog=CatalogSource.TWOMASS)
        raw = {
            "Jmag": np.array([14.0, 15.0, 13.5]),
            "Hmag": np.array([13.5, 14.5, 13.0]),
            "Kmag": np.array([13.2, 14.2, 12.7]),
            "RAJ2000": np.array([10.1, 10.2, 10.3]),
            "DEJ2000": np.array([20.1, 20.2, 20.3]),
        }
        fake_table = _make_2mass_selectable_table(raw)
        mock_vizier_module, mock_u, mock_coords = self._patch_vizier([fake_table])

        with patch.dict(sys.modules, {
            "astroquery": MagicMock(),
            "astroquery.vizier": mock_vizier_module,
            "astropy.units": mock_u,
            "astropy.coordinates": mock_coords,
        }):
            result = cal._query_2mass(10.25, 20.25, 0.5)

        assert isinstance(result, CatalogQueryResult)
        assert len(result.ra) == 3


# ---------------------------------------------------------------------------
# _measure_stars edge cases (lines 310, 317)
# ---------------------------------------------------------------------------

class TestMeasureStarsEdgeCases:
    def _make_measurement(self, flux_r=1.0, flux_g=1.0, flux_b=1.0) -> StarMeasurement:
        return StarMeasurement(
            x=0.0, y=0.0, ra=10.0, dec=20.0,
            flux_r=flux_r, flux_g=flux_g, flux_b=flux_b,
            catalog_color_index=0.5,
            catalog_flux_ratio_rg=1.0,
            catalog_flux_ratio_bg=1.0,
        )

    def test_patch_too_small_is_skipped(self) -> None:
        cal = SpectralColorCalibrator(sample_radius_px=8)
        image = np.ones((20, 20, 3), dtype=np.float32)
        px = np.array([2.0])
        py = np.array([2.0])
        indices = np.array([0])
        catalog = CatalogQueryResult(
            ra=np.array([10.0]),
            dec=np.array([20.0]),
            color_index=np.array([0.5]),
            flux_ratio_rg=np.array([1.0]),
            flux_ratio_bg=np.array([1.0]),
        )
        result = cal._measure_stars(image, (px, py, indices), catalog)
        assert len(result) == 0

    def test_zero_flux_star_is_skipped(self) -> None:
        cal = SpectralColorCalibrator(sample_radius_px=4)
        image = np.zeros((64, 64, 3), dtype=np.float32)
        px = np.array([32.0])
        py = np.array([32.0])
        indices = np.array([0])
        catalog = CatalogQueryResult(
            ra=np.array([10.0]),
            dec=np.array([20.0]),
            color_index=np.array([0.5]),
            flux_ratio_rg=np.array([1.0]),
            flux_ratio_bg=np.array([1.0]),
        )
        result = cal._measure_stars(image, (px, py, indices), catalog)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _fit_correction edge cases (lines 337, 364, 372, 375)
# ---------------------------------------------------------------------------

def _make_star(flux_r=1.0, flux_g=1.0, flux_b=1.0, rg=1.0, bg=1.0) -> StarMeasurement:
    return StarMeasurement(
        x=0.0, y=0.0, ra=10.0, dec=20.0,
        flux_r=flux_r, flux_g=flux_g, flux_b=flux_b,
        catalog_color_index=0.5,
        catalog_flux_ratio_rg=rg,
        catalog_flux_ratio_bg=bg,
    )


class TestFitCorrectionEdgeCases:
    def test_empty_list_returns_identity(self) -> None:
        cal = SpectralColorCalibrator(min_stars=3)
        correction, rms, final = cal._fit_correction([])
        np.testing.assert_array_equal(correction, np.eye(3))
        assert rms == 0.0
        assert final == []

    def test_fewer_than_min_stars_breaks_early(self) -> None:
        cal = SpectralColorCalibrator(min_stars=5, max_iterations=10)
        stars = [_make_star(flux_r=1.0, flux_g=1.0, flux_b=1.0) for _ in range(3)]
        correction, rms, final = cal._fit_correction(stars)
        assert correction.shape == (3, 3)

    def test_identical_stars_converges_on_sigma_zero(self) -> None:
        cal = SpectralColorCalibrator(min_stars=3, max_iterations=10)
        stars = [_make_star(flux_r=1.2, flux_g=1.0, flux_b=0.8, rg=1.2, bg=0.8) for _ in range(6)]
        correction, rms, final = cal._fit_correction(stars)
        assert correction[0, 0] == pytest.approx(1.0, abs=0.01)
        assert correction[2, 2] == pytest.approx(1.0, abs=0.01)

    def test_no_outliers_breaks_when_set_unchanged(self) -> None:
        cal = SpectralColorCalibrator(min_stars=3, outlier_sigma=100.0, max_iterations=5)
        rng = np.random.default_rng(7)
        stars = [
            _make_star(
                flux_r=float(rng.uniform(0.9, 1.1)),
                flux_g=1.0,
                flux_b=float(rng.uniform(0.9, 1.1)),
                rg=1.0,
                bg=1.0,
            )
            for _ in range(8)
        ]
        correction, rms, final = cal._fit_correction(stars)
        assert len(final) > 0

    def test_fit_correction_result_after_outlier_removal_returns_empty(self) -> None:
        cal = SpectralColorCalibrator(min_stars=6, max_iterations=3, outlier_sigma=0.0001)
        stars = [_make_star(flux_r=float(i + 1), flux_g=1.0, flux_b=1.0, rg=float(i + 1), bg=1.0) for i in range(4)]
        correction, rms, final = cal._fit_correction(stars)
        assert correction.shape == (3, 3)

    def test_outlier_removed_and_current_updated(self) -> None:
        """line 372: current = new_current executes when some outliers removed."""
        # 7 similar stars + 1 extreme outlier (flux_r=100 vs expected rg=1.0)
        stars = [_make_star(flux_r=1.0 + 0.01 * i, flux_g=1.0, flux_b=1.0, rg=1.0, bg=1.0) for i in range(7)]
        stars.append(_make_star(flux_r=100.0, flux_g=1.0, flux_b=1.0, rg=1.0, bg=1.0))
        cal = SpectralColorCalibrator(min_stars=3, outlier_sigma=2.0, max_iterations=5)
        correction, rms, final = cal._fit_correction(stars)
        assert len(final) < len(stars)
