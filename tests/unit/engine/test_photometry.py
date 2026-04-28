"""Tests for the photometry engine package."""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroai.astrometry.catalog import WcsSolution
from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.engine.photometry.aperture import AperturePhotometry
from astroai.engine.photometry.calibration import MagnitudeCalibrator
from astroai.engine.photometry.export import PhotometryExporter
from astroai.engine.photometry.models import PhotometryResult, StarMeasurement
from astroai.engine.photometry.pipeline_step import PhotometryStep


def _make_wcs() -> WcsSolution:
    scale_deg = 1.0 / 3600.0
    return WcsSolution(
        ra_center=180.0,
        dec_center=45.0,
        pixel_scale_arcsec=1.0,
        rotation_deg=0.0,
        fov_width_deg=0.1,
        fov_height_deg=0.1,
        cd_matrix=(scale_deg, 0.0, 0.0, scale_deg),
        crpix1=128.0,
        crpix2=128.0,
    )


def _make_synthetic_image(size: int = 256, n_stars: int = 10) -> np.ndarray:
    rng = np.random.RandomState(42)
    img = rng.normal(100.0, 5.0, (size, size)).astype(np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_stars):
        cx = rng.uniform(20, size - 20)
        cy = rng.uniform(20, size - 20)
        flux = rng.uniform(500, 5000)
        sigma = rng.uniform(1.5, 3.0)
        img += flux * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return img


class TestModels:
    def test_star_measurement_defaults(self):
        s = StarMeasurement(star_id=0, ra=10.0, dec=20.0, x_pixel=5.0, y_pixel=6.0, instr_mag=-8.0)
        assert s.catalog_mag == 0.0
        assert s.cal_mag == 0.0
        assert s.residual == 0.0

    def test_photometry_result_defaults(self):
        r = PhotometryResult()
        assert r.stars == []
        assert r.r_squared == 0.0
        assert r.n_matched == 0


class TestAperturePhotometry:
    def test_measure_gaussian_star(self):
        img = np.zeros((64, 64), dtype=np.float64)
        yy, xx = np.mgrid[0:64, 0:64]
        img += 1000.0 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 2.0 ** 2))

        ap = AperturePhotometry()
        flux = ap.measure(img, 32.0, 32.0, radius=5.0)
        assert flux > 0

    def test_measure_auto_radius(self):
        img = np.zeros((64, 64), dtype=np.float64)
        yy, xx = np.mgrid[0:64, 0:64]
        img += 500.0 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 2.5 ** 2))

        ap = AperturePhotometry()
        flux = ap.measure(img, 32.0, 32.0)
        assert flux > 0

    def test_measure_3d_image(self):
        img = np.zeros((64, 64, 3), dtype=np.float64)
        yy, xx = np.mgrid[0:64, 0:64]
        for c in range(3):
            img[:, :, c] = 800.0 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 2.0 ** 2))

        ap = AperturePhotometry()
        flux = ap.measure(img, 32.0, 32.0, radius=5.0)
        assert flux > 0

    def test_estimate_fwhm_dark_stamp_returns_default(self):
        """Cover line 66: peak <= 0 → return 3.0."""
        ap = AperturePhotometry()
        # All-zero image: after bg subtraction peak == 0 → returns 3.0
        img = np.zeros((64, 64), dtype=np.float64)
        fwhm = ap._estimate_fwhm(img, 32.0, 32.0)
        assert fwhm == pytest.approx(3.0)


class TestCalibration:
    def test_ridge_linear_fit(self):
        rng = np.random.RandomState(0)
        instr = rng.uniform(-10, -5, 50)
        catalog = 2.0 * instr + 5.0 + rng.normal(0, 0.01, 50)

        cal = MagnitudeCalibrator(alpha=1.0)
        cal.fit(instr, catalog)
        assert cal.r_squared >= 0.95

        predicted = cal.predict(instr)
        assert np.allclose(predicted, catalog, atol=0.5)

    def test_predict_before_fit_raises(self):
        cal = MagnitudeCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.predict(np.array([1.0]))


class TestExport:
    def test_to_csv(self, tmp_path: Path):
        result = PhotometryResult(
            stars=[
                StarMeasurement(0, 10.0, 20.0, 5.0, 6.0, -8.0, 12.0, 12.1, 0.1),
                StarMeasurement(1, 11.0, 21.0, 7.0, 8.0, -7.5, 11.5, 11.6, 0.1),
            ],
            r_squared=0.98,
            n_matched=2,
        )
        out = PhotometryExporter().to_csv(result, tmp_path / "phot.csv")
        assert out.exists()
        with out.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "star_id" in header
            rows = list(reader)
            assert len(rows) == 2

    def test_to_fits(self, tmp_path: Path):
        from astropy.io import fits

        result = PhotometryResult(
            stars=[StarMeasurement(0, 10.0, 20.0, 5.0, 6.0, -8.0, 12.0, 12.1, 0.1)],
            r_squared=0.99,
            n_matched=1,
        )
        out = PhotometryExporter().to_fits(result, tmp_path / "phot.fits")
        assert out.exists()
        with fits.open(out) as hdul:
            assert hdul[1].header["EXTNAME"] == "PHOTOMETRY"
            assert len(hdul[1].data) == 1


class TestPipelineStep:
    def test_stage_is_photometry(self):
        step = PhotometryStep()
        assert step.stage == PipelineStage.PHOTOMETRY
        assert step.name == "Photometry"

    def test_missing_wcs_fail_silently(self):
        ctx = PipelineContext()
        ctx.result = np.zeros((64, 64), dtype=np.float64)
        step = PhotometryStep(fail_silently=True)
        result_ctx = step.execute(ctx)
        assert "photometry_result" not in result_ctx.metadata

    def test_missing_wcs_raises(self):
        ctx = PipelineContext()
        ctx.result = np.zeros((64, 64), dtype=np.float64)
        step = PhotometryStep(fail_silently=False)
        with pytest.raises(RuntimeError, match="no WCS"):
            step.execute(ctx)

    def test_missing_image_fail_silently(self):
        ctx = PipelineContext()
        ctx.metadata["wcs_solution"] = _make_wcs()
        step = PhotometryStep(fail_silently=True)
        result_ctx = step.execute(ctx)
        assert "photometry_result" not in result_ctx.metadata

    def test_image_from_images_list(self):
        """Cover line 58: image taken from context.images[0] when result is None."""
        from unittest.mock import MagicMock
        mock_engine = MagicMock()
        mock_engine.run.return_value = MagicMock(n_matched=0, r_squared=0.0)
        ctx = PipelineContext()
        ctx.images = [np.zeros((32, 32), dtype=np.float64)]
        ctx.metadata["wcs_solution"] = _make_wcs()
        step = PhotometryStep(engine=mock_engine, fail_silently=True)
        result_ctx = step.execute(ctx)
        mock_engine.run.assert_called_once()

    def test_missing_image_raises_when_not_fail_silently(self):
        """Cover line 65: raise RuntimeError when no image and fail_silently=False."""
        ctx = PipelineContext()
        ctx.metadata["wcs_solution"] = _make_wcs()
        step = PhotometryStep(fail_silently=False)
        with pytest.raises(RuntimeError, match="no image"):
            step.execute(ctx)

    def test_engine_exception_fail_silently(self):
        """Cover lines 75-76: engine raises → logged, context returned unchanged."""
        from unittest.mock import MagicMock
        mock_engine = MagicMock()
        mock_engine.run.side_effect = RuntimeError("engine crash")
        ctx = PipelineContext()
        ctx.result = np.zeros((32, 32), dtype=np.float64)
        ctx.metadata["wcs_solution"] = _make_wcs()
        step = PhotometryStep(engine=mock_engine, fail_silently=True)
        result_ctx = step.execute(ctx)
        assert "photometry_result" not in result_ctx.metadata

    def test_engine_exception_raises_when_not_fail_silently(self):
        """Cover lines 77-78: engine raises → re-raised when fail_silently=False."""
        from unittest.mock import MagicMock
        mock_engine = MagicMock()
        mock_engine.run.side_effect = RuntimeError("engine crash")
        ctx = PipelineContext()
        ctx.result = np.zeros((32, 32), dtype=np.float64)
        ctx.metadata["wcs_solution"] = _make_wcs()
        step = PhotometryStep(engine=mock_engine, fail_silently=False)
        with pytest.raises(RuntimeError, match="engine crash"):
            step.execute(ctx)


class TestPipelineStageEnum:
    def test_photometry_after_astrometry(self):
        stages = list(PipelineStage)
        astro_idx = stages.index(PipelineStage.ASTROMETRY)
        phot_idx = stages.index(PipelineStage.PHOTOMETRY)
        assert phot_idx == astro_idx + 1


class TestCatalogClients:
    def test_gaia_query_success(self):
        from astroai.engine.photometry.catalog import GAIACatalogClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "metadata": [{"name": "ra"}, {"name": "dec"}, {"name": "phot_g_mean_mag"}],
            "data": [[180.1, 45.1, 12.3], [180.2, 45.2, 13.0]],
        }
        mock_resp.raise_for_status = MagicMock()

        client = GAIACatalogClient(fail_silently=False)
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
            stars = client.query(180.0, 45.0, 0.5)

        assert len(stars) == 2
        assert stars[0]["ra"] == 180.1
        assert stars[0]["phot_g_mean_mag"] == 12.3

    def test_gaia_query_fail_silently(self):
        from astroai.engine.photometry.catalog import GAIACatalogClient

        client = GAIACatalogClient(fail_silently=True)
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.side_effect = Exception("timeout")
            stars = client.query(180.0, 45.0, 0.5)
        assert stars == []

    def test_aavso_query_success(self):
        from astroai.engine.photometry.catalog import AAVSOCatalogClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "VSXObjects": {"VSXObject": [{"Name": "RR Lyr", "Period": "0.566839"}]}
        }
        mock_resp.raise_for_status = MagicMock()

        client = AAVSOCatalogClient(fail_silently=False)
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
            stars = client.query(180.0, 45.0, 0.5)

        assert len(stars) == 1
        assert stars[0]["Name"] == "RR Lyr"

    def test_aavso_query_fail_silently(self):
        from astroai.engine.photometry.catalog import AAVSOCatalogClient

        client = AAVSOCatalogClient(fail_silently=True)
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.side_effect = Exception("network")
            stars = client.query(180.0, 45.0, 0.5)
        assert stars == []

    def test_gaia_query_raises_when_not_fail_silently(self):
        """Cover line 48: GAIA exception re-raised when fail_silently=False."""
        from astroai.engine.photometry.catalog import GAIACatalogClient

        client = GAIACatalogClient(fail_silently=False)
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.side_effect = Exception("timeout")
            with pytest.raises(Exception, match="timeout"):
                client.query(180.0, 45.0, 0.5)

    def test_aavso_query_raises_when_not_fail_silently(self):
        """Cover line 80: AAVSO exception re-raised when fail_silently=False."""
        from astroai.engine.photometry.catalog import AAVSOCatalogClient

        client = AAVSOCatalogClient(fail_silently=False)
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value.get.side_effect = Exception("network")
            with pytest.raises(Exception, match="network"):
                client.query(180.0, 45.0, 0.5)


class TestPhotometryEngine:
    def _make_star_image(self, size: int = 128) -> np.ndarray:
        """Synthetic image with bright Gaussian stars on dark background."""
        rng = np.random.RandomState(7)
        img = rng.normal(10.0, 1.0, (size, size)).astype(np.float64)
        yy, xx = np.mgrid[0:size, 0:size]
        star_positions = [(30, 30), (60, 40), (90, 80), (40, 95), (70, 20)]
        for cy, cx in star_positions:
            img += 2000.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2))
        return img

    def _make_wcs(self) -> WcsSolution:
        scale_deg = 1.0 / 3600.0
        return WcsSolution(
            ra_center=180.0, dec_center=45.0,
            pixel_scale_arcsec=1.0, rotation_deg=0.0,
            fov_width_deg=0.1, fov_height_deg=0.1,
            cd_matrix=(scale_deg, 0.0, 0.0, scale_deg),
            crpix1=64.0, crpix2=64.0,
        )

    def test_run_with_mocked_catalog(self):
        from astroai.engine.photometry.engine import PhotometryEngine

        mock_catalog = MagicMock()
        mock_catalog.query.return_value = [
            {"ra": 180.0083, "dec": 45.0083, "phot_g_mean_mag": 12.5},
            {"ra": 180.0167, "dec": 45.0111, "phot_g_mean_mag": 13.1},
            {"ra": 180.0250, "dec": 45.0222, "phot_g_mean_mag": 11.8},
        ]

        engine = PhotometryEngine(catalog_client=mock_catalog)
        img = self._make_star_image()
        result = engine.run(img, self._make_wcs())

        assert result is not None
        assert result.r_squared >= 0.0

    def test_calibration_path_produces_calibrated_stars(self):
        from astroai.engine.photometry.engine import PhotometryEngine

        wcs = self._make_wcs()
        img = self._make_star_image()

        mock_catalog = MagicMock()
        mock_aperture = MagicMock()
        mock_calibrator = MagicMock()

        fake_fluxes = [5000.0, 3000.0, 1500.0, 800.0, 400.0]
        mock_aperture.measure.side_effect = fake_fluxes

        star_positions_deg_offset = [
            (0.0083, 0.0083, 12.5),
            (0.0167, 0.0111, 13.1),
            (0.0250, 0.0222, 11.8),
            (0.0111, 0.0194, 14.0),
            (0.0194, 0.0028, 14.5),
        ]
        mock_catalog.query.return_value = [
            {"ra": 180.0 + d[0], "dec": 45.0 + d[1], "phot_g_mean_mag": d[2]}
            for d in star_positions_deg_offset
        ]

        import numpy as np
        fake_cal_mags = np.array([12.5, 13.1, 11.8, 14.0, 14.5])
        mock_calibrator.predict.return_value = fake_cal_mags
        mock_calibrator.r_squared = 0.99

        engine = PhotometryEngine(
            catalog_client=mock_catalog,
            aperture=mock_aperture,
            calibrator=mock_calibrator,
        )

        with patch.object(engine, "_detect_stars", return_value=[
            (30.0, 30.0), (60.0, 40.0), (90.0, 80.0), (40.0, 95.0), (70.0, 20.0)
        ]):
            with patch.object(engine, "_match_stars", return_value=[
                {"x": 30.0, "y": 30.0, "ra": 180.008, "dec": 45.008,
                 "instr_mag": -9.0, "catalog_mag": 12.5},
                {"x": 60.0, "y": 40.0, "ra": 180.016, "dec": 45.011,
                 "instr_mag": -8.7, "catalog_mag": 13.1},
                {"x": 90.0, "y": 80.0, "ra": 180.025, "dec": 45.022,
                 "instr_mag": -8.4, "catalog_mag": 11.8},
            ]):
                result = engine.run(img, wcs)

        assert result.n_matched == 3
        assert result.r_squared == 0.99
        assert len(result.stars) == 3
        assert result.stars[0].star_id == 0
        assert result.stars[0].cal_mag == fake_cal_mags[0]
        assert result.stars[0].residual == pytest.approx(fake_cal_mags[0] - 12.5)

    def test_fewer_than_three_matches_returns_uncalibrated(self):
        from astroai.engine.photometry.engine import PhotometryEngine

        wcs = self._make_wcs()
        img = self._make_star_image()

        mock_catalog = MagicMock()
        mock_catalog.query.return_value = [
            {"ra": 180.0083, "dec": 45.0083, "phot_g_mean_mag": 12.5},
        ]

        engine = PhotometryEngine(catalog_client=mock_catalog)

        with patch.object(engine, "_detect_stars", return_value=[(30.0, 30.0), (60.0, 40.0)]):
            with patch.object(engine, "_match_stars", return_value=[
                {"x": 30.0, "y": 30.0, "ra": 180.008, "dec": 45.008,
                 "instr_mag": -9.0, "catalog_mag": 12.5},
                {"x": 60.0, "y": 40.0, "ra": 180.016, "dec": 45.011,
                 "instr_mag": -8.7, "catalog_mag": 13.1},
            ]):
                result = engine.run(img, wcs)

        assert result.n_matched == 0

    def test_run_empty_catalog_returns_uncalibrated(self):
        from astroai.engine.photometry.engine import PhotometryEngine

        mock_catalog = MagicMock()
        mock_catalog.query.return_value = []

        engine = PhotometryEngine(catalog_client=mock_catalog)
        img = self._make_star_image()
        result = engine.run(img, self._make_wcs())

        assert result.n_matched == 0

    def test_match_stars_appends_close_matches(self):
        """Cover line 153: _match_stars appends matched star when dist <= tolerance."""
        from astroai.engine.photometry.engine import PhotometryEngine
        from astroai.astrometry.catalog import WcsSolution

        scale_deg = 1.0 / 3600.0
        # crpix1=1, crpix2=1 so pixel (0,0) maps to catalog position (ra_center, dec_center)
        wcs = WcsSolution(
            ra_center=180.0, dec_center=45.0,
            pixel_scale_arcsec=1.0, rotation_deg=0.0,
            fov_width_deg=0.1, fov_height_deg=0.1,
            cd_matrix=(scale_deg, 0.0, 0.0, scale_deg),
            crpix1=1.0, crpix2=1.0,
        )
        # Detection at pixel (0, 0). The catalog star is at ra_center/dec_center
        # => cat_dx=0, cat_dy=0 => cat_px=0+0=0, cat_py=0+0=0 → dist=0 <= 5 px
        xs = np.array([0.0])
        ys = np.array([0.0])
        ras = np.array([180.0])
        decs = np.array([45.0])
        instr_mags = np.array([-9.0])
        catalog_stars = [{"ra": 180.0, "dec": 45.0, "phot_g_mean_mag": 12.5}]
        engine = PhotometryEngine()
        matched = engine._match_stars(xs, ys, ras, decs, instr_mags, catalog_stars, wcs)
        assert len(matched) == 1
        assert matched[0]["catalog_mag"] == pytest.approx(12.5)

    def test_run_blank_image_returns_empty(self):
        from astroai.engine.photometry.engine import PhotometryEngine

        mock_catalog = MagicMock()
        mock_catalog.query.return_value = []

        engine = PhotometryEngine(catalog_client=mock_catalog)
        img = np.zeros((64, 64), dtype=np.float64) + 10.0
        result = engine.run(img, self._make_wcs())

        assert isinstance(result.stars, list)

    def test_run_3d_image(self):
        from astroai.engine.photometry.engine import PhotometryEngine

        mock_catalog = MagicMock()
        mock_catalog.query.return_value = []

        engine = PhotometryEngine(catalog_client=mock_catalog)
        img_2d = self._make_star_image()
        img_3d = np.stack([img_2d, img_2d, img_2d], axis=-1)
        result = engine.run(img_3d, self._make_wcs())
        assert result is not None

    def test_pipeline_step_with_engine(self):
        from astroai.engine.photometry.engine import PhotometryEngine
        from astroai.engine.photometry.pipeline_step import PhotometryStep

        mock_catalog = MagicMock()
        mock_catalog.query.return_value = []

        engine = PhotometryEngine(catalog_client=mock_catalog)
        step = PhotometryStep(engine=engine)

        from astroai.core.pipeline.base import PipelineContext
        ctx = PipelineContext()
        ctx.result = self._make_star_image()
        ctx.metadata["wcs_solution"] = self._make_wcs()

        result_ctx = step.execute(ctx)
        assert "photometry_result" in result_ctx.metadata


class TestImports:
    def test_all_modules_importable(self):
        import astroai.engine.photometry
        import astroai.engine.photometry.aperture
        import astroai.engine.photometry.calibration
        import astroai.engine.photometry.catalog
        import astroai.engine.photometry.engine
        import astroai.engine.photometry.export
        import astroai.engine.photometry.models
        import astroai.engine.photometry.pipeline_step
