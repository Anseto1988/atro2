"""Unit tests for SpectralColorCalibrator with mock catalog data."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.processing.color.calibrator import (
    CalibrationResult,
    CatalogQueryResult,
    CatalogSource,
    SpectralColorCalibrator,
)


def _make_mock_wcs(width: int = 256, height: int = 256) -> Any:
    """Minimal WCS mock that maps pixels to RA/Dec linearly."""
    from unittest.mock import MagicMock
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


def _make_synthetic_rgb(
    width: int = 256,
    height: int = 256,
    star_positions: list[tuple[int, int]] | None = None,
    color_bias: tuple[float, float, float] = (1.2, 1.0, 0.8),
    seed: int = 42,
) -> NDArray[np.floating[Any]]:
    """Create a synthetic RGB image with known color bias and star sources."""
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 2.0, (height, width, 3)).astype(np.float32)

    if star_positions is None:
        star_positions = [
            (64, 64), (192, 64), (128, 128),
            (64, 192), (192, 192), (100, 100),
            (150, 80), (80, 150), (200, 200),
            (50, 128),
        ]

    yy, xx = np.mgrid[0:height, 0:width]
    for cx, cy in star_positions:
        star = 5000.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0**2))
        for ch in range(3):
            img[:, :, ch] += star * color_bias[ch]

    return np.clip(img, 0, None)


def _make_mock_catalog(
    star_positions: list[tuple[int, int]] | None = None,
    width: int = 256,
    height: int = 256,
) -> CatalogQueryResult:
    """Create mock catalog data matching star positions."""
    if star_positions is None:
        star_positions = [
            (64, 64), (192, 64), (128, 128),
            (64, 192), (192, 192), (100, 100),
            (150, 80), (80, 150), (200, 200),
            (50, 128),
        ]

    scale = 1.0 / 3600.0
    ra = np.array([180.0 + (cx - width / 2.0) * scale for cx, _ in star_positions])
    dec = np.array([45.0 + (cy - height / 2.0) * scale for _, cy in star_positions])

    n = len(star_positions)
    return CatalogQueryResult(
        ra=ra,
        dec=dec,
        color_index=np.zeros(n, dtype=np.float64),
        flux_ratio_rg=np.ones(n, dtype=np.float64),
        flux_ratio_bg=np.ones(n, dtype=np.float64),
    )


class TestSpectralColorCalibrator:
    def test_calibrate_corrects_color_bias(self) -> None:
        wcs = _make_mock_wcs()
        image = _make_synthetic_rgb(color_bias=(1.3, 1.0, 0.7))
        catalog = _make_mock_catalog()

        calibrator = SpectralColorCalibrator(
            catalog=CatalogSource.GAIA_DR3,
            sample_radius_px=4,
            min_stars=3,
        )
        calibrated, result = calibrator.calibrate(image, wcs, catalog_data=catalog)

        assert isinstance(result, CalibrationResult)
        assert result.stars_used >= 3
        assert calibrated.shape == image.shape
        assert calibrated.dtype == image.dtype

        r_ratio = result.white_balance[0]
        b_ratio = result.white_balance[2]
        assert r_ratio < 1.0, "Red channel should be scaled down"
        assert b_ratio > 1.0, "Blue channel should be scaled up"

    def test_calibrate_identity_for_neutral_image(self) -> None:
        wcs = _make_mock_wcs()
        image = _make_synthetic_rgb(color_bias=(1.0, 1.0, 1.0))
        catalog = _make_mock_catalog()

        calibrator = SpectralColorCalibrator(min_stars=3, sample_radius_px=4)
        calibrated, result = calibrator.calibrate(image, wcs, catalog_data=catalog)

        np.testing.assert_allclose(result.white_balance[0], 1.0, atol=0.1)
        np.testing.assert_allclose(result.white_balance[2], 1.0, atol=0.1)

    def test_calibrate_no_stars_returns_copy(self) -> None:
        wcs = _make_mock_wcs()
        image = _make_synthetic_rgb()
        empty_catalog = CatalogQueryResult(
            ra=np.array([], dtype=np.float64),
            dec=np.array([], dtype=np.float64),
            color_index=np.array([], dtype=np.float64),
            flux_ratio_rg=np.array([], dtype=np.float64),
            flux_ratio_bg=np.array([], dtype=np.float64),
        )

        calibrator = SpectralColorCalibrator()
        calibrated, result = calibrator.calibrate(image, wcs, catalog_data=empty_catalog)

        assert result.stars_used == 0
        np.testing.assert_array_equal(calibrated, image)

    def test_calibrate_rejects_grayscale(self) -> None:
        wcs = _make_mock_wcs()
        gray = np.zeros((128, 128), dtype=np.float32)

        calibrator = SpectralColorCalibrator()
        with pytest.raises(ValueError, match="Expected.*RGB"):
            calibrator.calibrate(gray, wcs)

    def test_outlier_rejection(self) -> None:
        wcs = _make_mock_wcs(width=512, height=512)
        positions = [
            (80, 80), (200, 80), (320, 80), (440, 80),
            (80, 200), (200, 200), (320, 200), (440, 200),
            (80, 320), (200, 320), (320, 320), (440, 320),
        ]
        image = _make_synthetic_rgb(
            width=512, height=512, star_positions=positions, color_bias=(1.3, 1.0, 0.7),
        )

        catalog = _make_mock_catalog(star_positions=positions, width=512, height=512)

        calibrator = SpectralColorCalibrator(
            min_stars=3, sample_radius_px=4, outlier_sigma=2.0,
        )
        _, result = calibrator.calibrate(image, wcs, catalog_data=catalog)

        assert result.stars_used >= 3

    def test_catalog_source_property(self) -> None:
        cal = SpectralColorCalibrator(catalog=CatalogSource.TWOMASS)
        assert cal.catalog == CatalogSource.TWOMASS

    def test_sample_radius_property(self) -> None:
        cal = SpectralColorCalibrator(sample_radius_px=12)
        assert cal.sample_radius == 12
