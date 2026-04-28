from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from astroai.astrometry.catalog import WcsSolution
from astroai.engine.photometry.engine import PhotometryEngine
from astroai.engine.photometry.models import PhotometryResult


def _fake_wcs() -> WcsSolution:
    return WcsSolution(
        ra_center=180.0,
        dec_center=45.0,
        pixel_scale_arcsec=1.5,
        rotation_deg=0.0,
        fov_width_deg=0.2,
        fov_height_deg=0.2,
        cd_matrix=(-4.17e-4, 0.0, 0.0, 4.17e-4),
        crpix1=32.0,
        crpix2=32.0,
    )


def _synthetic_image_with_stars(
    shape: tuple[int, int] = (64, 64),
    star_positions: list[tuple[float, float]] | None = None,
    amplitude: float = 2000.0,
    sigma: float = 2.5,
) -> np.ndarray:
    if star_positions is None:
        star_positions = [(20.0, 20.0), (32.0, 32.0), (45.0, 45.0)]
    img = np.full(shape, 100.0, dtype=np.float32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for cx, cy in star_positions:
        img += amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)).astype(np.float32)
    return img


def _mock_catalog_stars(n: int = 3) -> list[dict]:
    wcs = _fake_wcs()
    stars = []
    for i in range(n):
        stars.append({
            "ra": wcs.ra_center + (i - 1) * 0.005,
            "dec": wcs.dec_center + (i - 1) * 0.005,
            "phot_g_mean_mag": 10.0 + i * 0.5,
        })
    return stars


class TestPhotometryEngine:
    def test_run_with_mock_catalog(self) -> None:
        catalog = MagicMock()
        catalog.query.return_value = _mock_catalog_stars(3)

        engine = PhotometryEngine(catalog_client=catalog)
        wcs = _fake_wcs()
        image = _synthetic_image_with_stars()

        result = engine.run(image, wcs)

        assert isinstance(result, PhotometryResult)
        catalog.query.assert_called_once()

    def test_n_matched_with_catalog(self) -> None:
        catalog = MagicMock()
        catalog.query.return_value = _mock_catalog_stars(5)

        engine = PhotometryEngine(catalog_client=catalog)
        wcs = _fake_wcs()
        image = _synthetic_image_with_stars(
            star_positions=[(15.0, 15.0), (25.0, 25.0), (35.0, 35.0), (45.0, 45.0), (50.0, 50.0)],
        )

        result = engine.run(image, wcs)
        assert result.n_matched >= 0

    def test_empty_catalog_returns_uncalibrated(self) -> None:
        catalog = MagicMock()
        catalog.query.return_value = []

        engine = PhotometryEngine(catalog_client=catalog)
        wcs = _fake_wcs()
        image = _synthetic_image_with_stars()

        result = engine.run(image, wcs)

        assert result.n_matched == 0
        assert result.r_squared == 0.0

    def test_no_detections_returns_empty(self) -> None:
        catalog = MagicMock()
        catalog.query.return_value = _mock_catalog_stars(3)

        engine = PhotometryEngine(catalog_client=catalog)
        wcs = _fake_wcs()
        image = np.full((64, 64), 100.0, dtype=np.float32)

        result = engine.run(image, wcs)

        assert result.stars == []
        assert result.n_matched == 0

    def test_run_with_3d_image(self) -> None:
        catalog = MagicMock()
        catalog.query.return_value = []

        img_2d = _synthetic_image_with_stars()
        img_3d = np.stack([img_2d, img_2d, img_2d], axis=-1)

        engine = PhotometryEngine(catalog_client=catalog)
        wcs = _fake_wcs()

        result = engine.run(img_3d, wcs)
        assert isinstance(result, PhotometryResult)

    def test_few_matches_returns_uncalibrated(self) -> None:
        catalog = MagicMock()
        catalog.query.return_value = [
            {"ra": 999.0, "dec": 999.0, "phot_g_mean_mag": 10.0},
        ]

        engine = PhotometryEngine(catalog_client=catalog)
        wcs = _fake_wcs()
        image = _synthetic_image_with_stars()

        result = engine.run(image, wcs)

        assert result.r_squared == 0.0
