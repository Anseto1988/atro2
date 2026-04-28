from __future__ import annotations

import numpy as np
import pytest

from astroai.engine.photometry.aperture import AperturePhotometry


def _make_gaussian_star(
    shape: tuple[int, int] = (64, 64),
    center: tuple[float, float] = (32.0, 32.0),
    amplitude: float = 1000.0,
    sigma: float = 3.0,
) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    cx, cy = center
    img = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return img.astype(np.float32)


class TestAperturePhotometry:
    def test_measure_gaussian_star(self) -> None:
        img = _make_gaussian_star()
        ap = AperturePhotometry()
        flux = ap.measure(img, 32.0, 32.0, radius=6.0)
        assert flux > 0.0

    def test_measure_returns_positive_flux(self) -> None:
        img = _make_gaussian_star(amplitude=5000.0)
        ap = AperturePhotometry()
        flux = ap.measure(img, 32.0, 32.0, radius=8.0)
        assert flux > 1000.0

    def test_annulus_background_subtraction(self) -> None:
        bg_level = 200.0
        img = _make_gaussian_star(amplitude=5000.0, sigma=2.0) + bg_level
        ap = AperturePhotometry()
        flux_with_bg = ap.measure(img, 32.0, 32.0, radius=4.0)

        img_no_bg = _make_gaussian_star(amplitude=5000.0, sigma=2.0)
        flux_no_bg = ap.measure(img_no_bg, 32.0, 32.0, radius=4.0)

        relative_diff = abs(flux_with_bg - flux_no_bg) / flux_no_bg
        assert relative_diff < 0.5

    def test_auto_radius_fallback(self) -> None:
        img = _make_gaussian_star()
        ap = AperturePhotometry()
        flux = ap.measure(img, 32.0, 32.0)
        assert flux > 0.0

    def test_3d_image_averaged(self) -> None:
        img_2d = _make_gaussian_star()
        img_3d = np.stack([img_2d, img_2d, img_2d], axis=-1)
        ap = AperturePhotometry()
        flux = ap.measure(img_3d, 32.0, 32.0, radius=6.0)
        assert flux > 0.0

    def test_edge_star_no_error(self) -> None:
        img = _make_gaussian_star(center=(2.0, 2.0))
        ap = AperturePhotometry()
        flux = ap.measure(img, 2.0, 2.0, radius=3.0)
        assert flux >= 1e-10

    def test_flat_image_minimal_flux(self) -> None:
        img = np.full((64, 64), 100.0, dtype=np.float32)
        ap = AperturePhotometry()
        flux = ap.measure(img, 32.0, 32.0, radius=5.0)
        assert flux >= 1e-10

    def test_fwhm_estimation_peak_zero(self) -> None:
        img = np.zeros((64, 64), dtype=np.float32)
        fwhm = AperturePhotometry._estimate_fwhm(img, 32.0, 32.0)
        assert fwhm == 3.0

    def test_fwhm_estimation_returns_minimum(self) -> None:
        img = np.zeros((64, 64), dtype=np.float32)
        img[32, 32] = 10.0
        fwhm = AperturePhotometry._estimate_fwhm(img, 32.0, 32.0)
        assert fwhm >= 1.5
