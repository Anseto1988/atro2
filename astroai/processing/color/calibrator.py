"""Spectrophotometric color calibration using stellar catalog data.

Implements SPCC-equivalent color calibration by matching observed star
colors against known spectral types from GAIA DR3 or 2MASS catalogs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CalibrationResult",
    "CatalogQueryResult",
    "CatalogSource",
    "SpectralColorCalibrator",
]

logger = logging.getLogger(__name__)


class CatalogSource(Enum):
    GAIA_DR3 = "gaia_dr3"
    TWOMASS = "2mass"


@dataclass(frozen=True)
class StarMeasurement:
    x: float
    y: float
    ra: float
    dec: float
    flux_r: float
    flux_g: float
    flux_b: float
    catalog_color_index: float
    catalog_flux_ratio_rg: float
    catalog_flux_ratio_bg: float


@dataclass(frozen=True)
class CalibrationResult:
    correction_matrix: NDArray[np.floating[Any]]
    stars_used: int
    residual_rms: float
    white_balance: tuple[float, float, float]


@dataclass
class CatalogQueryResult:
    ra: NDArray[np.floating[Any]]
    dec: NDArray[np.floating[Any]]
    color_index: NDArray[np.floating[Any]]
    flux_ratio_rg: NDArray[np.floating[Any]]
    flux_ratio_bg: NDArray[np.floating[Any]]


class SpectralColorCalibrator:
    """Photometric color calibration via stellar spectral references.

    Queries a stellar catalog for stars in the field, measures their
    observed RGB flux ratios, and computes a least-squares correction
    matrix to align sensor colors with known spectral energy distributions.
    """

    def __init__(
        self,
        catalog: CatalogSource = CatalogSource.GAIA_DR3,
        sample_radius_px: int = 8,
        max_iterations: int = 10,
        outlier_sigma: float = 2.5,
        min_stars: int = 5,
    ) -> None:
        self._catalog = catalog
        self._sample_radius = sample_radius_px
        self._max_iterations = max_iterations
        self._outlier_sigma = outlier_sigma
        self._min_stars = min_stars

    @property
    def catalog(self) -> CatalogSource:
        return self._catalog

    @property
    def sample_radius(self) -> int:
        return self._sample_radius

    def calibrate(
        self,
        image: NDArray[np.floating[Any]],
        wcs: Any,
        catalog_data: CatalogQueryResult | None = None,
    ) -> tuple[NDArray[np.floating[Any]], CalibrationResult]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got {image.shape}")

        h, w = image.shape[:2]

        if catalog_data is None:
            catalog_data = self._query_catalog(wcs, w, h)

        star_pixels = self._catalog_to_pixels(catalog_data, wcs, w, h)
        if len(star_pixels[0]) == 0:
            logger.warning("No catalog stars found in field, returning uncalibrated image")
            identity = np.eye(3, dtype=np.float64)
            return image.copy(), CalibrationResult(
                correction_matrix=identity,
                stars_used=0,
                residual_rms=0.0,
                white_balance=(1.0, 1.0, 1.0),
            )

        measurements = self._measure_stars(
            image, star_pixels, catalog_data,
        )

        if len(measurements) < self._min_stars:
            logger.warning(
                "Only %d stars measured (minimum %d), returning uncalibrated",
                len(measurements), self._min_stars,
            )
            identity = np.eye(3, dtype=np.float64)
            return image.copy(), CalibrationResult(
                correction_matrix=identity,
                stars_used=len(measurements),
                residual_rms=0.0,
                white_balance=(1.0, 1.0, 1.0),
            )

        correction, residual, final_measurements = self._fit_correction(measurements)

        calibrated = self._apply_correction(image, correction)

        wb = (float(correction[0, 0]), float(correction[1, 1]), float(correction[2, 2]))
        result = CalibrationResult(
            correction_matrix=correction,
            stars_used=len(final_measurements),
            residual_rms=float(residual),
            white_balance=wb,
        )

        logger.info(
            "Color calibration: %d stars, RMS=%.4f, WB=(%.3f, %.3f, %.3f)",
            result.stars_used, result.residual_rms, *result.white_balance,
        )

        return calibrated, result

    def _query_catalog(
        self, wcs: Any, width: int, height: int,
    ) -> CatalogQueryResult:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        corners_x = np.array([0, width, width, 0], dtype=np.float64)
        corners_y = np.array([0, 0, height, height], dtype=np.float64)
        corners_sky = wcs.pixel_to_world(corners_x, corners_y)

        center_sky = wcs.pixel_to_world(width / 2.0, height / 2.0)
        center_ra = center_sky.ra.deg
        center_dec = center_sky.dec.deg

        corner_ras = corners_sky.ra.deg
        corner_decs = corners_sky.dec.deg
        max_sep = max(
            SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg).separation(
                SkyCoord(ra=r * u.deg, dec=d * u.deg)
            ).deg
            for r, d in zip(corner_ras, corner_decs)
        )
        search_radius = max_sep * 1.1

        if self._catalog == CatalogSource.GAIA_DR3:
            return self._query_gaia(center_ra, center_dec, search_radius)
        return self._query_2mass(center_ra, center_dec, search_radius)

    def _query_gaia(
        self, ra: float, dec: float, radius: float,
    ) -> CatalogQueryResult:
        from astroquery.gaia import Gaia
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        job = Gaia.cone_search_async(coord, radius=radius * u.deg)
        table = job.get_results()

        mask = (
            np.isfinite(table["phot_bp_mean_mag"])
            & np.isfinite(table["phot_rp_mean_mag"])
            & np.isfinite(table["phot_g_mean_mag"])
            & (table["phot_g_mean_mag"] < 16.0)
        )
        table = table[mask]

        bp_rp = np.array(table["bp_rp"], dtype=np.float64)

        flux_g = 10.0 ** (-0.4 * np.array(table["phot_g_mean_mag"], dtype=np.float64))
        flux_bp = 10.0 ** (-0.4 * np.array(table["phot_bp_mean_mag"], dtype=np.float64))
        flux_rp = 10.0 ** (-0.4 * np.array(table["phot_rp_mean_mag"], dtype=np.float64))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_rg = np.where(flux_g > 0, flux_rp / flux_g, 1.0)
            ratio_bg = np.where(flux_g > 0, flux_bp / flux_g, 1.0)

        return CatalogQueryResult(
            ra=np.array(table["ra"], dtype=np.float64),
            dec=np.array(table["dec"], dtype=np.float64),
            color_index=bp_rp,
            flux_ratio_rg=ratio_rg,
            flux_ratio_bg=ratio_bg,
        )

    def _query_2mass(
        self, ra: float, dec: float, radius: float,
    ) -> CatalogQueryResult:
        from astroquery.vizier import Vizier
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        vizier = Vizier(columns=["RAJ2000", "DEJ2000", "Jmag", "Hmag", "Kmag"], row_limit=5000)
        result = vizier.query_region(coord, radius=radius * u.deg, catalog="II/246/out")

        if not result:
            return CatalogQueryResult(
                ra=np.array([], dtype=np.float64),
                dec=np.array([], dtype=np.float64),
                color_index=np.array([], dtype=np.float64),
                flux_ratio_rg=np.array([], dtype=np.float64),
                flux_ratio_bg=np.array([], dtype=np.float64),
            )

        table = result[0]
        mask = (
            np.isfinite(table["Jmag"])
            & np.isfinite(table["Hmag"])
            & np.isfinite(table["Kmag"])
        )
        table = table[mask]

        j_h = np.array(table["Jmag"] - table["Hmag"], dtype=np.float64)

        flux_j = 10.0 ** (-0.4 * np.array(table["Jmag"], dtype=np.float64))
        flux_h = 10.0 ** (-0.4 * np.array(table["Hmag"], dtype=np.float64))
        flux_k = 10.0 ** (-0.4 * np.array(table["Kmag"], dtype=np.float64))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_rg = np.where(flux_h > 0, flux_k / flux_h, 1.0)
            ratio_bg = np.where(flux_h > 0, flux_j / flux_h, 1.0)

        return CatalogQueryResult(
            ra=np.array(table["RAJ2000"], dtype=np.float64),
            dec=np.array(table["DEJ2000"], dtype=np.float64),
            color_index=j_h,
            flux_ratio_rg=ratio_rg,
            flux_ratio_bg=ratio_bg,
        )

    def _catalog_to_pixels(
        self,
        catalog: CatalogQueryResult,
        wcs: Any,
        width: int,
        height: int,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.intp]]:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        if len(catalog.ra) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.intp)

        coords = SkyCoord(ra=catalog.ra * u.deg, dec=catalog.dec * u.deg)
        px, py = wcs.world_to_pixel(coords)
        px = np.asarray(px, dtype=np.float64)
        py = np.asarray(py, dtype=np.float64)

        margin = self._sample_radius
        in_bounds = (
            (px >= margin) & (px < width - margin)
            & (py >= margin) & (py < height - margin)
        )

        return px[in_bounds], py[in_bounds], np.where(in_bounds)[0]

    def _measure_stars(
        self,
        image: NDArray[np.floating[Any]],
        star_pixels: tuple[Any, ...],
        catalog: CatalogQueryResult,
    ) -> list[StarMeasurement]:
        px, py, indices = star_pixels
        r = self._sample_radius
        measurements: list[StarMeasurement] = []

        for i in range(len(px)):
            cx, cy = int(round(px[i])), int(round(py[i]))
            idx = indices[i]

            yy, xx = np.ogrid[cy - r : cy + r + 1, cx - r : cx + r + 1]
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            aperture = dist_sq <= r * r

            patch = image[cy - r : cy + r + 1, cx - r : cx + r + 1]
            if patch.shape[0] != 2 * r + 1 or patch.shape[1] != 2 * r + 1:
                continue

            flux_r = float(np.sum(patch[:, :, 0][aperture]))
            flux_g = float(np.sum(patch[:, :, 1][aperture]))
            flux_b = float(np.sum(patch[:, :, 2][aperture]))

            if flux_g <= 0 or flux_r <= 0 or flux_b <= 0:
                continue

            measurements.append(StarMeasurement(
                x=px[i], y=py[i],
                ra=catalog.ra[idx], dec=catalog.dec[idx],
                flux_r=flux_r, flux_g=flux_g, flux_b=flux_b,
                catalog_color_index=catalog.color_index[idx],
                catalog_flux_ratio_rg=catalog.flux_ratio_rg[idx],
                catalog_flux_ratio_bg=catalog.flux_ratio_bg[idx],
            ))

        return measurements

    def _fit_correction(
        self, measurements: list[StarMeasurement],
    ) -> tuple[NDArray[np.floating[Any]], float, list[StarMeasurement]]:
        current = list(measurements)

        for iteration in range(self._max_iterations):
            if len(current) < self._min_stars:
                break

            observed = np.array([
                [m.flux_r / m.flux_g, m.flux_b / m.flux_g]
                for m in current
            ])
            expected = np.array([
                [m.catalog_flux_ratio_rg, m.catalog_flux_ratio_bg]
                for m in current
            ])

            with np.errstate(divide="ignore", invalid="ignore"):
                ratios_r = np.where(observed[:, 0] > 0, expected[:, 0] / observed[:, 0], 1.0)
                ratios_b = np.where(observed[:, 1] > 0, expected[:, 1] / observed[:, 1], 1.0)

            scale_r = float(np.median(ratios_r))
            scale_b = float(np.median(ratios_b))

            residuals_r = (ratios_r - scale_r) ** 2
            residuals_b = (ratios_b - scale_b) ** 2
            total_residual = np.sqrt(residuals_r + residuals_b)

            rms = float(np.sqrt(np.mean(total_residual ** 2)))
            sigma = float(np.std(total_residual))
            median_residual = float(np.median(total_residual))

            if sigma < 1e-6 or median_residual < 1e-6:
                break

            threshold = max(self._outlier_sigma * sigma, median_residual * 3.0)
            inlier_mask = total_residual < threshold
            new_current = [m for m, keep in zip(current, inlier_mask) if keep]

            if len(new_current) == len(current):
                break
            current = new_current

        if not current:
            return np.eye(3, dtype=np.float64), 0.0, []

        observed = np.array([
            [m.flux_r / m.flux_g, m.flux_b / m.flux_g]
            for m in current
        ])
        expected = np.array([
            [m.catalog_flux_ratio_rg, m.catalog_flux_ratio_bg]
            for m in current
        ])

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios_r = np.where(observed[:, 0] > 0, expected[:, 0] / observed[:, 0], 1.0)
            ratios_b = np.where(observed[:, 1] > 0, expected[:, 1] / observed[:, 1], 1.0)

        scale_r = float(np.median(ratios_r))
        scale_b = float(np.median(ratios_b))

        correction = np.diag([scale_r, 1.0, scale_b]).astype(np.float64)

        final_residuals = np.sqrt(
            (ratios_r - scale_r) ** 2 + (ratios_b - scale_b) ** 2
        )
        rms = float(np.sqrt(np.mean(final_residuals ** 2)))

        return correction, rms, current

    def _apply_correction(
        self,
        image: NDArray[np.floating[Any]],
        correction: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        result = image.astype(np.float64).copy()
        for ch in range(3):
            result[:, :, ch] *= correction[ch, ch]
        return result.astype(image.dtype)
