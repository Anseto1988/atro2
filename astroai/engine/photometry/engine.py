from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from astroai.astrometry.catalog import WcsSolution, pixel_to_radec
from astroai.engine.photometry.aperture import AperturePhotometry
from astroai.engine.photometry.calibration import MagnitudeCalibrator
from astroai.engine.photometry.catalog import GAIACatalogClient
from astroai.engine.photometry.export import PhotometryExporter
from astroai.engine.photometry.models import PhotometryResult, StarMeasurement

logger = logging.getLogger(__name__)

_MATCH_TOLERANCE_PIX = 5.0


class PhotometryEngine:
    def __init__(
        self,
        catalog_client: GAIACatalogClient | None = None,
        aperture: AperturePhotometry | None = None,
        calibrator: MagnitudeCalibrator | None = None,
        exporter: PhotometryExporter | None = None,
        aperture_radius: float | None = None,
    ) -> None:
        self._catalog = catalog_client or GAIACatalogClient()
        self._aperture = aperture or AperturePhotometry()
        self._calibrator = calibrator or MagnitudeCalibrator()
        self._exporter = exporter or PhotometryExporter()
        self._aperture_radius = aperture_radius

    def run(
        self,
        image: NDArray[np.floating],
        wcs_solution: WcsSolution,
    ) -> PhotometryResult:
        detections = self._detect_stars(image)
        if not detections:
            logger.warning("No stars detected in image")
            return PhotometryResult()

        xs = np.array([d[0] for d in detections])
        ys = np.array([d[1] for d in detections])

        fluxes = np.array([
            self._aperture.measure(image, x, y, self._aperture_radius)
            for x, y in detections
        ])
        instr_mags = -2.5 * np.log10(np.maximum(fluxes, 1e-10))

        ras, decs = pixel_to_radec(wcs_solution, xs, ys)

        fov_radius = max(wcs_solution.fov_width_deg, wcs_solution.fov_height_deg) / 2.0
        catalog_stars = self._catalog.query(
            wcs_solution.ra_center, wcs_solution.dec_center, fov_radius,
        )

        if not catalog_stars:
            logger.warning("No catalog stars returned; skipping calibration")
            return self._build_uncalibrated(xs, ys, ras, decs, instr_mags)

        matched = self._match_stars(
            xs, ys, ras, decs, instr_mags, catalog_stars, wcs_solution,
        )

        if len(matched) < 3:
            logger.warning("Only %d matches — too few for calibration", len(matched))
            return self._build_uncalibrated(xs, ys, ras, decs, instr_mags)

        m_instr = np.array([m["instr_mag"] for m in matched])
        m_cat = np.array([m["catalog_mag"] for m in matched])

        self._calibrator.fit(m_instr, m_cat)
        cal_mags = self._calibrator.predict(m_instr)

        stars = []
        for i, m in enumerate(matched):
            residual = float(cal_mags[i] - m["catalog_mag"])
            stars.append(StarMeasurement(
                star_id=i,
                ra=m["ra"],
                dec=m["dec"],
                x_pixel=m["x"],
                y_pixel=m["y"],
                instr_mag=m["instr_mag"],
                catalog_mag=m["catalog_mag"],
                cal_mag=float(cal_mags[i]),
                residual=residual,
            ))

        return PhotometryResult(
            stars=stars,
            r_squared=self._calibrator.r_squared,
            n_matched=len(stars),
        )

    def _detect_stars(
        self, image: NDArray[np.floating],
    ) -> list[tuple[float, float]]:
        img = image if image.ndim == 2 else np.mean(image, axis=-1)
        img = img.astype(float)

        from scipy.ndimage import gaussian_laplace, label, maximum_position

        log = -gaussian_laplace(img, sigma=2.0)
        threshold = np.mean(log) + 3.0 * np.std(log)
        binary = log > threshold
        labeled, n_features = label(binary)
        if n_features == 0:
            return []

        positions = maximum_position(img, labeled, range(1, n_features + 1))
        return [(float(p[1]), float(p[0])) for p in positions]

    def _match_stars(
        self,
        xs: NDArray[np.floating],
        ys: NDArray[np.floating],
        ras: NDArray[np.floating],
        decs: NDArray[np.floating],
        instr_mags: NDArray[np.floating],
        catalog_stars: list[dict[str, Any]],
        wcs: WcsSolution,
    ) -> list[dict[str, Any]]:
        ra_key = "ra" if "ra" in catalog_stars[0] else "ra_icrs"
        dec_key = "dec" if "dec" in catalog_stars[0] else "dec_icrs"
        mag_key = "phot_g_mean_mag"

        cat_ras = np.array([s[ra_key] for s in catalog_stars], dtype=float)
        cat_decs = np.array([s[dec_key] for s in catalog_stars], dtype=float)
        cat_mags = np.array([s[mag_key] for s in catalog_stars], dtype=float)

        cos_dec = np.cos(np.radians(wcs.dec_center))
        cat_dx = (cat_ras - wcs.ra_center) * cos_dec / wcs.pixel_scale_deg
        cat_dy = (cat_decs - wcs.dec_center) / wcs.pixel_scale_deg
        cat_px = cat_dx + (wcs.crpix1 - 1.0)
        cat_py = cat_dy + (wcs.crpix2 - 1.0)

        det_coords = np.column_stack([xs, ys])
        cat_coords = np.column_stack([cat_px, cat_py])

        tree = KDTree(cat_coords)
        dists, idxs = tree.query(det_coords)

        matched = []
        for i, (dist, cat_i) in enumerate(zip(dists, idxs)):
            if dist <= _MATCH_TOLERANCE_PIX:
                matched.append({
                    "x": float(xs[i]),
                    "y": float(ys[i]),
                    "ra": float(ras[i]),
                    "dec": float(decs[i]),
                    "instr_mag": float(instr_mags[i]),
                    "catalog_mag": float(cat_mags[cat_i]),
                })
        return matched

    def _build_uncalibrated(
        self,
        xs: NDArray[np.floating],
        ys: NDArray[np.floating],
        ras: NDArray[np.floating],
        decs: NDArray[np.floating],
        instr_mags: NDArray[np.floating],
    ) -> PhotometryResult:
        stars = [
            StarMeasurement(
                star_id=i,
                ra=float(ras[i]),
                dec=float(decs[i]),
                x_pixel=float(xs[i]),
                y_pixel=float(ys[i]),
                instr_mag=float(instr_mags[i]),
            )
            for i in range(len(xs))
        ]
        return PhotometryResult(stars=stars, r_squared=0.0, n_matched=0)
