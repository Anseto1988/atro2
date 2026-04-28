from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


class AperturePhotometry:
    def measure(
        self,
        image: NDArray[np.floating],
        x: float,
        y: float,
        radius: float | None = None,
    ) -> float:
        if radius is None:
            radius = self._estimate_fwhm(image, x, y)

        yi, xi = int(round(y)), int(round(x))
        h, w = image.shape[:2]
        r_int = int(np.ceil(radius))

        y_lo = max(yi - r_int, 0)
        y_hi = min(yi + r_int + 1, h)
        x_lo = max(xi - r_int, 0)
        x_hi = min(xi + r_int + 1, w)

        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

        inner_r = 1.5 * radius
        outer_r = 2.5 * radius

        annulus_mask = (dist >= inner_r) & (dist <= outer_r)
        cutout = image[y_lo:y_hi, x_lo:x_hi]
        if image.ndim == 3:
            cutout = np.mean(cutout, axis=-1)

        bg = float(np.median(cutout[annulus_mask])) if np.any(annulus_mask) else 0.0

        star_mask = dist <= radius
        flux = float(np.sum(cutout[star_mask] - bg))
        return max(flux, 1e-10)

    @staticmethod
    def _estimate_fwhm(
        image: NDArray[np.floating], x: float, y: float, box: int = 15
    ) -> float:
        if image.ndim == 3:
            image = np.mean(image, axis=-1)

        h, w = image.shape
        yi, xi = int(round(y)), int(round(x))
        half = box // 2

        y_lo = max(yi - half, 0)
        y_hi = min(yi + half + 1, h)
        x_lo = max(xi - half, 0)
        x_hi = min(xi + half + 1, w)

        stamp = image[y_lo:y_hi, x_lo:x_hi].astype(float)
        bg = np.median(stamp)
        stamp = stamp - bg

        peak = stamp.max()
        if peak <= 0:
            return 3.0

        half_max = peak / 2.0
        above = stamp >= half_max
        n_pixels = float(np.sum(above))
        fwhm = 2.0 * np.sqrt(n_pixels / np.pi)
        return max(fwhm, 1.5)
