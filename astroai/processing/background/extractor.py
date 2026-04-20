"""Background extraction for astrophotography via tile-sampled modeling."""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import median_filter

__all__ = ["BackgroundExtractor", "ModelMethod"]

logger = logging.getLogger(__name__)


class ModelMethod(Enum):
    POLYNOMIAL = auto()
    RBF = auto()


class BackgroundExtractor:
    """Automatic background modeling from tile-sampled statistics.

    Divides the image into tiles, computes robust statistics per tile
    (rejecting star-dominated regions), and interpolates a smooth
    background model via 2D polynomial or RBF.
    """

    def __init__(
        self,
        tile_size: int = 64,
        method: ModelMethod = ModelMethod.RBF,
        poly_degree: int = 3,
        rbf_kernel: str = "thin_plate_spline",
        sigma_clip: float = 2.5,
        star_rejection_percentile: float = 80.0,
    ) -> None:
        self._tile_size = tile_size
        self._method = method
        self._poly_degree = poly_degree
        self._rbf_kernel = rbf_kernel
        self._sigma_clip = sigma_clip
        self._star_reject_pct = star_rejection_percentile

    def extract(
        self, frame: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Extract background model from a single-channel or RGB frame."""
        if frame.ndim == 3:
            channels = [frame[..., c] for c in range(frame.shape[2])]
            bg_channels = [self._extract_channel(ch) for ch in channels]
            return np.stack(bg_channels, axis=-1)
        return self._extract_channel(frame)

    def _extract_channel(
        self, channel: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        h, w = channel.shape
        samples_y, samples_x, samples_val = self._sample_tiles(channel)

        if len(samples_val) < 4:
            logger.warning("Too few background samples, returning median flat")
            return np.full_like(channel, np.median(channel))

        if self._method == ModelMethod.POLYNOMIAL:
            return self._fit_polynomial(samples_y, samples_x, samples_val, h, w)
        return self._fit_rbf(samples_y, samples_x, samples_val, h, w)

    def _sample_tiles(
        self, channel: NDArray[np.floating[Any]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        h, w = channel.shape
        ts = self._tile_size
        centers_y = []
        centers_x = []
        values = []

        smoothed = median_filter(channel, size=5)

        for y0 in range(0, h, ts):
            for x0 in range(0, w, ts):
                y1 = min(y0 + ts, h)
                x1 = min(x0 + ts, w)
                tile = smoothed[y0:y1, x0:x1]

                if tile.size == 0:
                    continue

                val = self._robust_tile_value(tile)
                if val is not None:
                    centers_y.append((y0 + y1) / 2.0)
                    centers_x.append((x0 + x1) / 2.0)
                    values.append(val)

        return (
            np.array(centers_y, dtype=np.float64),
            np.array(centers_x, dtype=np.float64),
            np.array(values, dtype=np.float64),
        )

    def _robust_tile_value(
        self, tile: NDArray[np.floating[Any]]
    ) -> float | None:
        flat = tile.ravel().astype(np.float64)
        threshold = np.percentile(flat, self._star_reject_pct)
        below = flat[flat <= threshold]

        if below.size < 4:
            return None

        mean = float(np.mean(below))
        std = float(np.std(below))
        if std < 1e-10:
            return mean

        clipped = below[np.abs(below - mean) <= self._sigma_clip * std]
        if clipped.size < 2:
            return mean

        return float(np.median(clipped))

    def _fit_polynomial(
        self,
        sy: NDArray[np.float64],
        sx: NDArray[np.float64],
        sv: NDArray[np.float64],
        h: int,
        w: int,
    ) -> NDArray[np.floating[Any]]:
        ny = sy / max(h - 1, 1)
        nx = sx / max(w - 1, 1)

        degree = self._poly_degree
        cols = []
        for dy in range(degree + 1):
            for dx in range(degree + 1 - dy):
                cols.append((ny ** dy) * (nx ** dx))
        A = np.column_stack(cols)

        coeffs, _, _, _ = np.linalg.lstsq(A, sv, rcond=None)

        yy, xx = np.mgrid[0:h, 0:w]
        yy_n = yy.astype(np.float64) / max(h - 1, 1)
        xx_n = xx.astype(np.float64) / max(w - 1, 1)

        result = np.zeros((h, w), dtype=np.float64)
        idx = 0
        for dy in range(degree + 1):
            for dx in range(degree + 1 - dy):
                result += coeffs[idx] * (yy_n ** dy) * (xx_n ** dx)
                idx += 1

        return result

    def _fit_rbf(
        self,
        sy: NDArray[np.float64],
        sx: NDArray[np.float64],
        sv: NDArray[np.float64],
        h: int,
        w: int,
    ) -> NDArray[np.floating[Any]]:
        points = np.column_stack([sy / max(h - 1, 1), sx / max(w - 1, 1)])
        interp = RBFInterpolator(points, sv, kernel=self._rbf_kernel)

        yy, xx = np.mgrid[0:h, 0:w]
        grid = np.column_stack([
            yy.ravel().astype(np.float64) / max(h - 1, 1),
            xx.ravel().astype(np.float64) / max(w - 1, 1),
        ])

        result = interp(grid).reshape(h, w)
        return result
