"""Star-detection based frame aligner using LoG blob detection."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from scipy.ndimage import gaussian_laplace, label, center_of_mass
from scipy.ndimage import shift as ndimage_shift

from astroai.engine.registration.aligner import FrameAligner

__all__ = ["StarAligner"]

logger = logging.getLogger(__name__)

_MIN_STARS = 3


class StarAligner:
    """Aligns frames via LoG star detection; falls back to phase correlation.

    Falls back to :class:`FrameAligner` when fewer than 3 stars are detected.
    """

    def __init__(
        self,
        upsample_factor: int = 10,
        log_sigma: float = 2.0,
        detection_threshold: float = 0.01,
        max_stars: int = 200,
    ) -> None:
        self.upsample_factor = upsample_factor
        self.log_sigma = log_sigma
        self.detection_threshold = detection_threshold
        self.max_stars = max_stars
        self._fallback = FrameAligner(upsample_factor=upsample_factor)

    def align(
        self,
        reference: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_gray = _to_grayscale(reference)
        tgt_gray = _to_grayscale(target)

        ref_stars = self._detect_stars(ref_gray)
        tgt_stars = self._detect_stars(tgt_gray)

        if len(ref_stars) >= _MIN_STARS and len(tgt_stars) >= _MIN_STARS:
            dy, dx = self._match_shift(ref_stars, tgt_stars)
            logger.debug(
                "Star alignment: %d/%d stars, shift=(%.2f, %.2f)",
                len(ref_stars), len(tgt_stars), dy, dx,
            )
            aligned = _apply_shift(target, dy, dx)
            transform = np.array(
                [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            return aligned, transform

        logger.debug(
            "Star fallback: ref=%d tgt=%d stars — using phase correlation",
            len(ref_stars), len(tgt_stars),
        )
        return self._fallback.align(reference, target)

    def align_batch(
        self,
        reference: np.ndarray,
        targets: list[np.ndarray],
    ) -> list[np.ndarray]:
        return [self.align(reference, t)[0] for t in targets]

    def _detect_stars(self, gray: np.ndarray) -> np.ndarray:
        """Return (N, 2) array of (y, x) star centroids via LoG detection."""
        normalized = _normalize(gray)
        blob = -gaussian_laplace(normalized, sigma=self.log_sigma)
        threshold = self.detection_threshold * blob.max() if blob.max() > 0 else 0.0
        mask = blob > threshold
        labeled, n_features = label(mask)
        if n_features == 0:
            return np.empty((0, 2), dtype=np.float64)

        indices = list(range(1, n_features + 1))
        centroids = center_of_mass(blob, labeled, indices)
        stars = np.array(centroids, dtype=np.float64)

        # Sort by response strength, keep top max_stars
        responses = np.array([blob[labeled == i].max() for i in indices])
        order = np.argsort(-responses)
        return stars[order[: self.max_stars]]

    @staticmethod
    def _match_shift(
        ref_stars: np.ndarray,
        tgt_stars: np.ndarray,
    ) -> tuple[float, float]:
        """Nearest-neighbour match; return median shift (dy, dx)."""
        shifts: list[tuple[float, float]] = []
        for cy, cx in tgt_stars:
            dists = np.hypot(ref_stars[:, 0] - cy, ref_stars[:, 1] - cx)
            nearest = int(np.argmin(dists))
            if dists[nearest] < 50.0:
                shifts.append((
                    ref_stars[nearest, 0] - cy,
                    ref_stars[nearest, 1] - cx,
                ))
        if len(shifts) < _MIN_STARS:
            return 0.0, 0.0
        arr = np.array(shifts)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))


def _normalize(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    if hi == lo:
        return np.zeros_like(img, dtype=np.float64)
    return cast(np.ndarray, ((img - lo) / (hi - lo)).astype(np.float64))


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float64)
    weights = np.array([0.2989, 0.5870, 0.1140])
    return cast(np.ndarray, np.dot(frame[..., :3], weights).astype(np.float64))


def _apply_shift(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    shift_vec: tuple[float, ...] = (-dy, -dx, 0.0) if image.ndim == 3 else (-dy, -dx)
    return cast(np.ndarray, ndimage_shift(
        image, shift_vec, order=3, mode="constant", cval=0.0
    ))
