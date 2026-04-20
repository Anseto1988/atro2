"""Star detection, separation and reduction for astrophotography."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

__all__ = ["StarManager"]


class StarManager:
    """Starless separation and star reduction for astro images.

    Detects stars via connected-component analysis, generates star masks,
    and supports starless image creation and star size reduction.
    """

    def __init__(
        self,
        detection_sigma: float = 4.0,
        min_star_area: int = 3,
        max_star_area: int = 5000,
        mask_dilation: int = 3,
    ) -> None:
        self._sigma = detection_sigma
        self._min_area = min_star_area
        self._max_area = max_star_area
        self._dilation = mask_dilation

    def create_star_mask(
        self, frame: NDArray[np.floating[Any]]
    ) -> NDArray[np.bool_]:
        """Create a binary mask where True = star pixel."""
        gray = self._to_grayscale(frame)
        mean = gray.mean()
        std = gray.std()
        if std < 1e-8:
            return np.zeros(gray.shape, dtype=np.bool_)

        threshold = mean + self._sigma * std
        binary = gray > threshold
        labeled, n_labels = ndimage.label(binary)

        mask = np.zeros(gray.shape, dtype=np.bool_)
        for i in range(1, n_labels + 1):
            region = labeled == i
            area = int(region.sum())
            if area < self._min_area or area > self._max_area:
                continue
            ys, xs = np.where(region)
            dy = ys.max() - ys.min() + 1
            dx = xs.max() - xs.min() + 1
            if dy == 0 or dx == 0:
                continue
            aspect = max(dy, dx) / max(min(dy, dx), 1)
            if aspect > 4.0:
                continue
            mask |= region

        if self._dilation > 0:
            struct = ndimage.generate_binary_structure(2, 1)
            mask = ndimage.binary_dilation(
                mask, structure=struct, iterations=self._dilation
            )
        return mask

    def separate(
        self, frame: NDArray[np.floating[Any]]
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Separate frame into (starless, stars_only) components."""
        mask = self.create_star_mask(frame)
        starless = self._inpaint_stars(frame, mask)
        stars_only = np.where(
            self._expand_mask(mask, frame.ndim), frame - starless, 0.0
        ).astype(frame.dtype)
        stars_only = np.clip(stars_only, 0.0, None)
        return starless.astype(frame.dtype), stars_only

    def reduce_stars(
        self,
        frame: NDArray[np.floating[Any]],
        factor: float = 0.5,
    ) -> NDArray[np.floating[Any]]:
        """Reduce star brightness/size by given factor (0=remove, 1=keep)."""
        factor = float(np.clip(factor, 0.0, 1.0))
        starless, stars_only = self.separate(frame)
        reduced_stars = stars_only * factor
        return np.clip(starless + reduced_stars, 0.0, None).astype(frame.dtype)

    def _inpaint_stars(
        self,
        frame: NDArray[np.floating[Any]],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating[Any]]:
        """Replace star pixels with interpolated background."""
        if frame.ndim == 2:
            return self._inpaint_channel(frame, mask)

        result = np.zeros_like(frame)
        for c in range(frame.shape[2]):
            result[..., c] = self._inpaint_channel(frame[..., c], mask)
        return result

    @staticmethod
    def _inpaint_channel(
        channel: NDArray[np.floating[Any]],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating[Any]]:
        """Inpaint masked regions using iterative Gaussian interpolation."""
        result = channel.copy()
        if not mask.any():
            return result

        bg_values = result[~mask]
        if bg_values.size == 0:
            return result
        bg_median = float(np.median(bg_values))

        result[mask] = bg_median

        for sigma in [8.0, 4.0, 2.0]:
            smoothed = ndimage.gaussian_filter(result, sigma=sigma)
            result[mask] = smoothed[mask]

        return result

    @staticmethod
    def _to_grayscale(frame: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        if frame.ndim == 2:
            return frame.astype(np.float64)
        return cast(NDArray[np.floating[Any]], (
            0.2989 * frame[..., 0]
            + 0.5870 * frame[..., 1]
            + 0.1140 * frame[..., 2]
        ).astype(np.float64))

    @staticmethod
    def _expand_mask(
        mask: NDArray[np.bool_], ndim: int
    ) -> NDArray[np.bool_]:
        if ndim == 3:
            return mask[..., np.newaxis]
        return mask
