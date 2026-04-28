"""Star detection, separation and reduction for astrophotography."""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

__all__ = ["StarManager"]

AUTO_TILE_THRESHOLD = 4096 * 4096
DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 64

OnTileProgress = Callable[[int, int], None]


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
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
    ) -> None:
        self._sigma = detection_sigma
        self._min_area = min_star_area
        self._max_area = max_star_area
        self._dilation = mask_dilation
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap

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
            if dy == 0 or dx == 0:  # pragma: no cover
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

    # ------------------------------------------------------------------
    # Tile-based processing
    # ------------------------------------------------------------------

    @staticmethod
    def needs_tiling(
        h: int, w: int, threshold: int = AUTO_TILE_THRESHOLD,
    ) -> bool:
        return h * w > threshold

    @staticmethod
    def _cosine_weight_1d(
        length: int, overlap: int, at_start: bool, at_end: bool,
    ) -> NDArray[np.floating[Any]]:
        w = np.ones(length, dtype=np.float64)
        if overlap <= 0 or overlap >= length:
            return w
        ramp = np.float64(0.5) * (1.0 - np.cos(np.linspace(0, np.pi, overlap)))
        if not at_start:
            w[:overlap] = ramp
        if not at_end:
            w[-overlap:] = ramp[::-1]
        return w

    @staticmethod
    def process_tiled(
        image: NDArray[np.floating[Any]],
        process_fn: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_TILE_OVERLAP,
        on_progress: OnTileProgress | None = None,
    ) -> NDArray[np.floating[Any]]:
        h, w = image.shape[:2]
        step = tile_size - overlap

        result = np.zeros_like(image, dtype=np.float64)
        weight_map = np.zeros((h, w), dtype=np.float64)

        y_starts = list(range(0, h, step))
        x_starts = list(range(0, w, step))
        total_tiles = len(y_starts) * len(x_starts)
        tile_idx = 0

        for y in y_starts:
            for x in x_starts:
                y1 = min(y + tile_size, h)
                x1 = min(x + tile_size, w)
                y0 = max(y1 - tile_size, 0)
                x0 = max(x1 - tile_size, 0)

                th, tw = y1 - y0, x1 - x0
                tile = image[y0:y1, x0:x1]

                ph, pw = tile_size - th, tile_size - tw
                if ph > 0 or pw > 0:
                    pad_w = [(0, ph), (0, pw)]
                    if image.ndim == 3:
                        pad_w.append((0, 0))
                    tile = np.pad(tile, pad_w, mode="reflect")

                processed = process_fn(tile)
                processed = processed[:th, :tw]

                wy = StarManager._cosine_weight_1d(
                    th, overlap, at_start=(y0 == 0), at_end=(y1 == h),
                )
                wx = StarManager._cosine_weight_1d(
                    tw, overlap, at_start=(x0 == 0), at_end=(x1 == w),
                )
                weight = wy[:, np.newaxis] * wx[np.newaxis, :]

                if image.ndim == 3:
                    result[y0:y1, x0:x1] += processed.astype(np.float64) * weight[..., np.newaxis]
                else:
                    result[y0:y1, x0:x1] += processed.astype(np.float64) * weight
                weight_map[y0:y1, x0:x1] += weight

                tile_idx += 1
                if on_progress is not None:
                    on_progress(tile_idx, total_tiles)

        weight_map = np.maximum(weight_map, 1e-8)
        if image.ndim == 3:
            result /= weight_map[..., np.newaxis]
        else:
            result /= weight_map

        return cast(NDArray[np.floating[Any]], result.astype(image.dtype))
