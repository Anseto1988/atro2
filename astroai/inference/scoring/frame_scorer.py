"""Neural frame scoring for astrophotography."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["FrameScorer"]


class FrameScorer:
    """Scores astrophotography frames on a 0..1 quality scale."""

    def __init__(
        self,
        star_threshold_sigma: float = 5.0,
        min_star_area: int = 3,
        cloud_block_size: int = 64,
    ) -> None:
        self._star_sigma = star_threshold_sigma
        self._min_star_area = min_star_area
        self._cloud_block = cloud_block_size

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.float64)
        return (
            0.2989 * frame[..., 0]
            + 0.5870 * frame[..., 1]
            + 0.1140 * frame[..., 2]
        ).astype(np.float64)

    def _detect_stars(
        self, gray: np.ndarray
    ) -> list[tuple[int, int, float]]:
        mean = gray.mean()
        std = gray.std()
        if std < 1e-8:
            return []
        mask = gray > (mean + self._star_sigma * std)
        labeled, n_labels = ndimage.label(mask)
        stars: list[tuple[int, int, float]] = []
        for i in range(1, n_labels + 1):
            region = labeled == i
            area = region.sum()
            if area < self._min_star_area:
                continue
            flux = float(gray[region].sum())
            ys, xs = np.where(region)
            cy = int(round(np.average(ys, weights=gray[ys, xs])))
            cx = int(round(np.average(xs, weights=gray[ys, xs])))
            stars.append((cy, cx, flux))
        return stars

    def _score_hfr(self, frame: np.ndarray) -> float:
        gray = self._to_grayscale(frame)
        stars = self._detect_stars(gray)
        if not stars:
            return 0.0
        hfrs: list[float] = []
        for cy, cx, flux in stars:
            r_max = 15
            y0 = max(cy - r_max, 0)
            y1 = min(cy + r_max + 1, gray.shape[0])
            x0 = max(cx - r_max, 0)
            x1 = min(cx + r_max + 1, gray.shape[1])
            patch = gray[y0:y1, x0:x1]
            yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
            local_cy = cy - y0
            local_cx = cx - x0
            radii = np.sqrt(
                (yy - local_cy) ** 2 + (xx - local_cx) ** 2
            )
            total_flux = patch.sum()
            if total_flux < 1e-8:
                continue
            half_flux = total_flux / 2.0
            max_r = radii.max()
            if max_r < 1.0:
                continue
            sorted_idx = np.argsort(radii.ravel())
            sorted_flux = patch.ravel()[sorted_idx]
            cum_flux = np.cumsum(sorted_flux)
            idx = np.searchsorted(cum_flux, half_flux)
            sorted_radii = radii.ravel()[sorted_idx]
            hfr = float(sorted_radii[min(idx, len(sorted_radii) - 1)])
            hfrs.append(hfr)
        if not hfrs:
            return 0.0
        median_hfr = float(np.median(hfrs))
        return float(np.clip(1.0 - median_hfr / 10.0, 0.0, 1.0))

    def _score_roundness(self, frame: np.ndarray) -> float:
        gray = self._to_grayscale(frame)
        stars = self._detect_stars(gray)
        if not stars:
            return 0.0
        roundness_vals: list[float] = []
        labeled, _ = ndimage.label(
            gray > (gray.mean() + self._star_sigma * gray.std())
        )
        for cy, cx, _ in stars:
            label_val = labeled[cy, cx]
            if label_val == 0:
                continue
            region = labeled == label_val
            ys, xs = np.where(region)
            if len(ys) < self._min_star_area:
                continue
            m00 = float(len(ys))
            mc_y = ys.mean()
            mc_x = xs.mean()
            dy = ys - mc_y
            dx = xs - mc_x
            mu20 = (dx ** 2).sum() / m00
            mu02 = (dy ** 2).sum() / m00
            mu11 = (dx * dy).sum() / m00
            trace = mu20 + mu02
            if trace < 1e-12:
                continue
            det = mu20 * mu02 - mu11 ** 2
            disc = trace ** 2 - 4.0 * det
            disc = max(disc, 0.0)
            l1 = (trace + np.sqrt(disc)) / 2.0
            l2 = (trace - np.sqrt(disc)) / 2.0
            if l1 < 1e-12:  # pragma: no cover
                continue
            ecc = np.sqrt(1.0 - l2 / l1)
            roundness_vals.append(1.0 - ecc)
        if not roundness_vals:
            return 0.0
        return float(np.clip(np.median(roundness_vals), 0.0, 1.0))

    def _score_cloud_coverage(self, frame: np.ndarray) -> float:
        gray = self._to_grayscale(frame)
        h, w = gray.shape
        bs = self._cloud_block
        if h < bs or w < bs:
            std_val = gray.std()
            med_val = np.median(gray)
            if med_val > gray.mean() * 1.2 and std_val < 20.0:
                return 0.0
            return 1.0
        n_y = h // bs
        n_x = w // bs
        cloud_blocks = 0
        total_blocks = 0
        global_median = np.median(gray)
        for iy in range(n_y):
            for ix in range(n_x):
                block = gray[
                    iy * bs:(iy + 1) * bs,
                    ix * bs:(ix + 1) * bs,
                ]
                local_std = block.std()
                local_med = np.median(block)
                is_bright = local_med > global_median * 1.3
                is_uniform = local_std < 15.0
                if is_bright and is_uniform:
                    cloud_blocks += 1
                total_blocks += 1
        if total_blocks == 0:  # pragma: no cover
            return 1.0
        cloud_frac = cloud_blocks / total_blocks
        return float(np.clip(1.0 - cloud_frac, 0.0, 1.0))

    def score(self, frame: np.ndarray) -> float:
        hfr = self._score_hfr(frame)
        roundness = self._score_roundness(frame)
        cloud = self._score_cloud_coverage(frame)
        return float(
            np.clip(0.4 * hfr + 0.35 * roundness + 0.25 * cloud, 0.0, 1.0)
        )

    def score_batch(self, frames: list[np.ndarray]) -> list[float]:
        return [self.score(f) for f in frames]
