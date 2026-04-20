"""Intelligent histogram stretching for astrophotography."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["IntelligentStretcher"]


class IntelligentStretcher:
    """Automatic MTF/STF-based histogram stretching.

    Enhances faint details in linear astro images without amplifying noise.
    Zero-parameter mode auto-computes optimal midtone balance from image stats.
    """

    def __init__(
        self,
        target_background: float = 0.25,
        shadow_clipping_sigmas: float = -2.8,
        linked_channels: bool = True,
    ) -> None:
        self._target_bg = target_background
        self._shadow_clip_sigmas = shadow_clipping_sigmas
        self._linked = linked_channels

    def stretch(
        self, frame: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Auto-stretch a linear frame to a visually useful nonlinear range."""
        original_dtype = frame.dtype
        img = frame.astype(np.float64)

        if img.ndim == 2:
            return self._stretch_channel(img).astype(original_dtype)

        if self._linked:
            return self._stretch_linked(img).astype(original_dtype)
        return self._stretch_independent(img).astype(original_dtype)

    def stretch_batch(
        self, frames: list[NDArray[np.floating[Any]]]
    ) -> list[NDArray[np.floating[Any]]]:
        return [self.stretch(f) for f in frames]

    def _stretch_linked(
        self, img: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Stretch RGB with a single set of parameters from combined stats."""
        normalized = self._normalize(img)
        combined = np.mean(normalized, axis=2)
        median_val, mad_val = self._background_stats(combined)
        shadow_clip = self._compute_shadow_clip(median_val, mad_val)
        midtone = self._compute_midtone(median_val, shadow_clip)

        result = np.zeros_like(normalized)
        for c in range(normalized.shape[2]):
            ch = normalized[..., c]
            clipped = np.clip((ch - shadow_clip) / max(1.0 - shadow_clip, 1e-10), 0.0, 1.0)
            result[..., c] = self._apply_mtf(clipped, midtone)
        return result

    def _stretch_independent(
        self, img: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Stretch each RGB channel independently."""
        normalized = self._normalize(img)
        result = np.zeros_like(normalized)
        for c in range(normalized.shape[2]):
            result[..., c] = self._stretch_channel(normalized[..., c])
        return result

    def _stretch_channel(
        self, channel: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Stretch a single channel using auto-computed STF parameters."""
        normalized = self._normalize_1d(channel)
        median_val, mad_val = self._background_stats(normalized)
        shadow_clip = self._compute_shadow_clip(median_val, mad_val)
        midtone = self._compute_midtone(median_val, shadow_clip)
        clipped = np.clip(
            (normalized - shadow_clip) / max(1.0 - shadow_clip, 1e-10), 0.0, 1.0
        )
        return self._apply_mtf(clipped, midtone)

    @staticmethod
    def _normalize(img: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        vmin = img.min()
        vmax = img.max()
        rng = vmax - vmin
        if rng < 1e-10:
            return np.zeros_like(img)
        return (img - vmin) / rng

    @staticmethod
    def _normalize_1d(ch: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        vmin = ch.min()
        vmax = ch.max()
        rng = vmax - vmin
        if rng < 1e-10:
            return np.zeros_like(ch)
        return (ch - vmin) / rng

    @staticmethod
    def _background_stats(
        channel: NDArray[np.floating[Any]],
    ) -> tuple[float, float]:
        """Compute median and MAD-based noise estimate."""
        median_val = float(np.median(channel))
        mad = float(np.median(np.abs(channel - median_val)))
        noise = mad * 1.4826
        return median_val, noise

    def _compute_shadow_clip(self, median_val: float, mad_val: float) -> float:
        clip = median_val + self._shadow_clip_sigmas * mad_val
        return max(clip, 0.0)

    def _compute_midtone(self, median_val: float, shadow_clip: float) -> float:
        """Compute midtone balance to map background to target_background."""
        bg_normalized = (median_val - shadow_clip) / max(1.0 - shadow_clip, 1e-10)
        bg_normalized = np.clip(bg_normalized, 1e-6, 1.0 - 1e-6)

        target = self._target_bg
        if bg_normalized >= target:
            return 0.5

        midtone = self._mtf_balance(bg_normalized, target)
        return np.clip(midtone, 0.01, 0.99)

    @staticmethod
    def _mtf_balance(bg: float, target: float) -> float:
        """Solve for m such that MTF(bg, m) = target."""
        if bg < 1e-10 or bg > 1.0 - 1e-10:
            return 0.5
        return (target * (bg - 1.0)) / ((2.0 * target - 1.0) * bg - target)

    @staticmethod
    def _apply_mtf(
        data: NDArray[np.floating[Any]], midtone: float
    ) -> NDArray[np.floating[Any]]:
        """Apply midtone transfer function: MTF(x,m) = ((m-1)*x)/((2m-1)*x - m)."""
        m = midtone
        numerator = (m - 1.0) * data
        denominator = (2.0 * m - 1.0) * data - m
        safe_denom = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        result = numerator / safe_denom
        return np.clip(result, 0.0, 1.0)
