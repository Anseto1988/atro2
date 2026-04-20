"""Lucy-Richardson deconvolution algorithm (scipy-based)."""
from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve

__all__ = ["Deconvolver", "gaussian_psf"]

logger = logging.getLogger(__name__)


def gaussian_psf(size: int = 5, sigma: float = 1.0) -> NDArray[np.float64]:
    """Create a normalized 2D Gaussian PSF kernel."""
    half = size // 2
    k = np.arange(-half, half + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * (k / sigma) ** 2)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return cast(NDArray[np.float64], kernel_2d / kernel_2d.sum())


class Deconvolver:
    """Lucy-Richardson iterative image deconvolution."""

    def __init__(
        self,
        iterations: int = 10,
        psf_size: int = 5,
        psf_sigma: float = 1.0,
        clip_output: bool = True,
    ) -> None:
        self._iterations = iterations
        self._psf = gaussian_psf(psf_size, psf_sigma)
        self._clip_output = clip_output

    def deconvolve(self, image: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Apply Lucy-Richardson deconvolution; handles grayscale and RGB."""
        is_rgb = image.ndim == 3 and image.shape[2] == 3
        if is_rgb:
            channels = [self._lr_channel(image[..., c]) for c in range(3)]
            result: NDArray[np.float64] = np.stack(channels, axis=-1)
        else:
            result = self._lr_channel(image)
        if self._clip_output:
            ceil = float(image.max()) if image.max() > 0 else 1.0
            result = np.clip(result, 0.0, ceil)
        return result.astype(image.dtype)

    def _lr_channel(self, channel: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Single-channel Lucy-Richardson iteration loop."""
        psf = self._psf
        psf_mirror = psf[::-1, ::-1]
        estimate = channel.astype(np.float64).copy()
        eps = np.finfo(np.float64).tiny

        for _ in range(self._iterations):
            blurred = convolve(estimate, psf, mode="reflect")
            ratio = channel.astype(np.float64) / (blurred + eps)
            correction = convolve(ratio, psf_mirror, mode="reflect")
            estimate = estimate * correction

        return estimate
