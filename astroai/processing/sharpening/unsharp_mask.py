"""Unsharp masking for astrophotography detail enhancement.

Classic unsharp mask: subtracts a Gaussian-blurred copy from the original,
then adds the scaled difference back. A threshold prevents amplifying noise
in flat regions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

__all__ = ["UnsharpMask"]


class UnsharpMask:
    """Unsharp mask sharpening with optional noise threshold.

    Parameters
    ----------
    radius:
        Gaussian blur radius in pixels (sigma). Larger values sharpen coarser
        structures; values 0.5–3.0 are typical for astrophotography.
    amount:
        Sharpening strength as a fraction [0, 1]. 0 = no effect; 1 = full
        difference added back.
    threshold:
        Minimum edge magnitude [0, 1] below which sharpening is suppressed.
        Prevents amplifying noise in flat sky backgrounds.
    """

    def __init__(
        self,
        radius: float = 1.0,
        amount: float = 0.5,
        threshold: float = 0.02,
    ) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if not 0.0 <= amount <= 1.0:
            raise ValueError(f"amount must be in [0, 1], got {amount}")
        if not 0.0 <= threshold <= 0.5:
            raise ValueError(f"threshold must be in [0, 0.5], got {threshold}")
        self._radius = radius
        self._amount = amount
        self._threshold = threshold

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def amount(self) -> float:
        return self._amount

    @property
    def threshold(self) -> float:
        return self._threshold

    def apply(
        self,
        image: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Apply unsharp masking to *image* and return the sharpened result.

        Works on both 2-D (grayscale) and 3-D (H×W×C) floating-point images.
        The output is clipped to [0, 1] and preserves the input dtype.
        """
        src = image.astype(np.float64, copy=False)

        if src.ndim == 2:
            blurred = gaussian_filter(src, sigma=self._radius)
        else:
            blurred = np.empty_like(src)
            for ch in range(src.shape[2]):
                blurred[:, :, ch] = gaussian_filter(src[:, :, ch], sigma=self._radius)

        diff = src - blurred

        if self._threshold > 0.0:
            mask = np.abs(diff) >= self._threshold
            sharpened = np.where(mask, src + self._amount * diff, src)
        else:
            sharpened = src + self._amount * diff

        return np.clip(sharpened, 0.0, 1.0).astype(image.dtype)
