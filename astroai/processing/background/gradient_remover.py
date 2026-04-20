"""Gradient removal by subtracting a modeled background."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroai.processing.background.extractor import BackgroundExtractor, ModelMethod

__all__ = ["GradientRemover"]


class GradientRemover:
    """Remove background gradients from astrophotography frames.

    Extracts a smooth background model and subtracts it, optionally
    normalizing the result to preserve the original median level.
    """

    def __init__(
        self,
        extractor: BackgroundExtractor | None = None,
        preserve_median: bool = True,
        clip_negative: bool = True,
    ) -> None:
        self._extractor = extractor or BackgroundExtractor()
        self._preserve_median = preserve_median
        self._clip_negative = clip_negative

    @property
    def extractor(self) -> BackgroundExtractor:
        return self._extractor

    def remove(
        self, frame: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Remove gradient from frame. Returns corrected frame, same dtype."""
        original_dtype = frame.dtype
        img = frame.astype(np.float64)

        background = self._extractor.extract(img)
        corrected = img - background

        if self._preserve_median:
            original_median = float(np.median(img))
            corrected += original_median

        if self._clip_negative:
            corrected = np.maximum(corrected, 0.0)

        return corrected.astype(original_dtype)

    def remove_batch(
        self, frames: list[NDArray[np.floating[Any]]]
    ) -> list[NDArray[np.floating[Any]]]:
        return [self.remove(f) for f in frames]

    def extract_background(
        self, frame: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Return only the background model for preview/inspection."""
        return self._extractor.extract(frame.astype(np.float64)).astype(frame.dtype)
