"""Synthetic flat frame generation from science light frames.

Estimates the illumination gradient (vignetting) of an optical system
by modeling the large-scale background structure of sky-subtracted light
frames.  The resulting synthetic flat can substitute for a real flat when
calibration frames are unavailable.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from astroai.processing.background.extractor import BackgroundExtractor, ModelMethod

__all__ = ["SyntheticFlatGenerator"]

logger = logging.getLogger(__name__)


class SyntheticFlatGenerator:
    """Generate synthetic flat frames from a stack of light frames.

    Algorithm:
    1. For each light frame, model the large-scale background via tile
       sampling and polynomial/RBF fitting (reuses BackgroundExtractor).
    2. Median-combine all per-frame background models to suppress
       astronomical signal contributions.
    3. Apply Gaussian smoothing to suppress remaining tile-grid artefacts.
    4. Normalize so the peak value equals 1.0 (flat convention).
    """

    def __init__(
        self,
        tile_size: int = 64,
        method: ModelMethod = ModelMethod.RBF,
        poly_degree: int = 4,
        smoothing_sigma: float = 8.0,
        min_frames: int = 1,
    ) -> None:
        self._extractor = BackgroundExtractor(
            tile_size=tile_size,
            method=method,
            poly_degree=poly_degree,
        )
        self._smoothing_sigma = smoothing_sigma
        self._min_frames = min_frames

    @property
    def smoothing_sigma(self) -> float:
        return self._smoothing_sigma

    @property
    def min_frames(self) -> int:
        return self._min_frames

    def generate(
        self,
        frames: Sequence[NDArray[np.floating[Any]]],
    ) -> NDArray[np.float32]:
        """Generate a synthetic flat from one or more light frames.

        Parameters
        ----------
        frames:
            Sequence of 2-D (H, W) or 3-D (H, W, C) float arrays.

        Returns
        -------
        NDArray[np.float32]
            Normalized flat frame in the same spatial shape as the input,
            with values in (0, 1].  Channels are processed independently
            for colour images.
        """
        if len(frames) < self._min_frames:
            raise ValueError(
                f"Need at least {self._min_frames} frame(s), got {len(frames)}"
            )

        models = [self._model_frame(f) for f in frames]
        combined = np.median(np.stack(models, axis=0), axis=0).astype(np.float64)

        combined = self._smooth(combined)
        return self._normalize(combined).astype(np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _model_frame(
        self, frame: NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        model: NDArray[np.floating[Any]] = self._extractor.extract(frame)
        result = np.asarray(model, dtype=np.float64)
        result = np.clip(result, 0.0, None)
        return result

    def _smooth(
        self, model: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._smoothing_sigma <= 0:
            return model
        if model.ndim == 3:
            smoothed = np.stack(
                [gaussian_filter(model[..., c], sigma=self._smoothing_sigma)
                 for c in range(model.shape[2])],
                axis=-1,
            )
            return smoothed.astype(np.float64)
        result: NDArray[np.float64] = gaussian_filter(model, sigma=self._smoothing_sigma).astype(np.float64)
        return result

    @staticmethod
    def _normalize(model: NDArray[np.float64]) -> NDArray[np.float64]:
        peak = float(model.max())
        if peak <= 0.0:
            logger.warning("Synthetic flat model has zero or negative peak — returning uniform flat")
            return np.ones_like(model)
        return model / peak
