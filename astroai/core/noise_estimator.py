"""Sky-background noise estimation for astrophotography images."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["NoiseEstimate", "NoiseEstimator"]


@dataclass(frozen=True)
class NoiseEstimate:
    sky_sigma: float
    snr_db: float
    noise_level_pct: float
    suggested_strength: float

    def __str__(self) -> str:
        return (
            f"σ={self.sky_sigma:.4f} "
            f"SNR={self.snr_db:.1f}dB "
            f"noise={self.noise_level_pct:.1f}% "
            f"→ strength={self.suggested_strength:.2f}"
        )


class NoiseEstimator:
    """Estimates sky-background noise via iterative sigma-clipping (MAD-based).

    Parameters
    ----------
    iterations:
        Number of sigma-clipping iterations.
    kappa:
        Rejection threshold in units of sigma.
    """

    def __init__(self, iterations: int = 5, kappa: float = 3.0) -> None:
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if kappa <= 0:
            raise ValueError("kappa must be > 0")
        self.iterations = iterations
        self.kappa = kappa

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, image: NDArray[np.floating]) -> NoiseEstimate:
        """Return a NoiseEstimate for *image*.

        Accepts (H, W), (H, W, 1), or (H, W, C) float arrays in [0, 1].
        Multi-channel images are converted to luminance before estimation.
        """
        arr = self._to_luminance(np.asarray(image, dtype=np.float64))
        flat = arr.ravel()
        sky_sigma = self._sigma_clip(flat)
        snr_db = self._snr(arr, sky_sigma)
        noise_level_pct = min(sky_sigma / 0.10 * 100.0, 100.0)
        suggested_strength = self._strength_from_sigma(sky_sigma)
        return NoiseEstimate(
            sky_sigma=float(sky_sigma),
            snr_db=float(snr_db),
            noise_level_pct=float(noise_level_pct),
            suggested_strength=float(suggested_strength),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_luminance(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                return arr[:, :, 0]
            # BT.601 luminance weights
            return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    def _sigma_clip(self, flat: NDArray[np.float64]) -> float:
        mask = np.ones(len(flat), dtype=bool)
        for _ in range(self.iterations):
            data = flat[mask]
            if len(data) < 10:
                break
            median = float(np.median(data))
            mad = float(np.median(np.abs(data - median)))
            sigma = mad * 1.4826  # MAD → Gaussian sigma
            if sigma == 0.0:
                break
            mask = mask & (np.abs(flat - median) <= self.kappa * sigma)
        sky_data = flat[mask]
        if len(sky_data) < 2:
            return float(np.std(flat))
        return float(np.std(sky_data))

    def _snr(self, arr: NDArray[np.float64], sky_sigma: float) -> float:
        if sky_sigma <= 0:
            return 0.0
        signal = float(np.mean(arr))
        return 20.0 * np.log10(max(signal, 1e-12) / sky_sigma)

    @staticmethod
    def _strength_from_sigma(sigma: float) -> float:
        """Map sky-sigma to a suggested denoising strength [0, 1]."""
        if sigma <= 0.005:
            return 0.2
        if sigma <= 0.015:
            return 0.2 + (sigma - 0.005) / 0.010 * 0.3  # 0.2 → 0.5
        if sigma <= 0.040:
            return 0.5 + (sigma - 0.015) / 0.025 * 0.3  # 0.5 → 0.8
        return min(0.8 + (sigma - 0.040) / 0.060 * 0.2, 1.0)  # 0.8 → 1.0
