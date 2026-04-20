"""AI-based frame alignment for astrophotography image registration."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as ndimage_shift

__all__ = ["FrameAligner"]


class FrameAligner:
    """Phase-correlation based frame aligner for low-SNR astro images."""

    def __init__(self, upsample_factor: int = 10) -> None:
        self.upsample_factor = upsample_factor

    def align(
        self,
        reference: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_gray = self._to_grayscale(reference)
        tgt_gray = self._to_grayscale(target)
        dy, dx = self._phase_correlate(ref_gray, tgt_gray)
        aligned = self._apply_shift(target, dy, dx)
        transform = np.array(
            [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return aligned, transform

    def align_batch(
        self,
        reference: np.ndarray,
        targets: list[np.ndarray],
    ) -> list[np.ndarray]:
        return [self.align(reference, t)[0] for t in targets]

    def _phase_correlate(
        self,
        ref: np.ndarray,
        target: np.ndarray,
    ) -> tuple[float, float]:
        f_ref = np.fft.fft2(ref)
        f_tgt = np.fft.fft2(target)
        cross_power = f_ref * np.conj(f_tgt)
        eps = np.finfo(cross_power.dtype).eps
        cross_power /= np.abs(cross_power) + eps
        correlation = np.fft.ifft2(cross_power).real

        if self.upsample_factor > 1:
            h, w = correlation.shape
            peak = np.unravel_index(
                np.argmax(correlation), correlation.shape
            )
            shift_y = float(peak[0])
            shift_x = float(peak[1])
            if shift_y > h // 2:
                shift_y -= h
            if shift_x > w // 2:
                shift_x -= w
            shift_y, shift_x = self._refine_shift(
                f_ref, f_tgt, shift_y, shift_x
            )
        else:
            peak = np.unravel_index(
                np.argmax(correlation), correlation.shape
            )
            h, w = correlation.shape
            shift_y = float(peak[0])
            shift_x = float(peak[1])
            if shift_y > h // 2:
                shift_y -= h
            if shift_x > w // 2:
                shift_x -= w

        return shift_y, shift_x

    def _refine_shift(
        self,
        f_ref: np.ndarray,
        f_tgt: np.ndarray,
        coarse_y: float,
        coarse_x: float,
    ) -> tuple[float, float]:
        u = self.upsample_factor
        h, w = f_ref.shape
        region = int(np.ceil(1.5 * u))
        dft_size = region * 2 + 1

        row_shifts = (
            np.arange(dft_size) - region
        ) / u + coarse_y
        col_shifts = (
            np.arange(dft_size) - region
        ) / u + coarse_x

        row_kern = np.exp(
            -2j * np.pi * np.fft.fftfreq(h)[:, None] * row_shifts[None, :]
        )
        col_kern = np.exp(
            -2j * np.pi * np.fft.fftfreq(w)[:, None] * col_shifts[None, :]
        )

        cross_power = f_ref * np.conj(f_tgt)
        eps = np.finfo(cross_power.dtype).eps
        cross_power /= np.abs(cross_power) + eps

        upsampled = (row_kern.conj().T @ cross_power @ col_kern).real
        peak = np.unravel_index(
            np.argmax(upsampled), upsampled.shape
        )
        refined_y = row_shifts[peak[0]]
        refined_x = col_shifts[peak[1]]
        return float(refined_y), float(refined_x)

    def _apply_shift(
        self,
        image: np.ndarray,
        dy: float,
        dx: float,
    ) -> np.ndarray:
        if image.ndim == 3:
            shift_vec = (-dy, -dx, 0)
        else:
            shift_vec = (-dy, -dx)
        return ndimage_shift(
            image, shift_vec, order=3, mode="constant", cval=0.0
        )

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.float64)
        weights = np.array([0.2989, 0.5870, 0.1140])
        return np.dot(frame[..., :3], weights).astype(np.float64)
