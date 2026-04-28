"""AI-gestützte Kometen-Kerndetektion mittels Differenz-Imaging."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.ndimage import label, center_of_mass

__all__ = ["CometTracker", "CometPosition"]

logger = logging.getLogger(__name__)


class CometPosition:
    __slots__ = ("y", "x", "confidence")

    def __init__(self, y: float, x: float, confidence: float = 1.0) -> None:
        self.y = y
        self.x = x
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"CometPosition(y={self.y:.2f}, x={self.x:.2f}, conf={self.confidence:.3f})"


class CometTracker:
    """Detektiert den Kometen-Kern in jedem Frame mittels Differenz-Imaging.

    Algorithmus:
      1. Erstelle Referenzbild (Median aller Frames) → enthält nur Sterne/Hintergrund.
      2. Subtrahiere Referenz von jedem Frame → hebt bewegte Objekte hervor.
      3. Finde hellstes zusammenhängendes Blob als Kometenkopf.
      4. Berechne Schwerpunkt des Blobs als Subpixel-Position.

    Args:
        min_blob_area: Mindestgröße eines Blobs in Pixeln.
        top_fraction: Anteil der hellsten Pixel, die als Blob-Maske dienen.
        fallback_to_peak: Bei keinem Blob → verwende den hellsten Pixel.
    """

    def __init__(
        self,
        min_blob_area: int = 5,
        top_fraction: float = 0.002,
        fallback_to_peak: bool = True,
    ) -> None:
        self._min_blob_area = min_blob_area
        self._top_fraction = top_fraction
        self._fallback_to_peak = fallback_to_peak

    def track(self, frames: Sequence[np.ndarray]) -> list[CometPosition]:
        """Bestimme Kometenkopf-Position für jeden Frame.

        Args:
            frames: Liste von (H, W) oder (H, W, C) Arrays.

        Returns:
            Liste von CometPosition-Objekten, eine pro Frame.
        """
        if not frames:
            raise ValueError("Keine Frames übergeben")

        gray_frames = [self._to_grayscale(f) for f in frames]
        reference = np.median(np.stack(gray_frames, axis=0), axis=0)

        positions: list[CometPosition] = []
        for i, gray in enumerate(gray_frames):
            diff = gray.astype(np.float64) - reference.astype(np.float64)
            diff = np.clip(diff, 0.0, None)
            pos = self._find_nucleus(diff, frame_idx=i)
            positions.append(pos)

        return positions

    def _find_nucleus(self, diff: np.ndarray, frame_idx: int) -> CometPosition:
        n_pixels = diff.size
        n_top = max(1, int(n_pixels * self._top_fraction))
        threshold = np.partition(diff.ravel(), -n_top)[-n_top]

        if threshold <= 0.0:
            if self._fallback_to_peak:
                peak = np.unravel_index(np.argmax(diff), diff.shape)
                logger.debug("Frame %d: no bright blob, fallback to peak %s", frame_idx, peak)
                return CometPosition(float(peak[0]), float(peak[1]), confidence=0.3)
            raise RuntimeError(f"Frame {frame_idx}: kein Blob gefunden und kein Fallback")

        mask = diff >= threshold
        labeled, n_labels = label(mask)

        best_label = -1
        best_sum = -1.0
        for lbl in range(1, n_labels + 1):
            region = labeled == lbl
            area = int(region.sum())
            if area < self._min_blob_area:
                continue
            s = float(diff[region].sum())
            if s > best_sum:
                best_sum = s
                best_label = lbl

        if best_label < 0:
            if self._fallback_to_peak:
                peak = np.unravel_index(np.argmax(diff), diff.shape)
                logger.debug("Frame %d: blob too small, fallback to peak %s", frame_idx, peak)
                return CometPosition(float(peak[0]), float(peak[1]), confidence=0.4)
            raise RuntimeError(f"Frame {frame_idx}: kein gültiger Blob")

        region_mask = labeled == best_label
        cy, cx = center_of_mass(diff, labels=labeled, index=best_label)
        confidence = min(1.0, best_sum / (np.max(diff) * self._min_blob_area + 1e-12))

        logger.debug(
            "Frame %d: comet nucleus at (%.2f, %.2f), conf=%.3f",
            frame_idx, cy, cx, confidence,
        )
        return CometPosition(float(cy), float(cx), confidence=float(confidence))

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.float64)
        weights = np.array([0.2989, 0.5870, 0.1140])
        return np.dot(frame[..., :3], weights).astype(np.float64)
