"""Real-time histogram widget for the current image."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["HistogramWidget"]

_NUM_BINS = 256
_BAR_COLOR = QColor(200, 168, 96, 180)
_LINE_COLOR = QColor(200, 168, 96, 255)
_MARGIN = 4


class HistogramWidget(QWidget):
    """Displays a pixel-value histogram from a float32 image array."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMinimumWidth(120)
        self.setAccessibleName("Histogramm")
        self.setToolTip("Pixelwert-Verteilung (logarithmisch)")
        self._bins: np.ndarray | None = None
        self._max_count: float = 1.0

    @Slot(object)
    def set_image_data(self, data: NDArray[np.float32]) -> None:
        flat = data.ravel()
        lo, hi = float(np.min(flat)), float(np.max(flat))
        if hi <= lo:
            self._bins = None
            self.update()
            return
        counts, _ = np.histogram(flat, bins=_NUM_BINS, range=(lo, hi))
        self._bins = counts.astype(np.float64)
        self._max_count = max(float(np.max(self._bins)), 1.0)
        self.update()

    def clear(self) -> None:
        self._bins = None
        self.update()

    def paintEvent(self, _event: object) -> None:
        if self._bins is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width() - 2 * _MARGIN
        h = self.height() - 2 * _MARGIN
        if w <= 0 or h <= 0:
            painter.end()
            return

        log_bins = np.log1p(self._bins)
        log_max = float(np.max(log_bins)) or 1.0

        bar_w = w / _NUM_BINS
        path = QPainterPath()
        path.moveTo(_MARGIN, _MARGIN + h)

        for i, val in enumerate(log_bins):
            bar_h = (val / log_max) * h
            x = _MARGIN + i * bar_w
            y = _MARGIN + h - bar_h
            path.lineTo(x, y)

        path.lineTo(_MARGIN + w, _MARGIN + h)
        path.closeSubpath()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(200, 168, 96, 60))
        painter.drawPath(path)

        painter.setPen(QPen(_LINE_COLOR, 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.end()
