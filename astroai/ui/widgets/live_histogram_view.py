"""Live multi-channel histogram widget (RGB + luminance, background thread)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QCheckBox, QSizePolicy, QVBoxLayout, QWidget

__all__ = ["HistogramView", "HistogramWorker"]

_NUM_BINS = 256
_MARGIN = 6
_LINE_ALPHA = 200

_COLOR_R = QColor("#e05c5c")
_COLOR_G = QColor("#5ce05c")
_COLOR_B = QColor("#5c8fe0")
_COLOR_LUM = QColor("#cccccc")


class _HistogramData:
    __slots__ = ("r", "g", "b", "lum")

    def __init__(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
        lum: np.ndarray,
    ) -> None:
        self.r = r
        self.g = g
        self.b = b
        self.lum = lum


class _ComputeWorker(QObject):
    """Runs in a QThread; computes per-channel histograms."""

    result_ready = Signal(object)  # _HistogramData

    @Slot(object)
    def compute(self, data: object) -> None:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            ch_r = arr[:, :, 0].ravel()
            ch_g = arr[:, :, 1].ravel()
            ch_b = arr[:, :, 2].ravel()
            ch_lum = (
                0.2126 * arr[:, :, 0]
                + 0.7152 * arr[:, :, 1]
                + 0.0722 * arr[:, :, 2]
            ).ravel()
        else:
            flat = arr.ravel()
            ch_r = ch_g = ch_b = ch_lum = flat

        def _hist(ch: np.ndarray) -> np.ndarray:
            lo, hi = float(np.min(ch)), float(np.max(ch))
            if hi <= lo:
                return np.zeros(_NUM_BINS, dtype=np.float64)
            counts, _ = np.histogram(ch, bins=_NUM_BINS, range=(lo, hi))
            return counts.astype(np.float64)

        self.result_ready.emit(
            _HistogramData(
                r=_hist(ch_r),
                g=_hist(ch_g),
                b=_hist(ch_b),
                lum=_hist(ch_lum),
            )
        )


class HistogramWorker(QObject):
    """Thread-safe histogram worker — compute() safe to call from any thread."""

    result_ready = Signal(object)  # _HistogramData
    _trigger = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._inner = _ComputeWorker()
        self._inner.moveToThread(self._thread)
        self._trigger.connect(self._inner.compute)
        self._inner.result_ready.connect(self.result_ready)
        self._thread.finished.connect(self._inner.deleteLater)
        self._thread.start()

    def compute(self, data: object) -> None:
        self._trigger.emit(data)

    def stop(self) -> None:
        self._thread.quit()
        self._thread.wait()


class _HistogramCanvas(QWidget):
    """QPainter-based canvas drawing overlaid per-channel curves."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._data: _HistogramData | None = None
        self._log_scale: bool = True

    def set_data(self, data: _HistogramData) -> None:
        self._data = data
        self.update()

    @Slot(bool)
    def set_log_scale(self, enabled: bool) -> None:
        self._log_scale = enabled
        self.update()

    def clear(self) -> None:
        self._data = None
        self.update()

    def _draw_channel(
        self,
        painter: QPainter,
        counts: np.ndarray,
        color: QColor,
        w: int,
        h: int,
    ) -> None:
        vals = np.log1p(counts) if self._log_scale else counts.astype(np.float64)
        max_val = float(np.max(vals)) or 1.0
        c = QColor(color)
        c.setAlpha(_LINE_ALPHA)
        painter.setPen(QPen(c, 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        bin_w = w / _NUM_BINS
        first = True
        for i, val in enumerate(vals):
            x = _MARGIN + i * bin_w
            y = _MARGIN + h - (val / max_val) * h
            if first:
                path.moveTo(x, y)
                first = False
            else:
                path.lineTo(x, y)
        painter.drawPath(path)

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width() - 2 * _MARGIN
        h = self.height() - 2 * _MARGIN
        if w <= 0 or h <= 0 or self._data is None:
            painter.end()
            return
        for counts, color in (
            (self._data.lum, _COLOR_LUM),
            (self._data.r, _COLOR_R),
            (self._data.g, _COLOR_G),
            (self._data.b, _COLOR_B),
        ):
            self._draw_channel(painter, counts, color, w, h)
        painter.end()


class HistogramView(QWidget):
    """Live histogram: RGB + luminance, log-scale toggle, background computation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.resize(280, 180)
        self.setAccessibleName("Live-Histogramm")
        self.setToolTip("RGB-Kanaele und Luminanz (Echtzeit)")

        self._worker = HistogramWorker(self)
        self._canvas = _HistogramCanvas(self)

        self._log_cb = QCheckBox("Log-Skala", self)
        self._log_cb.setChecked(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(2)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._log_cb)

        self._log_cb.toggled.connect(self._canvas.set_log_scale)
        self._worker.result_ready.connect(self._on_result_ready)

    @Slot(object)
    def set_image_data(self, data: object) -> None:
        if isinstance(data, np.ndarray):
            self._worker.compute(data)

    @Slot(object)
    def _on_result_ready(self, data: object) -> None:
        if isinstance(data, _HistogramData):
            self._canvas.set_data(data)

    def clear(self) -> None:
        self._canvas.clear()
