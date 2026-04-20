"""GPU-accelerated image viewer with zoom/pan for large FITS files."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ImageViewer"]

_ZOOM_FACTOR = 1.15
_MIN_ZOOM = 0.05
_MAX_ZOOM = 50.0
_TILE_SIZE = 2048
_PAN_STEP = 40


class ImageViewer(QWidget):
    """Displays astronomical images with lazy tile loading, zoom and pan."""

    zoom_changed = Signal(float)
    view_changed = Signal()
    pixel_hovered = Signal(int, int, float)  # x, y, value

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 200)
        self.setAccessibleName("Bild-Viewer")
        self.setToolTip("Zoom: +/- oder Mausrad | Pan: Pfeiltasten oder Drag | Home: Einpassen")

        self._raw_data: NDArray[np.float32] | None = None
        self._height = 0
        self._width = 0
        self._tile_cache: dict[tuple[int, int], QImage] = {}

        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._dragging = False
        self._drag_start = QPointF()

    def set_image_data(self, data: NDArray[np.float32]) -> None:
        if data.ndim == 3:
            data = np.mean(data, axis=0).astype(np.float32)
        self._raw_data = data
        self._height, self._width = data.shape
        self._tile_cache.clear()
        self._zoom = min(
            self.width() / max(self._width, 1),
            self.height() / max(self._height, 1),
        )
        self._offset = QPointF(0, 0)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def clear(self) -> None:
        self._raw_data = None
        self._tile_cache.clear()
        self.update()

    @property
    def zoom_level(self) -> float:
        return self._zoom

    @Slot(float)
    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, zoom))
        self._tile_cache.clear()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def _tile_key(self, tx: int, ty: int) -> tuple[int, int]:
        return (tx, ty)

    def _render_tile(self, tx: int, ty: int) -> QImage:
        key = self._tile_key(tx, ty)
        cached = self._tile_cache.get(key)
        if cached is not None:
            return cached

        assert self._raw_data is not None
        y0 = ty * _TILE_SIZE
        x0 = tx * _TILE_SIZE
        y1 = min(y0 + _TILE_SIZE, self._height)
        x1 = min(x0 + _TILE_SIZE, self._width)

        tile = self._raw_data[y0:y1, x0:x1]
        lo, hi = float(np.min(tile)), float(np.max(tile))
        rng = hi - lo if hi > lo else 1.0
        normalized = ((tile - lo) / rng * 255).clip(0, 255).astype(np.uint8)

        h, w = normalized.shape
        bytes_per_line = w
        qimg = QImage(normalized.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8).copy()
        self._tile_cache[key] = qimg
        return qimg

    def paintEvent(self, _event: object) -> None:
        if self._raw_data is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 1.0)

        vp_w = self.width()
        vp_h = self.height()

        img_x0 = -self._offset.x() / self._zoom
        img_y0 = -self._offset.y() / self._zoom
        img_x1 = img_x0 + vp_w / self._zoom
        img_y1 = img_y0 + vp_h / self._zoom

        tx0 = max(0, int(img_x0) // _TILE_SIZE)
        ty0 = max(0, int(img_y0) // _TILE_SIZE)
        tx1 = min((self._width - 1) // _TILE_SIZE, int(img_x1) // _TILE_SIZE)
        ty1 = min((self._height - 1) // _TILE_SIZE, int(img_y1) // _TILE_SIZE)

        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile_img = self._render_tile(tx, ty)
                sx = self._offset.x() + tx * _TILE_SIZE * self._zoom
                sy = self._offset.y() + ty * _TILE_SIZE * self._zoom
                sw = tile_img.width() * self._zoom
                sh = tile_img.height() * self._zoom
                painter.drawImage(QRectF(sx, sy, sw, sh), tile_img)

        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:
        pos = event.position()
        old_zoom = self._zoom
        if event.angleDelta().y() > 0:
            self._zoom = min(_MAX_ZOOM, self._zoom * _ZOOM_FACTOR)
        else:
            self._zoom = max(_MIN_ZOOM, self._zoom / _ZOOM_FACTOR)

        ratio = self._zoom / old_zoom
        self._offset = QPointF(
            pos.x() - ratio * (pos.x() - self._offset.x()),
            pos.y() - ratio * (pos.y() - self._offset.y()),
        )
        self._tile_cache.clear()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._dragging:
            delta = pos - self._drag_start
            self._offset += delta
            self._drag_start = pos
            self.view_changed.emit()
            self.update()
        elif self._raw_data is not None:
            ix = int((pos.x() - self._offset.x()) / self._zoom)
            iy = int((pos.y() - self._offset.y()) / self._zoom)
            if 0 <= ix < self._width and 0 <= iy < self._height:
                self.pixel_hovered.emit(ix, iy, float(self._raw_data[iy, ix]))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.set_zoom(self._zoom * _ZOOM_FACTOR)
        elif key == Qt.Key.Key_Minus:
            self.set_zoom(self._zoom / _ZOOM_FACTOR)
        elif key == Qt.Key.Key_Home:
            self.fit_to_view()
        elif key == Qt.Key.Key_Left:
            self._offset += QPointF(_PAN_STEP, 0)
            self.view_changed.emit()
            self.update()
        elif key == Qt.Key.Key_Right:
            self._offset += QPointF(-_PAN_STEP, 0)
            self.view_changed.emit()
            self.update()
        elif key == Qt.Key.Key_Up:
            self._offset += QPointF(0, _PAN_STEP)
            self.view_changed.emit()
            self.update()
        elif key == Qt.Key.Key_Down:
            self._offset += QPointF(0, -_PAN_STEP)
            self.view_changed.emit()
            self.update()
        else:
            super().keyPressEvent(event)

    @Slot()
    def fit_to_view(self) -> None:
        if self._raw_data is None:
            return
        self._zoom = min(
            self.width() / max(self._width, 1),
            self.height() / max(self._height, 1),
        )
        self._offset = QPointF(0, 0)
        self._tile_cache.clear()
        self.zoom_changed.emit(self._zoom)
        self.update()
