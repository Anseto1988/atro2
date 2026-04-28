"""Draggable split before/after comparison view."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["SplitCompareView"]

_ZOOM_FACTOR = 1.15
_MIN_ZOOM = 0.05
_MAX_ZOOM = 50.0
_TILE_SIZE = 2048
_SPLIT_HIT_PX = 8
_LABEL_PAD = 8
_PAN_STEP = 40


class SplitCompareView(QWidget):
    """Renders before (left) and after (right) images with a draggable vertical split."""

    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 200)
        self.setAccessibleName("Vorher/Nachher-Vergleich")
        self.setToolTip(
            "Teiler ziehen | Mausrad: Zoom | Drag: Pan | Home: Einpassen"
        )

        self._before: NDArray[np.float32] | None = None
        self._after: NDArray[np.float32] | None = None
        self._height = 0
        self._width = 0

        self._before_cache: dict[tuple[int, int], QImage] = {}
        self._after_cache: dict[tuple[int, int], QImage] = {}

        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._split = 0.5

        self._pan_drag = False
        self._drag_start = QPointF()
        self._split_drag = False

    def set_before(self, data: NDArray[np.float32]) -> None:
        arr = np.mean(data, axis=0).astype(np.float32) if data.ndim == 3 else data
        self._before = arr
        self._before_cache.clear()
        self._sync_dims(arr)
        self.update()

    def set_after(self, data: NDArray[np.float32]) -> None:
        arr = np.mean(data, axis=0).astype(np.float32) if data.ndim == 3 else data
        self._after = arr
        self._after_cache.clear()
        self._sync_dims(arr)
        self.update()

    def clear(self) -> None:
        self._before = None
        self._after = None
        self._before_cache.clear()
        self._after_cache.clear()
        self.update()

    def _sync_dims(self, arr: NDArray[np.float32]) -> None:
        h, w = arr.shape
        if h != self._height or w != self._width:
            self._height, self._width = h, w
            self._reset_view()

    @property
    def zoom_level(self) -> float:
        return self._zoom

    def _reset_view(self) -> None:
        if self._width == 0 or self._height == 0:
            return
        self._zoom = min(
            self.width() / max(self._width, 1),
            self.height() / max(self._height, 1),
        )
        self._offset = QPointF(0.0, 0.0)
        self._before_cache.clear()
        self._after_cache.clear()
        self.zoom_changed.emit(self._zoom)

    @Slot()
    def fit_to_view(self) -> None:
        self._reset_view()
        self.update()

    def _get_tile(
        self,
        data: NDArray[np.float32],
        cache: dict[tuple[int, int], QImage],
        tx: int,
        ty: int,
    ) -> QImage:
        key = (tx, ty)
        hit = cache.get(key)
        if hit is not None:
            return hit
        y0, x0 = ty * _TILE_SIZE, tx * _TILE_SIZE
        y1 = min(y0 + _TILE_SIZE, self._height)
        x1 = min(x0 + _TILE_SIZE, self._width)
        tile = data[y0:y1, x0:x1]
        lo, hi = float(np.min(tile)), float(np.max(tile))
        rng = hi - lo if hi > lo else 1.0
        norm = ((tile - lo) / rng * 255).clip(0, 255).astype(np.uint8)
        h, w = norm.shape
        qimg = QImage(norm.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        cache[key] = qimg
        return qimg

    def _paint_half(
        self,
        painter: QPainter,
        data: NDArray[np.float32],
        cache: dict[tuple[int, int], QImage],
        clip_x0: int,
        clip_x1: int,
    ) -> None:
        vp_w, vp_h = self.width(), self.height()
        ix0 = -self._offset.x() / self._zoom
        iy0 = -self._offset.y() / self._zoom
        ix1 = ix0 + vp_w / self._zoom
        iy1 = iy0 + vp_h / self._zoom
        tx0 = max(0, int(ix0) // _TILE_SIZE)
        ty0 = max(0, int(iy0) // _TILE_SIZE)
        tx1 = min((self._width - 1) // _TILE_SIZE, int(ix1) // _TILE_SIZE)
        ty1 = min((self._height - 1) // _TILE_SIZE, int(iy1) // _TILE_SIZE)
        painter.save()
        painter.setClipRect(clip_x0, 0, clip_x1 - clip_x0, vp_h)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile_img = self._get_tile(data, cache, tx, ty)
                sx = self._offset.x() + tx * _TILE_SIZE * self._zoom
                sy = self._offset.y() + ty * _TILE_SIZE * self._zoom
                sw = tile_img.width() * self._zoom
                sh = tile_img.height() * self._zoom
                painter.drawImage(QRectF(sx, sy, sw, sh), tile_img)
        painter.restore()

    def paintEvent(self, _event: object) -> None:
        if self._before is None and self._after is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 1.0)
        vp_h = self.height()
        split_x = int(self.width() * self._split)

        if self._before is not None:
            self._paint_half(painter, self._before, self._before_cache, 0, split_x)
        if self._after is not None:
            self._paint_half(painter, self._after, self._after_cache, split_x, self.width())

        # Divider line
        painter.setPen(QPen(QColor(255, 215, 0), 2))
        painter.drawLine(split_x, 0, split_x, vp_h)

        # Handle triangles at midpoint
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 215, 0))
        mid = vp_h // 2
        sz = 9
        painter.drawPolygon(
            QPolygonF([
                QPointF(float(split_x - 2), float(mid - sz)),
                QPointF(float(split_x - 2 - sz), float(mid)),
                QPointF(float(split_x - 2), float(mid + sz)),
            ])
        )
        painter.drawPolygon(
            QPolygonF([
                QPointF(float(split_x + 2), float(mid - sz)),
                QPointF(float(split_x + 2 + sz), float(mid)),
                QPointF(float(split_x + 2), float(mid + sz)),
            ])
        )

        # Labels
        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        fm = painter.fontMetrics()
        label_y = _LABEL_PAD + fm.ascent()
        painter.drawText(_LABEL_PAD, label_y, "Vorher")
        after_w = fm.horizontalAdvance("Nachher")
        painter.drawText(self.width() - after_w - _LABEL_PAD, label_y, "Nachher")

        painter.end()

    def _near_split(self, x: float) -> bool:
        return abs(x - self.width() * self._split) <= _SPLIT_HIT_PX

    def wheelEvent(self, event: QWheelEvent) -> None:
        pos = event.position()
        old = self._zoom
        if event.angleDelta().y() > 0:
            self._zoom = min(_MAX_ZOOM, self._zoom * _ZOOM_FACTOR)
        else:
            self._zoom = max(_MIN_ZOOM, self._zoom / _ZOOM_FACTOR)
        ratio = self._zoom / old
        self._offset = QPointF(
            pos.x() - ratio * (pos.x() - self._offset.x()),
            pos.y() - ratio * (pos.y() - self._offset.y()),
        )
        self._before_cache.clear()
        self._after_cache.clear()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        if self._near_split(pos.x()):
            self._split_drag = True
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self._pan_drag = True
            self._drag_start = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._split_drag = False
            self._pan_drag = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._split_drag:
            self._split = max(0.05, min(0.95, pos.x() / max(self.width(), 1)))
            self.update()
        elif self._pan_drag:
            delta = pos - self._drag_start
            self._offset += delta
            self._drag_start = pos
            self.update()
        else:
            cursor = (
                Qt.CursorShape.SplitHCursor
                if self._near_split(pos.x())
                else Qt.CursorShape.ArrowCursor
            )
            self.setCursor(cursor)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom = min(_MAX_ZOOM, self._zoom * _ZOOM_FACTOR)
            self._before_cache.clear()
            self._after_cache.clear()
            self.zoom_changed.emit(self._zoom)
            self.update()
        elif key == Qt.Key.Key_Minus:
            self._zoom = max(_MIN_ZOOM, self._zoom / _ZOOM_FACTOR)
            self._before_cache.clear()
            self._after_cache.clear()
            self.zoom_changed.emit(self._zoom)
            self.update()
        elif key == Qt.Key.Key_Home:
            self.fit_to_view()
        elif key == Qt.Key.Key_Left:
            self._offset += QPointF(_PAN_STEP, 0.0)
            self.update()
        elif key == Qt.Key.Key_Right:
            self._offset += QPointF(-_PAN_STEP, 0.0)
            self.update()
        elif key == Qt.Key.Key_Up:
            self._offset += QPointF(0.0, _PAN_STEP)
            self.update()
        elif key == Qt.Key.Key_Down:
            self._offset += QPointF(0.0, -_PAN_STEP)
            self.update()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, _event: object) -> None:
        self._reset_view()
        self.update()
