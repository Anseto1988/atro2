"""Interactive tone-curve editor panel."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from PySide6.QtCore import QPointF, Qt, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPaintEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["CurvesPanel"]

_IDENTITY: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
_MIN_POINTS = 2
_MAX_POINTS = 10
_HIT_RADIUS = 12.0


class CurveEditor(QWidget):
    """Custom curve-editor widget with draggable Bezier control points."""

    points_changed = Signal(list)

    _MARGIN = 24

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float]] = list(_IDENTITY)
        self._drag_idx: int = -1
        self.setMinimumSize(220, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAccessibleName("Kurven-Editor, Linksklick zum Hinzufügen von Punkten, Rechtsklick zum Entfernen")

    # -- public API --

    def get_points(self) -> list[tuple[float, float]]:
        return list(self._points)

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._points = list(points) if len(points) >= _MIN_POINTS else list(_IDENTITY)
        self.update()

    def reset(self) -> None:
        self._points = list(_IDENTITY)
        self.update()
        self.points_changed.emit(list(self._points))

    # -- coordinate helpers --

    def _plot_rect(self) -> tuple[float, float, float, float]:
        m = self._MARGIN
        return m, m, self.width() - 2 * m, self.height() - 2 * m

    def _to_widget(self, x: float, y: float) -> QPointF:
        ox, oy, w, h = self._plot_rect()
        return QPointF(ox + x * w, oy + (1.0 - y) * h)

    def _from_widget(self, px: float, py: float) -> tuple[float, float]:
        ox, oy, w, h = self._plot_rect()
        x = (px - ox) / max(w, 1.0)
        y = 1.0 - (py - oy) / max(h, 1.0)
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    def _nearest_point_idx(self, px: float, py: float) -> int:
        for i, (x, y) in enumerate(self._points):
            wp = self._to_widget(x, y)
            if abs(wp.x() - px) < _HIT_RADIUS and abs(wp.y() - py) < _HIT_RADIUS:
                return i
        return -1

    # -- mouse events --

    def mousePressEvent(self, event: QMouseEvent) -> None:
        px, py = event.position().x(), event.position().y()
        idx = self._nearest_point_idx(px, py)
        if event.button() == Qt.MouseButton.LeftButton:
            if idx >= 0:
                self._drag_idx = idx
            elif len(self._points) < _MAX_POINTS:
                nx, ny = self._from_widget(px, py)
                self._points.append((nx, ny))
                self._points.sort(key=lambda p: p[0])
                self._drag_idx = next(
                    i for i, p in enumerate(self._points) if p == (nx, ny)
                )
                self.update()
                self.points_changed.emit(list(self._points))
        elif event.button() == Qt.MouseButton.RightButton:
            if idx >= 0 and len(self._points) > _MIN_POINTS:
                # Endpoints cannot be removed
                if idx == 0 or idx == len(self._points) - 1:
                    return
                self._points.pop(idx)
                self.update()
                self.points_changed.emit(list(self._points))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_idx < 0:
            return
        px, py = event.position().x(), event.position().y()
        nx, ny = self._from_widget(px, py)
        # Lock x for first and last endpoints
        if self._drag_idx == 0:
            nx = 0.0
        elif self._drag_idx == len(self._points) - 1:
            nx = 1.0
        self._points[self._drag_idx] = (nx, ny)
        self._points.sort(key=lambda p: p[0])
        self.update()
        self.points_changed.emit(list(self._points))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_idx = -1

    # -- painting --

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        ox, oy, w, h = self._plot_rect()
        iw, ih = int(w), int(h)

        # Background
        p.fillRect(self.rect(), QColor(28, 28, 28))

        # Grid (4x4)
        grid_pen = QPen(QColor(48, 48, 48))
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        for i in range(1, 4):
            xg = ox + i * w / 4
            yg = oy + i * h / 4
            p.drawLine(int(xg), int(oy), int(xg), int(oy + ih))
            p.drawLine(int(ox), int(yg), int(ox + iw), int(yg))

        # Plot border
        border_pen = QPen(QColor(70, 70, 70))
        border_pen.setWidth(1)
        p.setPen(border_pen)
        p.drawRect(int(ox), int(oy), iw, ih)

        # Identity diagonal
        diag_pen = QPen(QColor(70, 70, 70))
        diag_pen.setStyle(Qt.PenStyle.DashLine)
        diag_pen.setWidth(1)
        p.setPen(diag_pen)
        p.drawLine(int(ox), int(oy + ih), int(ox + iw), int(oy))

        # Interpolated curve
        if len(self._points) >= 2:
            xs = np.array([pt[0] for pt in self._points], dtype=np.float64)
            ys = np.array([pt[1] for pt in self._points], dtype=np.float64)
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            _, uniq = np.unique(xs, return_index=True)
            xs, ys = xs[uniq], ys[uniq]
            try:
                cs = CubicSpline(xs, ys, bc_type="clamped")
                t = np.linspace(0.0, 1.0, 256)
                cy = np.clip(cs(t), 0.0, 1.0)
                path = QPainterPath()
                first = self._to_widget(float(t[0]), float(cy[0]))
                path.moveTo(first)
                for xi, yi in zip(t[1:], cy[1:]):
                    path.lineTo(self._to_widget(float(xi), float(yi)))
                curve_pen = QPen(QColor(220, 190, 80))
                curve_pen.setWidth(2)
                p.setPen(curve_pen)
                p.drawPath(path)
            except Exception:
                pass

        # Control points
        pt_pen = QPen(QColor(200, 200, 200))
        pt_pen.setWidth(1)
        p.setPen(pt_pen)
        p.setBrush(QBrush(QColor(200, 200, 200)))
        for x, y in self._points:
            wp = self._to_widget(x, y)
            p.drawEllipse(wp, 5.0, 5.0)


class CurvesPanel(QWidget):
    """Dock-panel for interactive tone-curve adjustment."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Tonwertkurven")
        group_layout = QVBoxLayout(group)

        # Channel selector
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Kanal:"))
        self._channel_combo = QComboBox()
        self._channel_combo.addItems(["Alle Kanäle", "R", "G", "B", "Lum"])
        self._channel_combo.setAccessibleName("Kanal-Auswahl für Kurvenanpassung")
        ch_row.addWidget(self._channel_combo, stretch=1)
        group_layout.addLayout(ch_row)

        # Curve editor canvas
        self._curve_editor = CurveEditor(self)
        group_layout.addWidget(self._curve_editor)

        # Info label
        info = QLabel(
            "Klick: Punkt hinzufügen  |  Ziehen: Punkt verschieben\n"
            "Rechtsklick: Punkt entfernen (min. 2, max. 10)"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-size: 10px;")
        group_layout.addWidget(info)

        # Reset button
        self._reset_btn = QPushButton("Alle Kurven zurücksetzen")
        self._reset_btn.setAccessibleName("Alle Tonwertkurven auf Diagonale zurücksetzen")
        group_layout.addWidget(self._reset_btn)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self._curve_editor.points_changed.connect(self._on_points_changed)
        self._reset_btn.clicked.connect(self._on_reset)
        self._model.curves_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _current_channel_points(self) -> list[tuple[float, float]]:
        ch = self._channel_combo.currentText()
        if ch == "R":
            return list(self._model.curves_r_points)
        if ch == "G":
            return list(self._model.curves_g_points)
        if ch == "B":
            return list(self._model.curves_b_points)
        # "Alle Kanäle" and "Lum" both use the RGB-composite curve
        return list(self._model.curves_rgb_points)

    def _sync_from_model(self) -> None:
        self._curve_editor.blockSignals(True)
        self._curve_editor.set_points(self._current_channel_points())
        self._curve_editor.blockSignals(False)
        self._curve_editor.update()

    @Slot(int)
    def _on_channel_changed(self, _: int) -> None:
        self._sync_from_model()

    @Slot(list)
    def _on_points_changed(self, points: list[tuple[float, float]]) -> None:
        ch = self._channel_combo.currentText()
        pts: list[tuple[float, float]] = [(float(x), float(y)) for x, y in points]
        if ch == "R":
            self._model.curves_r_points = pts
        elif ch == "G":
            self._model.curves_g_points = pts
        elif ch == "B":
            self._model.curves_b_points = pts
        else:
            self._model.curves_rgb_points = pts

    @Slot()
    def _on_reset(self) -> None:
        identity = list(_IDENTITY)
        self._model.curves_rgb_points = identity
        self._model.curves_r_points = identity
        self._model.curves_g_points = identity
        self._model.curves_b_points = identity
