"""Sky overlay widget: draws RA/Dec grid and object annotations on images."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from astroai.astrometry.catalog import WcsSolution

__all__ = ["SkyOverlay"]

_GRID_COLOR = QColor(80, 200, 255, 140)
_LABEL_COLOR = QColor(200, 240, 255, 200)
_CROSSHAIR_COLOR = QColor(255, 80, 80, 200)
_GRID_STEPS = 5  # number of RA/Dec lines per axis


class SkyOverlay(QWidget):
    """Transparent overlay that renders an RA/Dec grid over an image widget.

    Parented to an *image_widget* so it always stays aligned.  Call
    :meth:`set_solution` whenever a new plate solution is available.

    Args:
        image_widget: The QWidget over which to draw (typically ImageViewer).
        parent: Optional parent widget.
    """

    def __init__(
        self,
        image_widget: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent or image_widget)
        self._image_widget = image_widget
        self._solution: WcsSolution | None = None
        self._visible_grid = True

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(image_widget.geometry())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_solution(self, solution: WcsSolution | None) -> None:
        """Update the plate solution and redraw the overlay."""
        self._solution = solution
        self.update()

    def set_grid_visible(self, visible: bool) -> None:
        """Show or hide the RA/Dec grid."""
        self._visible_grid = visible
        self.update()

    # ------------------------------------------------------------------
    # Qt painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        if self._solution is None:
            return

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if self._visible_grid:
                self._draw_grid(painter)
            self._draw_center_crosshair(painter)
            self._draw_corner_labels(painter)
        finally:
            painter.end()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self, painter: QPainter) -> None:
        pen = QPen(_GRID_COLOR, 1.0, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        sol = self._solution
        assert sol is not None

        w, h = self.width(), self.height()
        px_w, px_h = float(w), float(h)

        # RA grid lines (constant RA, varying Dec)
        ra_step = sol.fov_width_deg / (_GRID_STEPS + 1)
        ra_start = sol.ra_center - sol.fov_width_deg / 2.0
        for i in range(1, _GRID_STEPS + 1):
            ra = ra_start + i * ra_step
            x = px_w * i / (_GRID_STEPS + 1)
            painter.drawLine(QPointF(x, 0.0), QPointF(x, px_h))

            label = _format_ra(ra)
            painter.setPen(QPen(_LABEL_COLOR))
            painter.setFont(QFont("Monospace", 7))
            painter.drawText(QPointF(x + 2, 10.0), label)
            painter.setPen(pen)

        # Dec grid lines (constant Dec, varying RA)
        dec_step = sol.fov_height_deg / (_GRID_STEPS + 1)
        dec_start = sol.dec_center - sol.fov_height_deg / 2.0
        for i in range(1, _GRID_STEPS + 1):
            dec = dec_start + i * dec_step
            y = px_h * (1.0 - i / (_GRID_STEPS + 1))
            painter.drawLine(QPointF(0.0, y), QPointF(px_w, y))

            label = _format_dec(dec)
            painter.setPen(QPen(_LABEL_COLOR))
            painter.setFont(QFont("Monospace", 7))
            painter.drawText(QPointF(2.0, y - 2.0), label)
            painter.setPen(pen)

    def _draw_center_crosshair(self, painter: QPainter) -> None:
        w, h = float(self.width()), float(self.height())
        cx, cy = w / 2.0, h / 2.0
        size = 12.0
        pen = QPen(_CROSSHAIR_COLOR, 1.5)
        painter.setPen(pen)
        painter.drawLine(QPointF(cx - size, cy), QPointF(cx + size, cy))
        painter.drawLine(QPointF(cx, cy - size), QPointF(cx, cy + size))

    def _draw_corner_labels(self, painter: QPainter) -> None:
        sol = self._solution
        assert sol is not None
        painter.setPen(QPen(_LABEL_COLOR))
        painter.setFont(QFont("Monospace", 8))
        label = (
            f"RA {_format_ra(sol.ra_center)}  "
            f"Dec {_format_dec(sol.dec_center)}  "
            f"{sol.pixel_scale_arcsec:.2f}\"/px"
        )
        painter.drawText(QRectF(4, self.height() - 18, self.width() - 8, 16), label)


# ------------------------------------------------------------------
# Coordinate formatting
# ------------------------------------------------------------------

def _format_ra(ra_deg: float) -> str:
    ra_deg = ra_deg % 360.0
    hours = ra_deg / 15.0
    h = int(hours)
    m = int((hours - h) * 60)
    s = ((hours - h) * 60 - m) * 60
    return f"{h:02d}h{m:02d}m{s:04.1f}s"


def _format_dec(dec_deg: float) -> str:
    sign = "+" if dec_deg >= 0 else "-"
    dec_deg = abs(dec_deg)
    d = int(dec_deg)
    m = int((dec_deg - d) * 60)
    s = ((dec_deg - d) * 60 - m) * 60
    return f"{sign}{d:02d}°{m:02d}'{s:04.1f}\""
