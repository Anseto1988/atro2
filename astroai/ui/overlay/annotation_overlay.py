"""Transparent annotation overlay rendered on top of the ImageViewer."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from astroai.ui.overlay.sky_objects import (
    CatalogObject,
    SkyObjectCatalog,
    WcsTransform,
)

if TYPE_CHECKING:
    from astroai.ui.widgets.image_viewer import ImageViewer

__all__ = ["AnnotationOverlay"]

_COLOR_DSO = QColor(120, 200, 255, 180)
_COLOR_STAR = QColor(255, 220, 100, 200)
_COLOR_BOUNDARY = QColor(80, 180, 80, 100)
_COLOR_GRID = QColor(100, 100, 180, 60)

_FONT_LABEL = QFont("Segoe UI", 9)
_FONT_LABEL.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)

_DSO_MARKER_RADIUS = 8.0
_STAR_MARKER_RADIUS = 4.0
_LABEL_OFFSET = 12.0


class AnnotationOverlay(QWidget):
    """Paints sky annotations over the image viewer using WCS coordinates."""

    def __init__(self, viewer: ImageViewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._wcs: WcsTransform | None = None
        self._catalog = SkyObjectCatalog()

        self._show_dso = True
        self._show_stars = True
        self._show_boundaries = False
        self._show_grid = False

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        viewer.zoom_changed.connect(self._on_zoom_changed)
        viewer.view_changed.connect(self._on_view_changed)

    def set_wcs(self, wcs: WcsTransform | None) -> None:
        self._wcs = wcs
        self.update()

    @property
    def has_wcs(self) -> bool:
        return self._wcs is not None

    def set_show_dso(self, enabled: bool) -> None:
        if self._show_dso != enabled:
            self._show_dso = enabled
            self.update()

    def set_show_stars(self, enabled: bool) -> None:
        if self._show_stars != enabled:
            self._show_stars = enabled
            self.update()

    def set_show_boundaries(self, enabled: bool) -> None:
        if self._show_boundaries != enabled:
            self._show_boundaries = enabled
            self.update()

    def set_show_grid(self, enabled: bool) -> None:
        if self._show_grid != enabled:
            self._show_grid = enabled
            self.update()

    @Slot(float)
    def _on_zoom_changed(self, _zoom: float) -> None:
        self.update()

    @Slot()
    def _on_view_changed(self) -> None:
        self.update()

    def _world_to_screen(self, ra_deg: float, dec_deg: float) -> QPointF | None:
        if self._wcs is None:
            return None
        result = self._wcs.world_to_pixel(ra_deg, dec_deg)
        if result is None:
            return None
        px, py = result
        zoom = self._viewer.zoom_level
        offset = self._viewer._offset
        sx = offset.x() + px * zoom
        sy = offset.y() + py * zoom
        return QPointF(sx, sy)

    def _is_in_viewport(self, screen_pt: QPointF, margin: float = 50.0) -> bool:
        return (
            -margin <= screen_pt.x() <= self.width() + margin
            and -margin <= screen_pt.y() <= self.height() + margin
        )

    def paintEvent(self, _event: object) -> None:
        if self._wcs is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._show_boundaries:
            self._paint_boundaries(painter)
        if self._show_grid:
            self._paint_grid(painter)
        if self._show_dso:
            self._paint_dso(painter)
        if self._show_stars:
            self._paint_stars(painter)

        painter.end()

    def _paint_dso(self, painter: QPainter) -> None:
        pen = QPen(_COLOR_DSO, 1.5)
        painter.setPen(pen)
        painter.setFont(_FONT_LABEL)

        for obj in self._catalog.deep_sky_objects:
            pt = self._world_to_screen(obj.ra_deg, obj.dec_deg)
            if pt is None or not self._is_in_viewport(pt):
                continue
            self._draw_dso_marker(painter, pt, obj)

    def _draw_dso_marker(self, painter: QPainter, pt: QPointF, obj: CatalogObject) -> None:
        r = _DSO_MARKER_RADIUS
        obj_type = obj.obj_type.lower()

        if obj_type in ("gx", "galaxy"):
            painter.drawEllipse(pt, r * 1.5, r * 0.8)
        elif obj_type in ("oc", "open cluster"):
            painter.drawEllipse(pt, r, r)
            painter.drawLine(pt + QPointF(-r, 0), pt + QPointF(r, 0))
            painter.drawLine(pt + QPointF(0, -r), pt + QPointF(0, r))
        elif obj_type in ("gc", "globular cluster"):
            painter.drawEllipse(pt, r, r)
            painter.drawLine(pt + QPointF(-r, 0), pt + QPointF(r, 0))
            painter.drawLine(pt + QPointF(0, -r), pt + QPointF(0, r))
            inner = r * 0.5
            painter.drawEllipse(pt, inner, inner)
        elif obj_type in ("pn", "planetary nebula"):
            painter.drawEllipse(pt, r, r)
            painter.drawLine(pt + QPointF(-r - 3, 0), pt + QPointF(-r, 0))
            painter.drawLine(pt + QPointF(r, 0), pt + QPointF(r + 3, 0))
            painter.drawLine(pt + QPointF(0, -r - 3), pt + QPointF(0, -r))
            painter.drawLine(pt + QPointF(0, r), pt + QPointF(0, r + 3))
        else:
            painter.drawRect(QRectF(pt.x() - r, pt.y() - r, r * 2, r * 2))

        label = obj.common_name or obj.designation
        painter.drawText(pt + QPointF(_LABEL_OFFSET, -4), label)

    def _paint_stars(self, painter: QPainter) -> None:
        pen = QPen(_COLOR_STAR, 1.0)
        painter.setPen(pen)
        painter.setFont(_FONT_LABEL)

        for star in self._catalog.named_stars:
            pt = self._world_to_screen(star.ra_deg, star.dec_deg)
            if pt is None or not self._is_in_viewport(pt):
                continue
            r = max(2.0, _STAR_MARKER_RADIUS - star.magnitude * 0.3)
            painter.drawEllipse(pt, r, r)
            painter.drawText(pt + QPointF(_LABEL_OFFSET, 4), star.name)

    def _paint_boundaries(self, painter: QPainter) -> None:
        pen = QPen(_COLOR_BOUNDARY, 1.0, Qt.PenStyle.DotLine)
        painter.setPen(pen)

        for seg in self._catalog.constellation_boundaries:
            p1 = self._world_to_screen(seg.ra1_deg, seg.dec1_deg)
            p2 = self._world_to_screen(seg.ra2_deg, seg.dec2_deg)
            if p1 is None or p2 is None:
                continue
            if not self._is_in_viewport(p1, 200) and not self._is_in_viewport(p2, 200):
                continue
            painter.drawLine(p1, p2)

    def _paint_grid(self, painter: QPainter) -> None:
        if self._wcs is None:
            return

        pen = QPen(_COLOR_GRID, 0.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setFont(QFont("Segoe UI", 7))

        w, h = self._wcs.image_size()
        corners = []
        for x, y in [(0, 0), (w, 0), (w, h), (0, h)]:
            coord = self._wcs.pixel_to_world(float(x), float(y))
            if coord is not None:
                corners.append(coord)

        if len(corners) < 2:
            return

        ra_vals = [c[0] for c in corners]
        dec_vals = [c[1] for c in corners]
        ra_min, ra_max = min(ra_vals), max(ra_vals)
        dec_min, dec_max = min(dec_vals), max(dec_vals)

        ra_span = ra_max - ra_min
        dec_span = dec_max - dec_min
        grid_step = self._choose_grid_step(max(ra_span, dec_span))

        ra_start = int(ra_min / grid_step) * grid_step
        dec_start = int(dec_min / grid_step) * grid_step

        steps = 50
        for ra in _frange(ra_start, ra_max + grid_step, grid_step):
            points = []
            for i in range(steps + 1):
                dec = dec_min + (dec_max - dec_min) * i / steps
                pt = self._world_to_screen(ra, dec)
                if pt is not None:
                    points.append(pt)
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])
            if points:
                painter.drawText(points[0] + QPointF(2, -2), f"{ra:.1f}\u00b0")

        for dec in _frange(dec_start, dec_max + grid_step, grid_step):
            points = []
            for i in range(steps + 1):
                ra = ra_min + (ra_max - ra_min) * i / steps
                pt = self._world_to_screen(ra, dec)
                if pt is not None:
                    points.append(pt)
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])
            if points:
                painter.drawText(points[0] + QPointF(2, 12), f"{dec:.1f}\u00b0")

    @staticmethod
    def _choose_grid_step(span_deg: float) -> float:
        for step in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0):
            if span_deg / step <= 12:
                return step
        return 30.0


def _frange(start: float, stop: float, step: float) -> list[float]:
    result: list[float] = []
    v = start
    while v <= stop:
        result.append(v)
        v += step
    return result
