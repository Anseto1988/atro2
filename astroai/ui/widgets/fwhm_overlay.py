"""Spatial FWHM heatmap overlay for ImageViewer — colour: green (low) → red (high)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from astroai.processing.stars.star_analysis import StarMetrics

__all__ = ["FWHMOverlay"]

_IDW_POWER = 2.0
_IDW_SMOOTH = 4.0
_GRID_MAX = 256       # heatmap grid resolution cap
_ALPHA = 180          # overlay alpha (0–255)


class FWHMOverlay(QWidget):
    """Transparent overlay that renders a spatial FWHM heatmap.

    Place this as a child of ImageViewer (or a sibling covering the viewport).
    Call :meth:`set_data` after every analysis run. The overlay repaints
    automatically and does not intercept mouse events.

    Colour encoding: green = low FWHM, yellow = medium, red = high FWHM.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self._stars: list[StarMetrics] = []
        self._image_size: tuple[int, int] = (0, 0)
        self._opacity: float = 0.6
        self._visible_overlay: bool = False
        self._heatmap: QImage | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @Slot(object, tuple)
    def set_data(
        self,
        stars: list[StarMetrics],
        image_size: tuple[int, int],
    ) -> None:
        self._stars = stars
        self._image_size = image_size
        self._heatmap = None
        self.update()

    @Slot(float)
    def set_opacity(self, value: float) -> None:
        self._opacity = float(np.clip(value, 0.0, 1.0))
        self.update()

    @Slot(bool)
    def set_overlay_visible(self, visible: bool) -> None:
        self._visible_overlay = visible
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event: object) -> None:
        if not self._visible_overlay or not self._stars or self._image_size == (0, 0):
            return
        if self._heatmap is None:
            self._heatmap = self._build_heatmap()
        if self._heatmap is None:
            return
        painter = QPainter(self)
        painter.setOpacity(self._opacity)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(self.rect(), self._heatmap)
        painter.end()

    # ------------------------------------------------------------------
    # Heatmap construction
    # ------------------------------------------------------------------

    def _build_heatmap(self) -> QImage | None:
        iw, ih = self._image_size
        if iw <= 0 or ih <= 0 or not self._stars:
            return None

        gw = min(iw, _GRID_MAX)
        gh = min(ih, _GRID_MAX)

        fwhms = np.array([s.fwhm for s in self._stars], dtype=np.float64)
        xs = np.array([s.x for s in self._stars], dtype=np.float64)
        ys = np.array([s.y for s in self._stars], dtype=np.float64)

        # Normalise star coords to grid space
        xs_g = xs / max(iw - 1, 1) * (gw - 1)
        ys_g = ys / max(ih - 1, 1) * (gh - 1)

        fmin, fmax = fwhms.min(), fwhms.max()
        normalized = (
            np.full_like(fwhms, 0.5)
            if (fmax - fmin) < 1e-6
            else (fwhms - fmin) / (fmax - fmin)
        )

        gx, gy = np.meshgrid(np.arange(gw, dtype=np.float64), np.arange(gh, dtype=np.float64))
        grid = self._idw(xs_g, ys_g, normalized, gx, gy)
        rgba = self._colormap(grid)

        img_bytes = rgba.astype(np.uint8).tobytes()
        qimg = QImage(img_bytes, gw, gh, gw * 4, QImage.Format.Format_RGBA8888)
        return qimg.copy()

    @staticmethod
    def _idw(
        xs: NDArray[np.floating],
        ys: NDArray[np.floating],
        values: NDArray[np.floating],
        gx: NDArray[np.floating],
        gy: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Inverse-distance-weighted interpolation onto a 2-D grid."""
        num = np.zeros(gx.shape, dtype=np.float64)
        den = np.zeros(gx.shape, dtype=np.float64)
        for xi, yi, vi in zip(xs, ys, values):
            dist2 = (gx - xi) ** 2 + (gy - yi) ** 2 + _IDW_SMOOTH**2
            w = 1.0 / (dist2 ** (_IDW_POWER / 2.0))
            num += w * vi
            den += w
        return num / np.maximum(den, 1e-12)

    @staticmethod
    def _colormap(grid: NDArray[np.floating]) -> NDArray[np.floating]:
        """Map [0, 1] to green→yellow→red RGBA."""
        g = np.clip(grid, 0.0, 1.0)
        rgba = np.zeros((*g.shape, 4), dtype=np.float32)
        rgba[..., 0] = 255.0 * np.clip(g * 2.0, 0.0, 1.0)                        # R
        rgba[..., 1] = 255.0 * (1.0 - np.clip(g * 2.0 - 1.0, 0.0, 1.0))          # G
        rgba[..., 2] = 0.0                                                          # B
        rgba[..., 3] = _ALPHA
        return rgba
