"""StarAnalysisPanel — Qt dock widget for PSF quality inspection.

Displays FWHM histogram, roundness scatter plot, aggregate stats, and
provides a CSV export action for per-star metrics.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from astroai.processing.stars.star_analysis import FrameAnalysisResult, StarMetrics

__all__ = ["StarAnalysisPanel"]

_ACCENT = QColor(200, 168, 96)
_RED = QColor(220, 80, 60)
_GREEN = QColor(80, 200, 100)
_MARGIN = 6


# ──────────────────────────────────────────────────────────────────────────────
# Sub-widgets
# ──────────────────────────────────────────────────────────────────────────────


class _FWHMHistogram(QWidget):
    """Bar histogram of per-star FWHM values."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(90)
        self.setMinimumWidth(120)
        self.setAccessibleName("FWHM-Histogramm")
        self._fwhms: NDArray[np.float64] | None = None
        self._bins: NDArray[np.float64] | None = None
        self._bin_edges: NDArray[np.float64] | None = None

    def set_stars(self, stars: list[StarMetrics]) -> None:
        if not stars:
            self._fwhms = None
            self._bins = None
        else:
            self._fwhms = np.array([s.fwhm for s in stars], dtype=np.float64)
            counts, edges = np.histogram(self._fwhms, bins=min(len(stars), 40))
            self._bins = counts.astype(np.float64)
            self._bin_edges = edges
        self.update()

    def paintEvent(self, _event: object) -> None:
        if self._bins is None or len(self._bins) == 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width() - 2 * _MARGIN
        h = self.height() - 2 * _MARGIN
        if w <= 0 or h <= 0:
            painter.end()
            return

        max_count = float(self._bins.max()) or 1.0
        bar_w = w / len(self._bins)
        path = QPainterPath()
        path.moveTo(_MARGIN, _MARGIN + h)
        for i, val in enumerate(self._bins):
            bar_h = (val / max_count) * h
            x = _MARGIN + i * bar_w
            path.lineTo(x, _MARGIN + h - bar_h)
        path.lineTo(_MARGIN + w, _MARGIN + h)
        path.closeSubpath()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(200, 168, 96, 60))
        painter.drawPath(path)
        painter.setPen(QPen(_ACCENT, 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.end()


class _RoundnessScatter(QWidget):
    """Scatter plot of FWHM (x) vs roundness/ellipticity (y)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMinimumWidth(120)
        self.setAccessibleName("Roundness-Scatter")
        self._stars: list[StarMetrics] = []

    def set_stars(self, stars: list[StarMetrics]) -> None:
        self._stars = stars
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width() - 2 * _MARGIN
        h = self.height() - 2 * _MARGIN

        # Axes
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawLine(_MARGIN, _MARGIN, _MARGIN, _MARGIN + h)
        painter.drawLine(_MARGIN, _MARGIN + h, _MARGIN + w, _MARGIN + h)

        if not self._stars:
            painter.end()
            return

        fwhms = np.array([s.fwhm for s in self._stars])
        ells = np.array([s.ellipticity for s in self._stars])
        f_min, f_max = fwhms.min(), fwhms.max()
        f_range = max(f_max - f_min, 0.01)

        for fwhm, ell in zip(fwhms, ells):
            px = _MARGIN + int((fwhm - f_min) / f_range * w)
            py = _MARGIN + int((1.0 - ell) * h)
            # Green = round (low ell), red = elongated (high ell)
            t = float(np.clip(ell, 0.0, 1.0))
            dot_color = QColor(
                int(t * 220 + (1 - t) * 80),
                int((1 - t) * 200 + t * 80),
                80,
                200,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(dot_color)
            painter.drawEllipse(px - 3, py - 3, 6, 6)

        # Axis labels
        painter.setPen(QColor(140, 140, 140))
        painter.drawText(_MARGIN, self.height() - 2, "FWHM →")
        painter.end()


# ──────────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────────


class StarAnalysisPanel(QWidget):
    """Qt dock panel for PSF / star quality inspection.

    Signals:
        overlay_toggled(bool): emitted when user toggles FWHM overlay.
        export_requested(): emitted when user requests CSV export.
    """

    overlay_toggled: Signal = Signal(bool)
    export_requested: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: FrameAnalysisResult | None = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @Slot(object)
    def set_result(self, result: FrameAnalysisResult) -> None:
        self._result = result
        self._fwhm_hist.set_stars(result.stars)
        self._scatter.set_stars(result.stars)
        self._update_stats(result)

    def clear(self) -> None:
        self._result = None
        self._fwhm_hist.set_stars([])
        self._scatter.set_stars([])
        self._update_stats(FrameAnalysisResult())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Stats group ──────────────────────────────────────────────
        stats_group = QGroupBox("PSF-Statistik")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(3)

        self._lbl_stars = self._make_stat_label("Sterne", "—")
        self._lbl_median_fwhm = self._make_stat_label("Median FWHM", "—")
        self._lbl_p90_fwhm = self._make_stat_label("P90 FWHM", "—")
        self._lbl_ellipticity = self._make_stat_label("Elliptizität", "—")
        self._lbl_strehl = self._make_stat_label("Strehl (proxy)", "—")
        self._lbl_hfr_delta = self._make_stat_label("HFR-Delta", "—")
        self._lbl_limit = self._make_stat_label("FWHM-Limit", "—")

        for row in (
            self._lbl_stars,
            self._lbl_median_fwhm,
            self._lbl_p90_fwhm,
            self._lbl_ellipticity,
            self._lbl_strehl,
            self._lbl_hfr_delta,
            self._lbl_limit,
        ):
            stats_layout.addLayout(row[2])
        layout.addWidget(stats_group)

        # ── FWHM histogram ───────────────────────────────────────────
        hist_group = QGroupBox("FWHM-Verteilung")
        hist_layout = QVBoxLayout(hist_group)
        self._fwhm_hist = _FWHMHistogram()
        hist_layout.addWidget(self._fwhm_hist)
        layout.addWidget(hist_group)

        # ── Roundness scatter ─────────────────────────────────────────
        scatter_group = QGroupBox("Roundness vs. FWHM")
        scatter_layout = QVBoxLayout(scatter_group)
        self._scatter = _RoundnessScatter()
        scatter_layout.addWidget(self._scatter)
        layout.addWidget(scatter_group)

        # ── Overlay toggle ────────────────────────────────────────────
        self._overlay_cb = QCheckBox("FWHM-Heatmap anzeigen")
        self._overlay_cb.setAccessibleName("Räumliche FWHM-Heatmap im Bild einblenden")
        self._overlay_cb.toggled.connect(self.overlay_toggled)
        layout.addWidget(self._overlay_cb)

        # ── CSV export button ─────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._export_btn = QPushButton("CSV exportieren …")
        self._export_btn.setAccessibleName("Per-Stern-Metriken als CSV exportieren")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export_clicked)
        btn_row.addStretch()
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stat_label(
        name: str, value: str
    ) -> tuple[QLabel, QLabel, QHBoxLayout]:
        row = QHBoxLayout()
        row.setSpacing(4)
        lbl_name = QLabel(f"{name}:")
        lbl_name.setMinimumWidth(110)
        lbl_val = QLabel(value)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lbl_name)
        row.addWidget(lbl_val, stretch=1)
        return lbl_name, lbl_val, row

    def _update_stats(self, r: FrameAnalysisResult) -> None:
        self._lbl_stars[1].setText(str(r.star_count) if r.star_count else "—")
        self._lbl_median_fwhm[1].setText(
            f"{r.median_fwhm:.2f} px" if r.median_fwhm else "—"
        )
        self._lbl_p90_fwhm[1].setText(
            f"{r.p90_fwhm:.2f} px" if r.p90_fwhm else "—"
        )
        self._lbl_ellipticity[1].setText(
            f"{r.median_ellipticity:.3f}" if r.star_count else "—"
        )
        self._lbl_strehl[1].setText(
            f"{r.median_strehl:.3f}" if r.star_count else "—"
        )
        self._lbl_hfr_delta[1].setText(
            f"{r.hfr_cross_val_delta:.3f} px" if r.hfr_cross_val_delta else "—"
        )
        if r.exceeds_fwhm_limit:
            self._lbl_limit[1].setText("⚠ überschritten")
            self._lbl_limit[1].setStyleSheet("color: #dc503c;")
        else:
            self._lbl_limit[1].setText("OK" if r.star_count else "—")
            self._lbl_limit[1].setStyleSheet("")
        self._export_btn.setEnabled(bool(r.stars))

    @Slot()
    def _on_export_clicked(self) -> None:
        if self._result is None or not self._result.stars:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "PSF-Metriken exportieren",
            "star_metrics.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        from astroai.processing.stars.star_analysis import StarAnalyzer

        csv_data = StarAnalyzer().to_csv(self._result)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(csv_data)
        self.export_requested.emit()
