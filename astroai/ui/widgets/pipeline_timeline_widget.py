"""Horizontal Gantt-style pipeline timeline widget.

Shows past steps (green), active step (blue, animated), future steps (grey + ETA),
with tooltips and optional DAG-dependency arrows.
"""
from __future__ import annotations

import math
import time
from typing import Sequence

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, QTimer, Slot
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QLabel, QSizePolicy, QToolTip, QWidget

from astroai.core.pipeline.timing import PipelineTimer, StepTiming

__all__ = ["PipelineTimelineWidget"]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_BAR_H = 28
_BAR_SPACING = 8
_LABEL_W = 120
_ETA_W = 64
_ANIM_PERIOD = 1.2          # seconds for one shimmer cycle
_CORNER_R = 5.0

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
_COL_DONE = QColor(58, 140, 70)
_COL_DONE_BORDER = QColor(80, 180, 90)
_COL_ACTIVE = QColor(50, 110, 190)
_COL_ACTIVE_BORDER = QColor(80, 155, 240)
_COL_PENDING = QColor(54, 50, 46)
_COL_PENDING_BORDER = QColor(80, 74, 68)
_COL_TEXT = QColor(230, 230, 220)
_COL_TEXT_DIM = QColor(140, 132, 120)
_COL_ARROW = QColor(100, 90, 76)


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec:02d}s"


class _StepEntry:
    """Data for one row in the timeline."""

    __slots__ = ("name", "state", "elapsed", "eta", "index")

    DONE = "done"
    ACTIVE = "active"
    PENDING = "pending"

    def __init__(
        self,
        name: str,
        state: str,
        elapsed: float | None,
        eta: float | None,
        index: int,
    ) -> None:
        self.name = name
        self.state = state
        self.elapsed = elapsed
        self.eta = eta
        self.index = index


class PipelineTimelineWidget(QWidget):
    """Horizontal Gantt widget showing pipeline step progress with ETA.

    Wire it up like this::

        timeline = PipelineTimelineWidget()
        # When pipeline starts:
        timeline.set_plan(["Load", "Calibrate", "Register", "Stack"])
        # Each progress callback:
        timeline.update_timer(my_pipeline_timer)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[_StepEntry] = []
        self._show_dag: bool = False
        self._anim_phase: float = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(self._preferred_height())
        self.setAccessibleName("Pipeline-Zeitstrahl")
        self._timer.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_plan(self, step_names: Sequence[str]) -> None:
        """Initialise the widget with all step names as pending."""
        self._entries = [
            _StepEntry(n, _StepEntry.PENDING, None, None, i)
            for i, n in enumerate(step_names)
        ]
        self._refresh_size()
        self.update()

    def update_timer(self, timer: PipelineTimer) -> None:
        """Sync widget state from a live PipelineTimer."""
        timings = timer.timings
        active = timer.active

        done_types = {t.step_type for t in timings if t.is_finished}

        for entry in self._entries:
            # Find matching timing by name
            matched = next(
                (t for t in timings if t.step_type == entry.name), None
            )
            if matched is not None and matched.is_finished:
                entry.state = _StepEntry.DONE
                entry.elapsed = matched.elapsed
                entry.eta = None
            elif active is not None and active.step_type == entry.name:
                entry.state = _StepEntry.ACTIVE
                entry.elapsed = active.elapsed
                entry.eta = timer.eta_for(entry.name)
            elif entry.state != _StepEntry.DONE:
                entry.state = _StepEntry.PENDING
                entry.elapsed = None
                entry.eta = timer.eta_for(entry.name)

        self.update()

    def set_dag_overlay(self, enabled: bool) -> None:
        """Toggle optional dependency arrows between steps."""
        self._show_dag = enabled
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preferred_height(self) -> int:
        n = max(len(self._entries), 1)
        return n * (_BAR_H + _BAR_SPACING) + _BAR_SPACING

    def _refresh_size(self) -> None:
        h = self._preferred_height()
        self.setMinimumHeight(h)
        self.setFixedHeight(h)

    @Slot()
    def _tick(self) -> None:
        self._anim_phase = (time.monotonic() % _ANIM_PERIOD) / _ANIM_PERIOD
        has_active = any(e.state == _StepEntry.ACTIVE for e in self._entries)
        if has_active:
            self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event: object) -> None:
        if not self._entries:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont(self.font())
        font.setPointSize(9)
        painter.setFont(font)

        w = self.width()
        bar_w = w - _LABEL_W - _ETA_W - 16

        for i, entry in enumerate(self._entries):
            y = _BAR_SPACING + i * (_BAR_H + _BAR_SPACING)
            self._draw_label(painter, entry, y)
            self._draw_bar(painter, entry, y, bar_w)
            self._draw_eta(painter, entry, y, bar_w)

        if self._show_dag:
            self._draw_dag_arrows(painter, bar_w)

        painter.end()

    def _draw_label(self, painter: QPainter, entry: _StepEntry, y: int) -> None:
        rect = QRectF(0, y, _LABEL_W, _BAR_H)
        color = _COL_TEXT if entry.state != _StepEntry.PENDING else _COL_TEXT_DIM
        painter.setPen(color)
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            entry.name + "  ",
        )

    def _draw_bar(self, painter: QPainter, entry: _StepEntry, y: int, bar_w: int) -> None:
        x = _LABEL_W + 8
        rect = QRectF(x, y, bar_w, _BAR_H)
        path = QPainterPath()
        path.addRoundedRect(rect, _CORNER_R, _CORNER_R)

        if entry.state == _StepEntry.DONE:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_COL_DONE)
            painter.drawPath(path)
            painter.setPen(QPen(_COL_DONE_BORDER, 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            # Duration label inside
            painter.setPen(_COL_TEXT)
            if entry.elapsed is not None:
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, _fmt_seconds(entry.elapsed))

        elif entry.state == _StepEntry.ACTIVE:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_COL_ACTIVE)
            painter.drawPath(path)
            self._draw_shimmer(painter, rect, path)
            painter.setPen(QPen(_COL_ACTIVE_BORDER, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            # Elapsed so far
            painter.setPen(_COL_TEXT)
            if entry.elapsed is not None:
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, _fmt_seconds(entry.elapsed))

        else:  # pending
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_COL_PENDING)
            painter.drawPath(path)
            painter.setPen(QPen(_COL_PENDING_BORDER, 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

    def _draw_shimmer(self, painter: QPainter, rect: QRectF, path: QPainterPath) -> None:
        """Animated highlight sweep for active step."""
        shine_x = rect.x() + self._anim_phase * (rect.width() + 40) - 20
        grad = QLinearGradient(shine_x - 20, 0, shine_x + 20, 0)
        grad.setColorAt(0.0, QColor(255, 255, 255, 0))
        grad.setColorAt(0.5, QColor(255, 255, 255, 45))
        grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.save()
        painter.setClipPath(path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(grad)
        painter.drawRect(rect)
        painter.restore()

    def _draw_eta(self, painter: QPainter, entry: _StepEntry, y: int, bar_w: int) -> None:
        x = _LABEL_W + 8 + bar_w + 6
        rect = QRectF(x, y, _ETA_W - 6, _BAR_H)
        if entry.state == _StepEntry.PENDING and entry.eta is not None:
            painter.setPen(_COL_TEXT_DIM)
            painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             f"~{_fmt_seconds(entry.eta)}")
        elif entry.state == _StepEntry.ACTIVE and entry.eta is not None:
            painter.setPen(QColor(160, 200, 255))
            painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             f"~{_fmt_seconds(entry.eta)}")

    def _draw_dag_arrows(self, painter: QPainter, bar_w: int) -> None:
        """Draw simple vertical dependency lines between sequential steps."""
        x_center = _LABEL_W + 8 + bar_w / 2
        for i in range(len(self._entries) - 1):
            y1 = _BAR_SPACING + i * (_BAR_H + _BAR_SPACING) + _BAR_H
            y2 = _BAR_SPACING + (i + 1) * (_BAR_H + _BAR_SPACING)
            painter.setPen(QPen(_COL_ARROW, 1.0, Qt.PenStyle.DotLine))
            painter.drawLine(int(x_center), y1, int(x_center), y2)

    # ------------------------------------------------------------------
    # Tooltip on hover
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event: object) -> None:
        pos = event.pos() if hasattr(event, "pos") else QPoint(0, 0)
        for i, entry in enumerate(self._entries):
            y = _BAR_SPACING + i * (_BAR_H + _BAR_SPACING)
            if y <= pos.y() <= y + _BAR_H:
                self._show_tooltip(entry, event.globalPos())
                return
        QToolTip.hideText()

    def _show_tooltip(self, entry: _StepEntry, global_pos: QPoint) -> None:
        lines = [f"<b>{entry.name}</b>"]
        if entry.elapsed is not None:
            lines.append(f"Dauer: {_fmt_seconds(entry.elapsed)}")
        if entry.eta is not None:
            lines.append(f"ETA: ~{_fmt_seconds(entry.eta)}")
        state_label = {
            _StepEntry.DONE: "Abgeschlossen",
            _StepEntry.ACTIVE: "Läuft",
            _StepEntry.PENDING: "Ausstehend",
        }.get(entry.state, entry.state)
        lines.append(f"Status: {state_label}")
        QToolTip.showText(global_pos, "<br>".join(lines), self)

    # ------------------------------------------------------------------
    # Size hint
    # ------------------------------------------------------------------

    def sizeHint(self) -> QSize:
        return QSize(600, self._preferred_height())
