"""Visual pipeline workflow graph showing processing stages."""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from astroai.ui.models import PipelineModel, StepState

__all__ = ["WorkflowGraph"]

_NODE_W = 120
_NODE_H = 48
_NODE_SPACING = 24
_ARROW_SIZE = 8
_CORNER_RADIUS = 8

_COLORS: dict[StepState, tuple[QColor, QColor, QColor]] = {
    StepState.PENDING: (QColor(42, 32, 24), QColor(58, 46, 30), QColor(168, 148, 120)),
    StepState.ACTIVE: (QColor(58, 42, 16), QColor(138, 106, 48), QColor(237, 224, 204)),
    StepState.DONE: (QColor(26, 42, 24), QColor(74, 122, 58), QColor(180, 220, 170)),
    StepState.ERROR: (QColor(42, 24, 24), QColor(122, 58, 58), QColor(220, 170, 170)),
}


class WorkflowGraph(QWidget):
    """Renders the pipeline as a horizontal node-and-arrow graph."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model.step_changed.connect(self._on_step_changed)
        self._model.pipeline_reset.connect(self._on_reset)
        self.setMinimumHeight(_NODE_H + 24)
        self.setMinimumWidth(200)
        self.setAccessibleName("Pipeline-Workflow")
        self.setToolTip("Verarbeitungs-Pipeline: Kalibrierung \u2192 Registrierung \u2192 Stacking \u2192 Stretching \u2192 Entrauschen")

    @Slot(str, str)
    def _on_step_changed(self, _key: str, _state: str) -> None:
        self.update()

    @Slot()
    def _on_reset(self) -> None:
        self.update()

    def _total_width(self) -> float:
        n = len(self._model.steps)
        return n * _NODE_W + (n - 1) * _NODE_SPACING

    def paintEvent(self, _event: object) -> None:
        steps = self._model.steps
        if not steps:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        total_w = self._total_width()
        x_start = (self.width() - total_w) / 2
        y_center = self.height() / 2

        font = QFont(self.font())
        font.setPointSize(10)
        painter.setFont(font)

        rects: list[QRectF] = []
        for i, step in enumerate(steps):
            x = x_start + i * (_NODE_W + _NODE_SPACING)
            y = y_center - _NODE_H / 2
            rect = QRectF(x, y, _NODE_W, _NODE_H)
            rects.append(rect)

            bg, border, text_color = _COLORS[step.state]

            path = QPainterPath()
            path.addRoundedRect(rect, _CORNER_RADIUS, _CORNER_RADIUS)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg)
            painter.drawPath(path)

            pen_width = 2.5 if step.state is StepState.ACTIVE else 1.5
            painter.setPen(QPen(border, pen_width))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            if step.state is StepState.ACTIVE and step.progress > 0:
                prog_w = rect.width() * step.progress
                clip = QRectF(rect.x(), rect.y(), prog_w, rect.height())
                painter.save()
                painter.setClipRect(clip)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(138, 106, 48, 60))
                painter.drawPath(path)
                painter.restore()

            painter.setPen(text_color)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, step.label)

        arrow_color = QColor(100, 84, 60)
        painter.setPen(QPen(arrow_color, 1.5))
        for i in range(len(rects) - 1):
            r1 = rects[i]
            r2 = rects[i + 1]
            x1 = r1.right() + 2
            x2 = r2.left() - 2
            cy = y_center

            painter.drawLine(int(x1), int(cy), int(x2 - _ARROW_SIZE), int(cy))

            arrow = QPainterPath()
            arrow.moveTo(x2, cy)
            arrow.lineTo(x2 - _ARROW_SIZE, cy - _ARROW_SIZE / 2)
            arrow.lineTo(x2 - _ARROW_SIZE, cy + _ARROW_SIZE / 2)
            arrow.closeSubpath()
            painter.setBrush(arrow_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(arrow)
            painter.setPen(QPen(arrow_color, 1.5))

        painter.end()

    def sizeHint(self) -> QSize:
        w = int(self._total_width()) + 40
        return QSize(max(w, 200), _NODE_H + 24)
