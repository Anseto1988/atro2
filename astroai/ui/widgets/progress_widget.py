"""Progress display with cancel button for long-running operations."""
from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)

__all__ = ["ProgressWidget"]


class ProgressWidget(QWidget):
    """Shows a labeled progress bar with an optional cancel button."""

    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel("Bereit")
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)

        self._cancel_btn = QPushButton("Abbrechen")
        self._cancel_btn.setObjectName("cancelButton")
        self._cancel_btn.setFixedWidth(90)
        self._cancel_btn.setAccessibleName("Vorgang abbrechen")

        self._bar.setAccessibleName("Fortschrittsbalken")
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        self._cancel_btn.setEnabled(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self._label)
        layout.addWidget(self._bar, stretch=1)
        layout.addWidget(self._cancel_btn)

    @Slot(str)
    def set_status(self, text: str) -> None:
        self._label.setText(text)

    @Slot(float)
    def set_progress(self, value: float) -> None:
        self._bar.setValue(int(value * 1000))

    @Slot(bool)
    def set_cancellable(self, enabled: bool) -> None:
        self._cancel_btn.setEnabled(enabled)

    @Slot()
    def reset(self) -> None:
        self._label.setText("Bereit")
        self._bar.setValue(0)
        self._cancel_btn.setEnabled(False)

    @Slot()
    def set_indeterminate(self) -> None:
        self._bar.setRange(0, 0)

    @Slot()
    def set_determinate(self) -> None:
        self._bar.setRange(0, 1000)
