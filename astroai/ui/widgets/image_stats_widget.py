"""Image statistics dock widget — per-channel mean, std, min, max."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

__all__ = ["ImageStatsWidget"]

_HEADERS = ["Kanal", "Mittel", "Std", "Min", "Max"]
_PLACEHOLDER = "—"
_CHANNEL_LABELS_COLOR = ["R", "G", "B"]
_CHANNEL_LABEL_MONO = "L"


def _fmt(value: float) -> str:
    return f"{value:.4f}"


class ImageStatsWidget(QWidget):
    """Shows per-channel statistics for a float32 image array."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._table = QTableWidget(0, len(_HEADERS), self)
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)
        self.clear()

    @Slot(object)
    def set_image_data(self, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            channels = [(_CHANNEL_LABEL_MONO, arr.ravel())]
        elif arr.ndim == 3 and arr.shape[2] == 3:
            channels = [
                (_CHANNEL_LABELS_COLOR[i], arr[:, :, i].ravel()) for i in range(3)
            ]
        else:
            self.clear()
            return

        self._table.setRowCount(len(channels))
        for row, (label, ch) in enumerate(channels):
            self._set_item(row, 0, label)
            self._set_item(row, 1, _fmt(float(np.mean(ch))))
            self._set_item(row, 2, _fmt(float(np.std(ch))))
            self._set_item(row, 3, _fmt(float(np.min(ch))))
            self._set_item(row, 4, _fmt(float(np.max(ch))))

    def clear(self) -> None:
        self._table.setRowCount(1)
        self._set_item(0, 0, _CHANNEL_LABEL_MONO)
        for col in range(1, len(_HEADERS)):
            self._set_item(0, col, _PLACEHOLDER)

    def _set_item(self, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, col, item)
