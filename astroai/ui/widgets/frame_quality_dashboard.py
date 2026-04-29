"""Frame Quality Dashboard — per-frame score table with CSV export."""
from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["FrameQualityDashboard"]

_COL_INDEX = 0
_COL_SCORE = 1
_COL_STATUS = 2
_HEADERS = ["Frame", "Score", "Status"]

_STATUS_ACCEPTED = "Akzeptiert"
_STATUS_REJECTED = "Abgelehnt"


class _NumericItem(QTableWidgetItem):
    """Table item that sorts numerically."""

    def __init__(self, value: float, display: str) -> None:
        super().__init__(display)
        self._value = value
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setFlags(self.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _NumericItem):
            return self._value < other._value
        return super().__lt__(other)  # type: ignore[arg-type]


class FrameQualityDashboard(QWidget):
    """Sortable table showing per-frame quality scores with CSV export."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scores: list[float] = []
        self._rejected: set[int] = set()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._summary_label = QLabel("Keine Scores vorhanden")
        self._summary_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._summary_label)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_INDEX, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_SCORE, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_STATUS, QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._export_btn = QPushButton("CSV exportieren")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export_clicked)
        btn_row.addStretch()
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

    def set_scores(
        self,
        scores: list[float],
        rejected_indices: list[int] | None = None,
    ) -> None:
        """Populate the table with *scores*. *rejected_indices* marks which frames were rejected."""
        self._scores = list(scores)
        self._rejected = set(rejected_indices or [])

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(scores))

        for i, score in enumerate(scores):
            status = _STATUS_REJECTED if i in self._rejected else _STATUS_ACCEPTED

            idx_item = _NumericItem(float(i), str(i))
            score_item = _NumericItem(score, f"{score:.4f}")
            status_item = QTableWidgetItem(status)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            if i in self._rejected:
                for item in (idx_item, score_item, status_item):
                    item.setForeground(Qt.GlobalColor.red)
            else:
                for item in (idx_item, score_item, status_item):
                    item.setForeground(Qt.GlobalColor.green)

            self._table.setItem(i, _COL_INDEX, idx_item)
            self._table.setItem(i, _COL_SCORE, score_item)
            self._table.setItem(i, _COL_STATUS, status_item)

        self._table.setSortingEnabled(True)
        self._export_btn.setEnabled(bool(scores))
        self._update_summary()

    def clear(self) -> None:
        self._scores = []
        self._rejected = set()
        self._table.setRowCount(0)
        self._export_btn.setEnabled(False)
        self._summary_label.setText("Keine Scores vorhanden")

    def export_csv(self, path: str | Path) -> int:
        """Write table data to CSV. Returns number of data rows written."""
        rows = self.row_data()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(_HEADERS)
            writer.writerows(rows)
        return len(rows)

    def row_data(self) -> list[tuple[int, float, str]]:
        """Return current visible row data as (frame_index, score, status) tuples."""
        result: list[tuple[int, float, str]] = []
        for i, (score, status) in enumerate(
            zip(self._scores, [
                _STATUS_REJECTED if j in self._rejected else _STATUS_ACCEPTED
                for j in range(len(self._scores))
            ])
        ):
            result.append((i, score, status))
        return result

    def accepted_count(self) -> int:
        return sum(1 for i in range(len(self._scores)) if i not in self._rejected)

    def rejected_count(self) -> int:
        return len(self._rejected)

    def total_count(self) -> int:
        return len(self._scores)

    def _update_summary(self) -> None:
        n = len(self._scores)
        if n == 0:
            self._summary_label.setText("Keine Scores vorhanden")
            return
        accepted = self.accepted_count()
        rejected = self.rejected_count()
        mean = sum(self._scores) / n
        self._summary_label.setText(
            f"{n} Frames  |  Akzeptiert: {accepted}  |  Abgelehnt: {rejected}  |  Ø Score: {mean:.4f}"
        )

    @Slot()
    def _on_export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Frame-Scores exportieren",
            "frame_scores.csv",
            "CSV-Dateien (*.csv)",
        )
        if path:
            self.export_csv(path)
