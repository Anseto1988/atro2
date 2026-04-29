"""Qt dock panel showing per-frame AI alignment quality scores."""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["AlignmentQualityPanel"]

_COL_FRAME = 0
_COL_SCORE = 1
_COL_STATUS = 2
_COL_REASON = 3
_HEADERS = ["Frame", "Score", "Status", "Ablehnungsgrund"]

_COLOR_ACCEPTED = QColor(0xCC, 0xFF, 0xCC)     # light green
_COLOR_REJECTED = QColor(0xFF, 0xCC, 0xCC)     # light red
_COLOR_REFERENCE = QColor(0xCC, 0xDD, 0xFF)    # light blue


class AlignmentQualityPanel(QWidget):
    """Per-frame alignment confidence score table with rejection indicators."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("AI-Alignment Qualität")
        group_layout = QVBoxLayout(group)

        # Summary row
        summary_row = QHBoxLayout()
        self._summary_label = QLabel("Keine Ergebnisse")
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        summary_row.addWidget(self._summary_label)
        summary_row.addStretch()
        group_layout.addLayout(summary_row)

        # Results table
        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(False)
        self._table.setAccessibleName(
            "AI-Alignment Qualitätstabelle – Frame-Scores und Ablehnungsgründe"
        )
        group_layout.addWidget(self._table)

        layout.addWidget(group)

    def _connect_signals(self) -> None:
        self._model.pipeline_reset.connect(self._on_reset)

    @Slot()
    def _on_reset(self) -> None:
        self._table.setRowCount(0)
        self._summary_label.setText("Keine Ergebnisse")

    def update_alignment_results(
        self,
        scores: list[float],
        rejected: list[bool],
        reasons: list[str],
        reference_index: int = 0,
    ) -> None:
        """Populate the table with per-frame alignment results.

        Args:
            scores: Per-frame confidence values (0.0 … 1.0).
            rejected: Per-frame rejection flags.
            reasons: Human-readable rejection reason per frame (empty string if accepted).
            reference_index: Index of the reference frame (always shown as accepted).
        """
        n = len(scores)
        self._table.setRowCount(n)

        n_rejected = sum(rejected)
        n_accepted = n - n_rejected
        self._summary_label.setText(
            f"{n_accepted}/{n} Frames akzeptiert  "
            f"({n_rejected} abgelehnt)"
        )

        for i, (score, is_rejected, reason) in enumerate(zip(scores, rejected, reasons)):
            is_ref = i == reference_index

            status = "Referenz" if is_ref else ("Abgelehnt" if is_rejected else "OK")
            score_text = "—" if is_ref else f"{score:.3f}"
            bg = _COLOR_REFERENCE if is_ref else (
                _COLOR_REJECTED if is_rejected else _COLOR_ACCEPTED
            )

            for col, text in enumerate([str(i), score_text, status, reason]):
                item = QTableWidgetItem(text)
                item.setBackground(bg)
                if col == _COL_SCORE and not is_ref:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self._table.setItem(i, col, item)

        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)
