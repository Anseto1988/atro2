"""Panel showing loaded light frames with quality scores and interactive selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QPoint, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from astroai.project.project_file import FrameEntry

__all__ = ["FrameListPanel"]

_HEADERS = ["Dateiname", "Belichtung (s)", "Qualität", "Ausgewählt"]
_COL_NAME = 0
_COL_EXP = 1
_COL_SCORE = 2
_COL_SEL = 3


def _format_exposure(total_s: float) -> str:
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = int(total_s % 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _score_text(score: float | None) -> str:
    if score is None:
        return "—"
    return f"{score * 100:.1f}%"


def _exposure_text(exp: float | None) -> str:
    if exp is None:
        return "—"
    return f"{exp:.1f}"


class FrameListPanel(QWidget):
    """Table of FrameEntry objects; double-click or right-click to manage selection."""

    selection_changed = Signal(int, bool)   # (frame_index, new_selected)
    remove_requested = Signal(list)         # list[int] — row indices to remove

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[FrameEntry] = []
        self._sort_col: int = -1
        self._sort_asc: bool = True
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._group = QGroupBox("Light-Frames")
        group_layout = QVBoxLayout(self._group)

        self._count_label = QLabel("Keine Frames geladen")
        self._count_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self._count_label)

        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setToolTip(
            "Doppelklick: Auswahl umschalten | Rechtsklick: Massenoperationen"
        )
        self._table.setAccessibleName("Geladene Light-Frames mit Qualitaetsbewertung")
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.horizontalHeader().setSortIndicatorShown(True)
        self._table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)
        self._table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        group_layout.addWidget(self._table)

        layout.addWidget(self._group)

    def refresh(self, entries: list[FrameEntry]) -> None:
        """Repopulate the table; stores a reference; re-applies active sort."""
        self._entries = entries
        if self._sort_col >= 0:
            self._entries.sort(key=self._sort_key_fn(), reverse=not self._sort_asc)
        self._repopulate_table()

    def _repopulate_table(self) -> None:
        """Fill table rows from current _entries order and refresh count label."""
        self._table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            name_item = QTableWidgetItem(Path(entry.path).name)
            name_item.setToolTip(entry.path)
            name_item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )

            exp_item = QTableWidgetItem(_exposure_text(entry.exposure))
            exp_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )

            score_item = QTableWidgetItem(_score_text(entry.quality_score))
            score_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            if entry.quality_score is not None and entry.quality_score < 0.4:
                score_item.setForeground(
                    self._table.palette().color(self._table.palette().ColorRole.BrightText)
                )

            sel_item = QTableWidgetItem("✓" if entry.selected else "✗")
            sel_item.setTextAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )

            self._table.setItem(row, _COL_NAME, name_item)
            self._table.setItem(row, _COL_EXP, exp_item)
            self._table.setItem(row, _COL_SCORE, score_item)
            self._table.setItem(row, _COL_SEL, sel_item)

        self._refresh_count_label()

    def _sort_key_fn(self) -> Callable[[FrameEntry], Any]:
        """Return a sort key function for the active sort column."""
        col = self._sort_col

        def key(e: FrameEntry) -> Any:
            if col == _COL_NAME:
                return Path(e.path).name.lower()
            if col == _COL_EXP:
                return e.exposure if e.exposure is not None else -1.0
            if col == _COL_SCORE:
                return e.quality_score if e.quality_score is not None else -1.0
            if col == _COL_SEL:
                return 1 if e.selected else 0
            return 0

        return key

    # ------------------------------------------------------------------
    # Column sort
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_header_clicked(self, col: int) -> None:
        if not self._entries:
            return
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True
        order = (
            Qt.SortOrder.AscendingOrder if self._sort_asc else Qt.SortOrder.DescendingOrder
        )
        self._table.horizontalHeader().setSortIndicator(col, order)
        self._entries.sort(key=self._sort_key_fn(), reverse=not self._sort_asc)
        self._repopulate_table()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    @Slot(QPoint)
    def _show_context_menu(self, pos: QPoint) -> None:
        if not self._entries:
            return
        selected_rows = sorted({item.row() for item in self._table.selectedItems()})
        menu = QMenu(self)
        select_all_act = menu.addAction("Alle auswählen")
        deselect_all_act = menu.addAction("Alle abwählen")
        invert_act = menu.addAction("Auswahl umkehren")
        menu.addSeparator()
        remove_act = menu.addAction(
            f"{len(selected_rows)} Frame(s) entfernen"
            if selected_rows
            else "Entfernen"
        )
        remove_act.setEnabled(bool(selected_rows))

        action = menu.exec(self._table.mapToGlobal(pos))
        if action == select_all_act:
            self._set_all_selected(True)
        elif action == deselect_all_act:
            self._set_all_selected(False)
        elif action == invert_act:
            self._invert_selection()
        elif action == remove_act and selected_rows:
            self.remove_requested.emit(selected_rows)

    # ------------------------------------------------------------------
    # Bulk selection helpers
    # ------------------------------------------------------------------

    def select_all(self) -> None:
        self._set_all_selected(True)

    def deselect_all(self) -> None:
        self._set_all_selected(False)

    def invert_selection(self) -> None:
        self._invert_selection()

    def _set_all_selected(self, value: bool) -> None:
        for i, entry in enumerate(self._entries):
            if entry.selected != value:
                entry.selected = value
                item = self._table.item(i, _COL_SEL)
                if item is not None:
                    item.setText("✓" if value else "✗")
                self.selection_changed.emit(i, value)
        self._refresh_count_label()

    def _invert_selection(self) -> None:
        for i, entry in enumerate(self._entries):
            entry.selected = not entry.selected
            item = self._table.item(i, _COL_SEL)
            if item is not None:
                item.setText("✓" if entry.selected else "✗")
            self.selection_changed.emit(i, entry.selected)
        self._refresh_count_label()

    # ------------------------------------------------------------------
    # Single-row toggle
    # ------------------------------------------------------------------

    @Slot(int, int)
    def _on_cell_double_clicked(self, row: int, _col: int) -> None:
        if row < 0 or row >= len(self._entries):
            return
        entry = self._entries[row]
        entry.selected = not entry.selected
        sel_item = self._table.item(row, _COL_SEL)
        if sel_item is not None:
            sel_item.setText("✓" if entry.selected else "✗")
        self._refresh_count_label()
        self.selection_changed.emit(row, entry.selected)

    def _refresh_count_label(self) -> None:
        n = len(self._entries)
        if n == 0:
            self._count_label.setText("Keine Frames geladen")
        else:
            scored = sum(1 for e in self._entries if e.quality_score is not None)
            selected = sum(1 for e in self._entries if e.selected)
            total_s = sum(
                e.exposure for e in self._entries
                if e.selected and e.exposure is not None
            )
            exp_str = f" — {_format_exposure(total_s)}" if total_s > 0 else ""
            self._count_label.setText(
                f"{n} Frame(s) — {selected} ausgewählt{exp_str} — {scored} bewertet"
            )
