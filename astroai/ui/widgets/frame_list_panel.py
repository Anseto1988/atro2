"""Panel showing loaded light frames with quality scores and interactive selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QPoint, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QAction, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
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
    files_dropped = Signal(list)            # list[str] — absolute file paths dropped onto panel
    preview_requested = Signal(str)         # str — absolute path of frame to preview

    _FITS_SUFFIXES: frozenset[str] = frozenset({".fits", ".fit", ".fts"})

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[FrameEntry] = []
        self._sort_col: int = -1
        self._sort_asc: bool = True
        self._filter_text: str = ""
        self._setup_ui()
        self.setAcceptDrops(True)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._group = QGroupBox("Light-Frames")
        group_layout = QVBoxLayout(self._group)

        self._count_label = QLabel("Keine Frames geladen")
        self._count_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self._count_label)

        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText("Frames filtern…")
        self._filter_input.setClearButtonEnabled(True)
        self._filter_input.setToolTip("Filtert Frames nach Dateiname (Groß/Kleinschreibung wird ignoriert)")
        self._filter_input.textChanged.connect(self._on_filter_changed)
        group_layout.addWidget(self._filter_input)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Min. Qualität:"))
        self._quality_spinbox = QDoubleSpinBox()
        self._quality_spinbox.setRange(0.0, 100.0)
        self._quality_spinbox.setSingleStep(5.0)
        self._quality_spinbox.setSuffix(" %")
        self._quality_spinbox.setValue(0.0)
        self._quality_spinbox.setToolTip(
            "Frames unterhalb dieser Qualität werden abgewählt"
        )
        threshold_row.addWidget(self._quality_spinbox)
        self._threshold_btn = QPushButton("Anwenden")
        self._threshold_btn.setToolTip(
            "Alle Frames unterhalb der Qualitätsschwelle abwählen"
        )
        self._threshold_btn.clicked.connect(self._on_apply_threshold)
        threshold_row.addWidget(self._threshold_btn)
        group_layout.addLayout(threshold_row)

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
            display_name = Path(entry.path).name
            if entry.notes:
                display_name = f"* {display_name}"
            name_item = QTableWidgetItem(display_name)
            tooltip = entry.path
            if entry.notes:
                tooltip = f"{entry.path}\n\nNotiz: {entry.notes}"
            name_item.setToolTip(tooltip)
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

        self._apply_filter()
        self._refresh_count_label()

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_filter_changed(self, text: str) -> None:
        self._filter_text = text
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Show/hide rows based on current filter text (matched against filename)."""
        pattern = self._filter_text.lower()
        for row, entry in enumerate(self._entries):
            visible = not pattern or pattern in Path(entry.path).name.lower()
            self._table.setRowHidden(row, not visible)

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
        preview_act = menu.addAction("Vorschau anzeigen")
        preview_act.setEnabled(len(selected_rows) == 1)
        notes_act = menu.addAction("Notiz bearbeiten…")
        notes_act.setEnabled(len(selected_rows) == 1)
        menu.addSeparator()
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

        action = self._exec_menu(menu, pos)
        if action == preview_act and len(selected_rows) == 1:
            entry = self._entries[selected_rows[0]]
            self.preview_requested.emit(entry.path)
        elif action == notes_act and len(selected_rows) == 1:
            self._edit_notes(selected_rows[0])
        elif action == select_all_act:
            self._set_all_selected(True)
        elif action == deselect_all_act:
            self._set_all_selected(False)
        elif action == invert_act:
            self._invert_selection()
        elif action == remove_act and selected_rows:
            self.remove_requested.emit(selected_rows)

    def _exec_menu(self, menu: QMenu, pos: QPoint) -> QAction | None:
        return menu.exec(self._table.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Frame notes
    # ------------------------------------------------------------------

    def _edit_notes(self, row: int) -> None:
        if row < 0 or row >= len(self._entries):
            return
        entry = self._entries[row]
        text, ok = QInputDialog.getText(
            self,
            "Notiz bearbeiten",
            f"Notiz für {Path(entry.path).name}:",
            text=entry.notes,
        )
        if ok:
            entry.notes = text.strip()
            self._repopulate_table()

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
    # Quality threshold
    # ------------------------------------------------------------------

    @Slot()
    def _on_apply_threshold(self) -> None:
        self.apply_quality_threshold(self._quality_spinbox.value())

    def apply_quality_threshold(self, min_pct: float) -> None:
        """Deselect frames whose quality_score < min_pct/100. Unscored frames unchanged."""
        threshold = min_pct / 100.0
        for i, entry in enumerate(self._entries):
            if entry.quality_score is None:
                continue
            new_selected = entry.quality_score >= threshold
            if entry.selected != new_selected:
                entry.selected = new_selected
                item = self._table.item(i, _COL_SEL)
                if item is not None:
                    item.setText("✓" if new_selected else "✗")
                self.selection_changed.emit(i, new_selected)
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

    # ------------------------------------------------------------------
    # Drag & drop support
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            fits_urls = [
                u for u in event.mimeData().urls()
                if Path(u.toLocalFile()).suffix.lower() in self._FITS_SUFFIXES
            ]
            if fits_urls:
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [
            u.toLocalFile()
            for u in event.mimeData().urls()
            if Path(u.toLocalFile()).suffix.lower() in self._FITS_SUFFIXES
        ]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()
