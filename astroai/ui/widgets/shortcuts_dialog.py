"""Keyboard shortcuts reference dialog."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["ShortcutsDialog"]

_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Projekt", [
        ("Neues Projekt", "Ctrl+N"),
        ("Projekt öffnen", "Ctrl+O"),
        ("Projekt speichern", "Ctrl+S"),
        ("Projekt speichern unter", "Ctrl+Shift+S"),
        ("Projekt schließen / Beenden", "Ctrl+Q"),
    ]),
    ("Frames & Kalibrierung", [
        ("Bild öffnen", "Ctrl+Shift+O"),
        ("Light-Frames importieren", "Ctrl+Shift+L"),
        ("Ordner importieren (rekursiv)", "Ctrl+Shift+F"),
        ("Kalibrierung automatisch zuordnen", "Ctrl+Shift+K"),
        ("Frames-Statistik exportieren", "Ctrl+Shift+E"),
    ]),
    ("Ansicht", [
        ("An Fenster anpassen", "F"),
        ("Auto-Stretch Vorschau", "Ctrl+Shift+A"),
        ("Vorher/Nachher Vergleich", "Ctrl+D"),
        ("Bild kopieren", "Ctrl+C"),
        ("Vorschau-Bild speichern", "Ctrl+Shift+P"),
    ]),
    ("Pipeline", [
        ("Kalibrierung ausführen", "Ctrl+R"),
        ("Stack & Process", "Ctrl+Shift+R"),
        ("Abbrechen", "Esc"),
    ]),
    ("Hilfe", [
        ("Projektübersicht", "Ctrl+I"),
        ("Tastaturkürzel", "Ctrl+?"),
    ]),
]


def _make_section(title: str, rows: list[tuple[str, str]], parent: QWidget) -> QGroupBox:
    box = QGroupBox(title, parent)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(4, 4, 4, 4)

    table = QTableWidget(len(rows), 2, box)
    table.setHorizontalHeaderLabels(["Funktion", "Kürzel"])
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setAlternatingRowColors(True)

    for i, (func, keys) in enumerate(rows):
        func_item = QTableWidgetItem(func)
        func_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        keys_item = QTableWidgetItem(keys)
        keys_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        keys_item.setData(
            Qt.ItemDataRole.TextAlignmentRole,
            int(Qt.AlignmentFlag.AlignCenter),
        )
        table.setItem(i, 0, func_item)
        table.setItem(i, 1, keys_item)

    table.resizeColumnsToContents()
    table.setMaximumHeight(table.verticalHeader().length() + table.horizontalHeader().height() + 4)
    layout.addWidget(table)
    return box


class ShortcutsDialog(QDialog):
    """Non-modal read-only keyboard shortcut reference."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tastaturkürzel")
        self.setMinimumWidth(420)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        for title, rows in _SECTIONS:
            layout.addWidget(_make_section(title, rows, self))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
