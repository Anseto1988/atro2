"""Panel displaying FITS header metadata for the loaded image."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["FITSMetadataPanel"]

# Friendly German labels for common FITS keywords
_LABEL_MAP: dict[str, str] = {
    "OBJECT": "Ziel",
    "DATE-OBS": "Datum",
    "EXPTIME": "Belichtung (s)",
    "FILTER": "Filter",
    "TELESCOP": "Teleskop",
    "INSTRUME": "Kamera",
    "FOCALLEN": "Brennweite (mm)",
    "XPIXSZ": "Pixelgröße (µm)",
    "RA": "RA",
    "DEC": "Dec",
    "GAIN": "Gain",
    "CCD-TEMP": "Temperatur (°C)",
    "XBINNING": "Binning X",
    "YBINNING": "Binning Y",
    "NAXIS1": "Breite (px)",
    "NAXIS2": "Höhe (px)",
}


class FITSMetadataPanel(QWidget):
    """Shows selected FITS header key-value pairs for the currently loaded image."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._group = QGroupBox("FITS-Metadaten")
        group_layout = QVBoxLayout(self._group)

        self._placeholder = QLabel("Kein FITS-Bild geladen")
        self._placeholder.setStyleSheet("color: #888; font-size: 11px;")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group_layout.addWidget(self._placeholder)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Eigenschaft", "Wert"])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAccessibleName("FITS-Header-Metadaten")
        self._table.hide()
        group_layout.addWidget(self._table)

        layout.addWidget(self._group)

    def set_header(self, header: object) -> None:
        """Populate the table from a dict[str, str]; pass None to clear."""
        if not isinstance(header, dict) or not header:
            self._table.hide()
            self._placeholder.show()
            return

        rows: list[tuple[str, str]] = []
        for key, label in _LABEL_MAP.items():
            val = header.get(key)
            if val is not None:
                rows.append((label, str(val)))

        if not rows:
            self._table.hide()
            self._placeholder.show()
            return

        self._table.setRowCount(len(rows))
        for row, (label, value) in enumerate(rows):
            key_item = QTableWidgetItem(label)
            key_item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            val_item = QTableWidgetItem(value)
            val_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, 0, key_item)
            self._table.setItem(row, 1, val_item)

        self._table.resizeColumnsToContents()
        self._placeholder.hide()
        self._table.show()

    def clear(self) -> None:
        self._table.setRowCount(0)
        self._table.hide()
        self._placeholder.show()
