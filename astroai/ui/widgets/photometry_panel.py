"""Photometry results panel: star table, calibration summary, and export."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from astroai.engine.photometry.models import PhotometryResult

__all__ = ["PhotometryPanel"]

_COLUMNS = ["ID", "RA", "Dec", "Instr. Mag", "Cal. Mag", "Kat. Mag", "Residual"]


class PhotometryPanel(QWidget):
    """Dock-panel showing photometry results with export capabilities."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Photometrie-Ergebnisse")
        self._result: PhotometryResult | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Photometrie")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        self._summary_label = QLabel("Keine Photometrie-Daten")
        self._summary_label.setObjectName("photometrySummaryLabel")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._table)

        btn_layout = QHBoxLayout()
        self._csv_btn = QPushButton("CSV exportieren")
        self._csv_btn.setAccessibleName("Photometrie-Ergebnisse als CSV exportieren")
        self._csv_btn.setEnabled(False)
        self._csv_btn.clicked.connect(self._export_csv)
        btn_layout.addWidget(self._csv_btn)

        self._fits_btn = QPushButton("FITS exportieren")
        self._fits_btn.setAccessibleName("Photometrie-Ergebnisse als FITS exportieren")
        self._fits_btn.setEnabled(False)
        self._fits_btn.clicked.connect(self._export_fits)
        btn_layout.addWidget(self._fits_btn)

        layout.addLayout(btn_layout)

    def set_result(self, result: PhotometryResult | None) -> None:
        """Populate the panel with a PhotometryResult."""
        self._result = result
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        if result is None or not result.stars:
            self._summary_label.setText("Keine Photometrie-Daten")
            self._csv_btn.setEnabled(False)
            self._fits_btn.setEnabled(False)
            self._table.setSortingEnabled(True)
            return

        self._summary_label.setText(
            f"Abgeglichene Sterne: {result.n_matched}   |   "
            f"Kalibrierung R²: {result.r_squared:.4f}"
        )

        self._table.setRowCount(len(result.stars))
        for row, star in enumerate(result.stars):
            self._table.setItem(row, 0, _num_item(star.star_id))
            self._table.setItem(row, 1, _num_item(round(star.ra, 6)))
            self._table.setItem(row, 2, _num_item(round(star.dec, 6)))
            self._table.setItem(row, 3, _num_item(round(star.instr_mag, 4)))
            self._table.setItem(row, 4, _num_item(round(star.cal_mag, 4)))
            self._table.setItem(row, 5, _num_item(round(star.catalog_mag, 4)))
            self._table.setItem(row, 6, _num_item(round(star.residual, 4)))

        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)
        self._csv_btn.setEnabled(True)
        self._fits_btn.setEnabled(True)

    def clear(self) -> None:
        self.set_result(None)

    @Slot()
    def _export_csv(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV exportieren", "", "CSV-Dateien (*.csv)"
        )
        if not path:
            return
        from astroai.engine.photometry.export import PhotometryExporter
        PhotometryExporter().to_csv(self._result, Path(path))

    @Slot()
    def _export_fits(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "FITS exportieren", "", "FITS-Dateien (*.fits *.fit)"
        )
        if not path:
            return
        from astroai.engine.photometry.export import PhotometryExporter
        PhotometryExporter().to_fits(self._result, Path(path))


def _num_item(value: int | float) -> QTableWidgetItem:
    item = QTableWidgetItem()
    item.setData(Qt.ItemDataRole.DisplayRole, value)
    return item
