"""Export configuration panel — output directory, filename, and format."""
from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["ExportPanel"]

_FORMAT_LABELS: list[str] = ["FITS", "TIFF32", "XISF"]
_FORMAT_VALUES: list[str] = ["fits", "tiff", "xisf"]


class ExportPanel(QWidget):
    """Panel for configuring pipeline output path and file format."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._group = QGroupBox("Export-Einstellungen")
        group_layout = QVBoxLayout(self._group)

        dir_row = QHBoxLayout()
        dir_label = QLabel("Ausgabeverzeichnis:")
        dir_label.setMinimumWidth(150)
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Kein Verzeichnis ausgewählt…")
        self._dir_edit.setAccessibleName("Ausgabeverzeichnis fuer exportierte Bilder")
        self._browse_btn = QPushButton("Durchsuchen…")
        self._browse_btn.setAccessibleName("Verzeichnis auswaehlen Dialog oeffnen")
        dir_row.addWidget(dir_label)
        dir_row.addWidget(self._dir_edit, stretch=1)
        dir_row.addWidget(self._browse_btn)
        group_layout.addLayout(dir_row)

        name_row = QHBoxLayout()
        name_label = QLabel("Dateiname:")
        name_label.setMinimumWidth(150)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("output")
        self._name_edit.setAccessibleName("Ausgabedateiname ohne Dateiendung")
        name_row.addWidget(name_label)
        name_row.addWidget(self._name_edit, stretch=1)
        group_layout.addLayout(name_row)

        fmt_row = QHBoxLayout()
        fmt_label = QLabel("Format:")
        fmt_label.setMinimumWidth(150)
        self._fmt_combo = QComboBox()
        for label in _FORMAT_LABELS:
            self._fmt_combo.addItem(label)
        self._fmt_combo.setAccessibleName("Ausgabeformat FITS TIFF oder XISF")
        fmt_row.addWidget(fmt_label)
        fmt_row.addWidget(self._fmt_combo, stretch=1)
        group_layout.addLayout(fmt_row)

        self._info_label = QLabel(
            "Ausgabe wird automatisch nach Abschluss\n"
            "des Stack-&-Process-Laufs gespeichert."
        )
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self._info_label)

        layout.addWidget(self._group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._browse_btn.clicked.connect(self._on_browse)
        self._dir_edit.editingFinished.connect(self._on_dir_edited)
        self._name_edit.editingFinished.connect(self._on_name_edited)
        self._fmt_combo.currentIndexChanged.connect(self._on_format_changed)
        self._model.export_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._dir_edit.blockSignals(True)
        self._dir_edit.setText(self._model.output_path)
        self._dir_edit.blockSignals(False)

        self._name_edit.blockSignals(True)
        self._name_edit.setText(self._model.output_filename)
        self._name_edit.blockSignals(False)

        self._fmt_combo.blockSignals(True)
        fmt = self._model.output_format
        idx = _FORMAT_VALUES.index(fmt) if fmt in _FORMAT_VALUES else 0
        self._fmt_combo.setCurrentIndex(idx)
        self._fmt_combo.blockSignals(False)

    @Slot()
    def _on_browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Ausgabeverzeichnis wählen",
            self._model.output_path or "",
        )
        if directory:
            self._model.output_path = directory

    @Slot()
    def _on_dir_edited(self) -> None:
        self._model.output_path = self._dir_edit.text().strip()

    @Slot()
    def _on_name_edited(self) -> None:
        name = self._name_edit.text().strip() or "output"
        self._model.output_filename = name

    @Slot(int)
    def _on_format_changed(self, index: int) -> None:
        if 0 <= index < len(_FORMAT_VALUES):
            self._model.output_format = _FORMAT_VALUES[index]
