"""Photometric color calibration configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["ColorCalibrationPanel"]

_CATALOG_OPTIONS = [
    ("GAIA DR3", "gaia_dr3"),
    ("2MASS", "2mass"),
]


class ColorCalibrationPanel(QWidget):
    """Panel for configuring spectrophotometric color calibration."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._enabled_cb = QCheckBox("Farbkalibrierung aktivieren")
        self._enabled_cb.setAccessibleName("Photometrische Farbkalibrierung aktivieren")
        layout.addWidget(self._enabled_cb)

        self._settings_group = QGroupBox("Einstellungen")
        group_layout = QVBoxLayout(self._settings_group)

        catalog_row = QHBoxLayout()
        catalog_label = QLabel("Katalog:")
        catalog_label.setMinimumWidth(100)
        self._catalog_combo = QComboBox()
        for display, value in _CATALOG_OPTIONS:
            self._catalog_combo.addItem(display, value)
        self._catalog_combo.setAccessibleName("Sternkatalog fuer Referenzfarben")
        catalog_row.addWidget(catalog_label)
        catalog_row.addWidget(self._catalog_combo, stretch=1)
        group_layout.addLayout(catalog_row)

        radius_row = QHBoxLayout()
        radius_label = QLabel("Sampling-Radius:")
        radius_label.setMinimumWidth(100)
        self._radius_spin = QSpinBox()
        self._radius_spin.setRange(3, 20)
        self._radius_spin.setSuffix(" px")
        self._radius_spin.setValue(8)
        self._radius_spin.setAccessibleName("Apertur-Radius fuer Stern-Sampling in Pixeln")
        radius_row.addWidget(radius_label)
        radius_row.addWidget(self._radius_spin, stretch=1)
        group_layout.addLayout(radius_row)

        self._info_label = QLabel("Benötigt WCS-Lösung aus Plate Solving (F-1).")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self._info_label)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._enabled_cb.toggled.connect(self._on_enabled_changed)
        self._catalog_combo.currentIndexChanged.connect(self._on_catalog_changed)
        self._radius_spin.valueChanged.connect(self._on_radius_changed)
        self._model.color_calibration_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._enabled_cb.blockSignals(True)
        self._enabled_cb.setChecked(self._model.color_calibration_enabled)
        self._enabled_cb.blockSignals(False)

        self._catalog_combo.blockSignals(True)
        idx = self._catalog_combo.findData(self._model.color_calibration_catalog)
        if idx >= 0:
            self._catalog_combo.setCurrentIndex(idx)
        self._catalog_combo.blockSignals(False)

        self._radius_spin.blockSignals(True)
        self._radius_spin.setValue(self._model.color_calibration_sample_radius)
        self._radius_spin.blockSignals(False)

        self._settings_group.setEnabled(self._model.color_calibration_enabled)

    @Slot(bool)
    def _on_enabled_changed(self, checked: bool) -> None:
        self._model.color_calibration_enabled = checked
        self._settings_group.setEnabled(checked)

    @Slot(int)
    def _on_catalog_changed(self, index: int) -> None:
        value = self._catalog_combo.itemData(index)
        if value:
            self._model.color_calibration_catalog = value

    @Slot(int)
    def _on_radius_changed(self, value: int) -> None:
        self._model.color_calibration_sample_radius = value
