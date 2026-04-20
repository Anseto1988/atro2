"""Starless configuration panel for AI-based star removal settings."""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["StarlessPanel"]

_FORMAT_OPTIONS = [
    ("xisf", "XISF"),
    ("tiff", "TIFF 32-bit"),
    ("fits", "FITS"),
]


class StarlessPanel(QWidget):
    """Panel for configuring the starless processing step."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._enabled_cb = QCheckBox("Starless aktivieren")
        self._enabled_cb.setAccessibleName("Starless-Verarbeitung aktivieren")
        layout.addWidget(self._enabled_cb)

        self._settings_group = QGroupBox("Einstellungen")
        group_layout = QVBoxLayout(self._settings_group)

        strength_row = QHBoxLayout()
        strength_label = QLabel("Staerke:")
        strength_label.setMinimumWidth(80)
        self._strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._strength_slider.setRange(0, 100)
        self._strength_slider.setValue(100)
        self._strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._strength_slider.setTickInterval(10)
        self._strength_slider.setAccessibleName("Starless-Staerke")
        self._strength_value = QLabel("100%")
        self._strength_value.setMinimumWidth(40)
        self._strength_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        strength_row.addWidget(strength_label)
        strength_row.addWidget(self._strength_slider, stretch=1)
        strength_row.addWidget(self._strength_value)
        group_layout.addLayout(strength_row)

        format_row = QHBoxLayout()
        format_label = QLabel("Ausgabeformat:")
        format_label.setMinimumWidth(80)
        self._format_combo = QComboBox()
        self._format_combo.setAccessibleName("Starless-Ausgabeformat")
        for key, display in _FORMAT_OPTIONS:
            self._format_combo.addItem(display, key)
        format_row.addWidget(format_label)
        format_row.addWidget(self._format_combo, stretch=1)
        group_layout.addLayout(format_row)

        self._mask_cb = QCheckBox("Sternmaske speichern")
        self._mask_cb.setAccessibleName("Sternmaske als separate Datei speichern")
        group_layout.addWidget(self._mask_cb)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._enabled_cb.toggled.connect(self._on_enabled_changed)
        self._strength_slider.valueChanged.connect(self._on_strength_changed)
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        self._mask_cb.toggled.connect(self._on_mask_changed)
        self._model.starless_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._enabled_cb.blockSignals(True)
        self._enabled_cb.setChecked(self._model.starless_enabled)
        self._enabled_cb.blockSignals(False)

        self._strength_slider.blockSignals(True)
        self._strength_slider.setValue(int(self._model.starless_strength * 100))
        self._strength_slider.blockSignals(False)
        self._strength_value.setText(f"{int(self._model.starless_strength * 100)}%")

        idx = self._format_combo.findData(self._model.starless_format)
        if idx >= 0:
            self._format_combo.blockSignals(True)
            self._format_combo.setCurrentIndex(idx)
            self._format_combo.blockSignals(False)

        self._mask_cb.blockSignals(True)
        self._mask_cb.setChecked(self._model.save_star_mask)
        self._mask_cb.blockSignals(False)

        self._settings_group.setEnabled(self._model.starless_enabled)

    @Slot(bool)
    def _on_enabled_changed(self, checked: bool) -> None:
        self._model.starless_enabled = checked
        self._settings_group.setEnabled(checked)

    @Slot(int)
    def _on_strength_changed(self, value: int) -> None:
        self._strength_value.setText(f"{value}%")
        self._model.starless_strength = value / 100.0

    @Slot(int)
    def _on_format_changed(self, _index: int) -> None:
        fmt = self._format_combo.currentData()
        if fmt:
            self._model.starless_format = fmt

    @Slot(bool)
    def _on_mask_changed(self, checked: bool) -> None:
        self._model.save_star_mask = checked
