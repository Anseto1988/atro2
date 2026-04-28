"""Background gradient removal configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Signal, Slot
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

__all__ = ["BackgroundRemovalPanel"]

_METHODS = [("rbf", "RBF (Radial Basis Function)"), ("poly", "Polynom")]


class BackgroundRemovalPanel(QWidget):
    """Panel for configuring background gradient removal."""

    PREVIEW_STEP = "background_removal"
    preview_requested = Signal(dict)

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._enabled_cb = QCheckBox("Hintergrundentfernung aktivieren")
        self._enabled_cb.setAccessibleName("Hintergrundgradient-Entfernung aktivieren")
        layout.addWidget(self._enabled_cb)

        self._settings_group = QGroupBox("Einstellungen")
        group_layout = QVBoxLayout(self._settings_group)

        method_row = QHBoxLayout()
        method_label = QLabel("Methode:")
        method_label.setMinimumWidth(130)
        self._method_combo = QComboBox()
        for key, label in _METHODS:
            self._method_combo.addItem(label, key)
        self._method_combo.setAccessibleName("Interpolationsmethode fuer Hintergrundmodell")
        method_row.addWidget(method_label)
        method_row.addWidget(self._method_combo, stretch=1)
        group_layout.addLayout(method_row)

        tile_row = QHBoxLayout()
        tile_label = QLabel("Kachel-Größe:")
        tile_label.setMinimumWidth(130)
        self._tile_spin = QSpinBox()
        self._tile_spin.setRange(16, 256)
        self._tile_spin.setSingleStep(16)
        self._tile_spin.setSuffix(" px")
        self._tile_spin.setValue(64)
        self._tile_spin.setAccessibleName("Kachelgroesse fuer Hintergrundmodellierung")
        tile_row.addWidget(tile_label)
        tile_row.addWidget(self._tile_spin, stretch=1)
        group_layout.addLayout(tile_row)

        self._preserve_median_cb = QCheckBox("Median erhalten")
        self._preserve_median_cb.setChecked(True)
        self._preserve_median_cb.setAccessibleName("Originalen Median-Helligkeitswert nach Entfernung erhalten")
        group_layout.addWidget(self._preserve_median_cb)

        self._info_label = QLabel(
            "Entfernt Helligkeitsgradienten durch Lichtver-\n"
            "schmutzung oder optische Vignettierung."
        )
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self._info_label)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._enabled_cb.toggled.connect(self._on_enabled_changed)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._tile_spin.valueChanged.connect(self._on_tile_size_changed)
        self._preserve_median_cb.toggled.connect(self._on_preserve_median_changed)
        self._model.background_removal_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._enabled_cb.blockSignals(True)
        self._enabled_cb.setChecked(self._model.background_removal_enabled)
        self._enabled_cb.blockSignals(False)

        method_key = self._model.background_removal_method
        for i, (key, _) in enumerate(_METHODS):
            if key == method_key:
                self._method_combo.blockSignals(True)
                self._method_combo.setCurrentIndex(i)
                self._method_combo.blockSignals(False)
                break

        self._tile_spin.blockSignals(True)
        self._tile_spin.setValue(self._model.background_removal_tile_size)
        self._tile_spin.blockSignals(False)

        self._preserve_median_cb.blockSignals(True)
        self._preserve_median_cb.setChecked(self._model.background_removal_preserve_median)
        self._preserve_median_cb.blockSignals(False)

        self._settings_group.setEnabled(self._model.background_removal_enabled)

    def _emit_preview(self) -> None:
        self.preview_requested.emit({
            "tile_size": self._model.background_removal_tile_size,
            "method": self._model.background_removal_method,
            "preserve_median": self._model.background_removal_preserve_median,
        })

    @Slot(bool)
    def _on_enabled_changed(self, checked: bool) -> None:
        self._model.background_removal_enabled = checked
        self._settings_group.setEnabled(checked)
        self._emit_preview()

    @Slot(int)
    def _on_method_changed(self, index: int) -> None:
        key = self._method_combo.itemData(index)
        if key is not None:
            self._model.background_removal_method = str(key)
            self._emit_preview()

    @Slot(int)
    def _on_tile_size_changed(self, value: int) -> None:
        self._model.background_removal_tile_size = value
        self._emit_preview()

    @Slot(bool)
    def _on_preserve_median_changed(self, checked: bool) -> None:
        self._model.background_removal_preserve_median = checked
        self._emit_preview()
