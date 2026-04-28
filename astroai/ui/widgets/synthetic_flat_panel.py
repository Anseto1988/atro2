"""Synthetic flat frame generation configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["SyntheticFlatPanel"]


class SyntheticFlatPanel(QWidget):
    """Panel for configuring synthetic flat frame generation."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._enabled_cb = QCheckBox("Synthetisches Flat aktivieren")
        self._enabled_cb.setAccessibleName("Synthetische Flat-Generierung aktivieren")
        layout.addWidget(self._enabled_cb)

        self._settings_group = QGroupBox("Einstellungen")
        group_layout = QVBoxLayout(self._settings_group)

        tile_row = QHBoxLayout()
        tile_label = QLabel("Kachel-Größe:")
        tile_label.setMinimumWidth(130)
        self._tile_spin = QSpinBox()
        self._tile_spin.setRange(16, 256)
        self._tile_spin.setSingleStep(16)
        self._tile_spin.setSuffix(" px")
        self._tile_spin.setValue(64)
        self._tile_spin.setAccessibleName("Kachelgroesse fuer Hintergrundmodellierung in Pixeln")
        tile_row.addWidget(tile_label)
        tile_row.addWidget(self._tile_spin, stretch=1)
        group_layout.addLayout(tile_row)

        sigma_row = QHBoxLayout()
        sigma_label = QLabel("Glättungs-Sigma:")
        sigma_label.setMinimumWidth(130)
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.0, 50.0)
        self._sigma_spin.setSingleStep(1.0)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setSuffix(" px")
        self._sigma_spin.setValue(8.0)
        self._sigma_spin.setAccessibleName("Gauß-Glättungs-Sigma fuer synthetisches Flat")
        sigma_row.addWidget(sigma_label)
        sigma_row.addWidget(self._sigma_spin, stretch=1)
        group_layout.addLayout(sigma_row)

        self._info_label = QLabel(
            "Modelliert Vignettierung aus den Lightframes selbst.\n"
            "Verwenden, wenn keine echten Flat-Frames vorhanden."
        )
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self._info_label)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._enabled_cb.toggled.connect(self._on_enabled_changed)
        self._tile_spin.valueChanged.connect(self._on_tile_size_changed)
        self._sigma_spin.valueChanged.connect(self._on_sigma_changed)
        self._model.synthetic_flat_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._enabled_cb.blockSignals(True)
        self._enabled_cb.setChecked(self._model.synthetic_flat_enabled)
        self._enabled_cb.blockSignals(False)

        self._tile_spin.blockSignals(True)
        self._tile_spin.setValue(self._model.synthetic_flat_tile_size)
        self._tile_spin.blockSignals(False)

        self._sigma_spin.blockSignals(True)
        self._sigma_spin.setValue(self._model.synthetic_flat_smoothing_sigma)
        self._sigma_spin.blockSignals(False)

        self._settings_group.setEnabled(self._model.synthetic_flat_enabled)

    @Slot(bool)
    def _on_enabled_changed(self, checked: bool) -> None:
        self._model.synthetic_flat_enabled = checked
        self._settings_group.setEnabled(checked)

    @Slot(int)
    def _on_tile_size_changed(self, value: int) -> None:
        self._model.synthetic_flat_tile_size = value

    @Slot(float)
    def _on_sigma_changed(self, value: float) -> None:
        self._model.synthetic_flat_smoothing_sigma = value
