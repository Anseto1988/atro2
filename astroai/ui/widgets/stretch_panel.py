"""Intelligent stretch configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["StretchPanel"]


class StretchPanel(QWidget):
    """Panel for configuring the intelligent histogram stretch step."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._settings_group = QGroupBox("Intelligente Histogramm-Streckung")
        group_layout = QVBoxLayout(self._settings_group)

        bg_row = QHBoxLayout()
        bg_label = QLabel("Zielhintergrund:")
        bg_label.setMinimumWidth(150)
        self._bg_spin = QDoubleSpinBox()
        self._bg_spin.setRange(0.0, 1.0)
        self._bg_spin.setSingleStep(0.01)
        self._bg_spin.setDecimals(3)
        self._bg_spin.setValue(0.25)
        self._bg_spin.setAccessibleName("Ziel-Hintergrundhelligkeit nach Streckung 0 bis 1")
        bg_row.addWidget(bg_label)
        bg_row.addWidget(self._bg_spin, stretch=1)
        group_layout.addLayout(bg_row)

        sigma_row = QHBoxLayout()
        sigma_label = QLabel("Schatten-Sigma:")
        sigma_label.setMinimumWidth(150)
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(-10.0, 0.0)
        self._sigma_spin.setSingleStep(0.1)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setValue(-2.8)
        self._sigma_spin.setAccessibleName("Schatten-Clipping in Sigma-Einheiten negativ")
        sigma_row.addWidget(sigma_label)
        sigma_row.addWidget(self._sigma_spin, stretch=1)
        group_layout.addLayout(sigma_row)

        self._linked_cb = QCheckBox("Kanäle verknüpfen")
        self._linked_cb.setChecked(True)
        self._linked_cb.setAccessibleName("RGB-Kanaele fuer einheitliche Streckung verknuepfen")
        group_layout.addWidget(self._linked_cb)

        self._info_label = QLabel(
            "Streckt das Histogramm automatisch. Niedrigeres\n"
            "Schatten-Sigma erhält mehr dunkle Details."
        )
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self._info_label)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._bg_spin.valueChanged.connect(self._on_bg_changed)
        self._sigma_spin.valueChanged.connect(self._on_sigma_changed)
        self._linked_cb.toggled.connect(self._on_linked_changed)
        self._model.stretch_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._bg_spin.blockSignals(True)
        self._bg_spin.setValue(self._model.stretch_target_background)
        self._bg_spin.blockSignals(False)

        self._sigma_spin.blockSignals(True)
        self._sigma_spin.setValue(self._model.stretch_shadow_clipping_sigmas)
        self._sigma_spin.blockSignals(False)

        self._linked_cb.blockSignals(True)
        self._linked_cb.setChecked(self._model.stretch_linked_channels)
        self._linked_cb.blockSignals(False)

    @Slot(float)
    def _on_bg_changed(self, value: float) -> None:
        self._model.stretch_target_background = value

    @Slot(float)
    def _on_sigma_changed(self, value: float) -> None:
        self._model.stretch_shadow_clipping_sigmas = value

    @Slot(bool)
    def _on_linked_changed(self, checked: bool) -> None:
        self._model.stretch_linked_channels = checked
