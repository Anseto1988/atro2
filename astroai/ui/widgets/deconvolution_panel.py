"""Deconvolution configuration panel for Lucy-Richardson / ONNX sharpness reconstruction."""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
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

__all__ = ["DeconvolutionPanel"]


class DeconvolutionPanel(QWidget):
    """Panel for configuring the deconvolution processing step."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._enabled_cb = QCheckBox("Deconvolution aktivieren")
        self._enabled_cb.setAccessibleName("Deconvolution-Verarbeitung aktivieren")
        layout.addWidget(self._enabled_cb)

        self._settings_group = QGroupBox("Einstellungen")
        group_layout = QVBoxLayout(self._settings_group)

        iter_row = QHBoxLayout()
        iter_label = QLabel("Iterationen:")
        iter_label.setMinimumWidth(90)
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 100)
        self._iter_spin.setValue(10)
        self._iter_spin.setAccessibleName("Anzahl Lucy-Richardson-Iterationen")
        iter_row.addWidget(iter_label)
        iter_row.addWidget(self._iter_spin, stretch=1)
        group_layout.addLayout(iter_row)

        sigma_row = QHBoxLayout()
        sigma_label = QLabel("PSF Sigma:")
        sigma_label.setMinimumWidth(90)
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.1, 10.0)
        self._sigma_spin.setSingleStep(0.1)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setValue(1.0)
        self._sigma_spin.setAccessibleName("Gausssche PSF Sigma")
        sigma_row.addWidget(sigma_label)
        sigma_row.addWidget(self._sigma_spin, stretch=1)
        group_layout.addLayout(sigma_row)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._enabled_cb.toggled.connect(self._on_enabled_changed)
        self._iter_spin.valueChanged.connect(self._on_iterations_changed)
        self._sigma_spin.valueChanged.connect(self._on_sigma_changed)
        self._model.deconvolution_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._enabled_cb.blockSignals(True)
        self._enabled_cb.setChecked(self._model.deconvolution_enabled)
        self._enabled_cb.blockSignals(False)

        self._iter_spin.blockSignals(True)
        self._iter_spin.setValue(self._model.deconvolution_iterations)
        self._iter_spin.blockSignals(False)

        self._sigma_spin.blockSignals(True)
        self._sigma_spin.setValue(self._model.deconvolution_psf_sigma)
        self._sigma_spin.blockSignals(False)

        self._settings_group.setEnabled(self._model.deconvolution_enabled)

    @Slot(bool)
    def _on_enabled_changed(self, checked: bool) -> None:
        self._model.deconvolution_enabled = checked
        self._settings_group.setEnabled(checked)

    @Slot(int)
    def _on_iterations_changed(self, value: int) -> None:
        self._model.deconvolution_iterations = value

    @Slot(float)
    def _on_sigma_changed(self, value: float) -> None:
        self._model.deconvolution_psf_sigma = value
