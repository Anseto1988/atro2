"""Stacking configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["StackingPanel"]

_METHODS = ["mean", "median", "sigma_clip"]


class StackingPanel(QWidget):
    """Panel for configuring the frame stacking step."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Frame-Stacking")
        form = QFormLayout(group)

        self._method_combo = QComboBox()
        self._method_combo.addItems(_METHODS)
        self._method_combo.setAccessibleName("Stacking-Methode mean median sigma_clip")
        form.addRow(QLabel("Methode:"), self._method_combo)

        self._sigma_low_spin = QDoubleSpinBox()
        self._sigma_low_spin.setRange(0.0, 10.0)
        self._sigma_low_spin.setSingleStep(0.1)
        self._sigma_low_spin.setDecimals(2)
        self._sigma_low_spin.setAccessibleName("Sigma-Clipping untere Grenze")
        form.addRow(QLabel("Sigma unten:"), self._sigma_low_spin)

        self._sigma_high_spin = QDoubleSpinBox()
        self._sigma_high_spin.setRange(0.0, 10.0)
        self._sigma_high_spin.setSingleStep(0.1)
        self._sigma_high_spin.setDecimals(2)
        self._sigma_high_spin.setAccessibleName("Sigma-Clipping obere Grenze")
        form.addRow(QLabel("Sigma oben:"), self._sigma_high_spin)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        self._sigma_low_spin.valueChanged.connect(self._on_sigma_low_changed)
        self._sigma_high_spin.valueChanged.connect(self._on_sigma_high_changed)
        self._model.stacking_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        method = self._model.stacking_method
        idx = self._method_combo.findText(method)
        self._method_combo.blockSignals(True)
        self._method_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._method_combo.blockSignals(False)

        self._sigma_low_spin.blockSignals(True)
        self._sigma_low_spin.setValue(self._model.stacking_sigma_low)
        self._sigma_low_spin.blockSignals(False)

        self._sigma_high_spin.blockSignals(True)
        self._sigma_high_spin.setValue(self._model.stacking_sigma_high)
        self._sigma_high_spin.blockSignals(False)

        is_sigma = method == "sigma_clip"
        self._sigma_low_spin.setEnabled(is_sigma)
        self._sigma_high_spin.setEnabled(is_sigma)

    @Slot(str)
    def _on_method_changed(self, value: str) -> None:
        self._model.stacking_method = value
        is_sigma = value == "sigma_clip"
        self._sigma_low_spin.setEnabled(is_sigma)
        self._sigma_high_spin.setEnabled(is_sigma)

    @Slot(float)
    def _on_sigma_low_changed(self, value: float) -> None:
        self._model.stacking_sigma_low = value

    @Slot(float)
    def _on_sigma_high_changed(self, value: float) -> None:
        self._model.stacking_sigma_high = value
