"""Registration configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["RegistrationPanel"]

_METHODS = [("Stern-Detektion (LoG)", "star"), ("Phasenkorrelation", "phase_correlation")]


class RegistrationPanel(QWidget):
    """Panel for configuring frame registration method and parameters."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Registrierungseinstellungen")
        group_layout = QVBoxLayout(group)

        method_row = QHBoxLayout()
        method_label = QLabel("Methode:")
        method_label.setMinimumWidth(150)
        self._method_combo = QComboBox()
        for label_text, value in _METHODS:
            self._method_combo.addItem(label_text, userData=value)
        self._method_combo.setAccessibleName("Registrierungsmethode (Stern-Detektion oder Phasenkorrelation)")
        method_row.addWidget(method_label)
        method_row.addWidget(self._method_combo, stretch=1)
        group_layout.addLayout(method_row)

        upsample_row = QHBoxLayout()
        upsample_label = QLabel("Upsample-Faktor:")
        upsample_label.setMinimumWidth(150)
        self._upsample_spin = QSpinBox()
        self._upsample_spin.setRange(1, 100)
        self._upsample_spin.setValue(10)
        self._upsample_spin.setAccessibleName("Upsample-Faktor fuer Phasenkorrelation (1-100)")
        upsample_row.addWidget(upsample_label)
        upsample_row.addWidget(self._upsample_spin, stretch=1)
        group_layout.addLayout(upsample_row)

        ref_row = QHBoxLayout()
        ref_label = QLabel("Referenz-Frame-Index:")
        ref_label.setMinimumWidth(150)
        self._ref_index_spin = QSpinBox()
        self._ref_index_spin.setRange(0, 9999)
        self._ref_index_spin.setValue(0)
        self._ref_index_spin.setAccessibleName("Index des Referenz-Frames fuer die Registrierung")
        ref_row.addWidget(ref_label)
        ref_row.addWidget(self._ref_index_spin, stretch=1)
        group_layout.addLayout(ref_row)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._upsample_spin.valueChanged.connect(self._on_upsample_changed)
        self._ref_index_spin.valueChanged.connect(self._on_ref_index_changed)
        self._model.registration_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._method_combo.blockSignals(True)
        method_values = [self._method_combo.itemData(i) for i in range(self._method_combo.count())]
        idx = method_values.index(self._model.registration_method) if self._model.registration_method in method_values else 0
        self._method_combo.setCurrentIndex(idx)
        self._method_combo.blockSignals(False)

        self._upsample_spin.blockSignals(True)
        self._upsample_spin.setValue(self._model.registration_upsample_factor)
        self._upsample_spin.blockSignals(False)

        self._ref_index_spin.blockSignals(True)
        self._ref_index_spin.setValue(self._model.registration_reference_frame_index)
        self._ref_index_spin.blockSignals(False)

    @Slot(int)
    def _on_method_changed(self, index: int) -> None:
        self._model.registration_method = self._method_combo.itemData(index)

    @Slot(int)
    def _on_upsample_changed(self, value: int) -> None:
        self._model.registration_upsample_factor = value

    @Slot(int)
    def _on_ref_index_changed(self, value: int) -> None:
        self._model.registration_reference_frame_index = value
