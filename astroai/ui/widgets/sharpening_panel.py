"""Unsharp mask sharpening configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["SharpeningPanel"]


class SharpeningPanel(QWidget):
    """Panel for configuring unsharp mask sharpening."""

    PREVIEW_STEP = "sharpening"
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

        self._settings_group = QGroupBox("Unschärfemaske (Schärfung)")
        group_layout = QVBoxLayout(self._settings_group)

        radius_row = QHBoxLayout()
        radius_label = QLabel("Radius (σ):")
        radius_label.setMinimumWidth(130)
        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.1, 10.0)
        self._radius_spin.setSingleStep(0.1)
        self._radius_spin.setDecimals(1)
        self._radius_spin.setSuffix(" px")
        self._radius_spin.setValue(1.0)
        self._radius_spin.setAccessibleName("Unscharfmasken-Radius in Pixeln")
        radius_row.addWidget(radius_label)
        radius_row.addWidget(self._radius_spin, stretch=1)
        group_layout.addLayout(radius_row)

        amount_row = QHBoxLayout()
        amount_label = QLabel("Stärke:")
        amount_label.setMinimumWidth(130)
        self._amount_spin = QDoubleSpinBox()
        self._amount_spin.setRange(0.0, 1.0)
        self._amount_spin.setSingleStep(0.05)
        self._amount_spin.setDecimals(2)
        self._amount_spin.setValue(0.5)
        self._amount_spin.setAccessibleName("Schaerfungsstaerke 0 bis 1")
        amount_row.addWidget(amount_label)
        amount_row.addWidget(self._amount_spin, stretch=1)
        group_layout.addLayout(amount_row)

        threshold_row = QHBoxLayout()
        threshold_label = QLabel("Schwellenwert:")
        threshold_label.setMinimumWidth(130)
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 0.5)
        self._threshold_spin.setSingleStep(0.005)
        self._threshold_spin.setDecimals(3)
        self._threshold_spin.setValue(0.02)
        self._threshold_spin.setAccessibleName("Rauschwellenwert 0 bis 0.5")
        threshold_row.addWidget(threshold_label)
        threshold_row.addWidget(self._threshold_spin, stretch=1)
        group_layout.addLayout(threshold_row)

        self._info_label = QLabel(
            "Schärft feine Details durch Subtraktion einer\n"
            "weichgezeichneten Kopie. Schwellenwert schützt\n"
            "flache Hintergrundbereiche vor Rauschverstärkung."
        )
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self._info_label)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._radius_spin.valueChanged.connect(self._on_radius_changed)
        self._amount_spin.valueChanged.connect(self._on_amount_changed)
        self._threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self._model.sharpening_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._radius_spin.blockSignals(True)
        self._radius_spin.setValue(self._model.sharpening_radius)
        self._radius_spin.blockSignals(False)

        self._amount_spin.blockSignals(True)
        self._amount_spin.setValue(self._model.sharpening_amount)
        self._amount_spin.blockSignals(False)

        self._threshold_spin.blockSignals(True)
        self._threshold_spin.setValue(self._model.sharpening_threshold)
        self._threshold_spin.blockSignals(False)

    def _emit_preview(self) -> None:
        self.preview_requested.emit({
            "radius": self._model.sharpening_radius,
            "amount": self._model.sharpening_amount,
            "threshold": self._model.sharpening_threshold,
        })

    @Slot(float)
    def _on_radius_changed(self, value: float) -> None:
        self._model.sharpening_radius = value
        self._emit_preview()

    @Slot(float)
    def _on_amount_changed(self, value: float) -> None:
        self._model.sharpening_amount = value
        self._emit_preview()

    @Slot(float)
    def _on_threshold_changed(self, value: float) -> None:
        self._model.sharpening_threshold = value
        self._emit_preview()
