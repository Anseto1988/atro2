"""Star reduction panel widget — shrinks star sizes via morphological erosion."""
from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from astroai.processing.stars.star_reducer import StarReductionConfig
from astroai.ui.models import PipelineModel

__all__ = ["StarReductionPanel"]


class StarReductionPanel(QWidget):
    """Panel for star reduction (morphological erosion of star areas)."""

    star_reduction_changed = Signal(StarReductionConfig)

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Sternreduktion")
        group_layout = QVBoxLayout(group)

        # Amount
        row_amount = QHBoxLayout()
        lbl_amount = QLabel("Stärke:")
        lbl_amount.setMinimumWidth(80)
        self._spin_amount = QDoubleSpinBox()
        self._spin_amount.setRange(0.0, 1.0)
        self._spin_amount.setSingleStep(0.05)
        self._spin_amount.setDecimals(2)
        self._spin_amount.setValue(0.5)
        self._spin_amount.setAccessibleName("Sternreduktion Stärke")
        row_amount.addWidget(lbl_amount)
        row_amount.addWidget(self._spin_amount, stretch=1)
        group_layout.addLayout(row_amount)

        # Radius
        row_radius = QHBoxLayout()
        lbl_radius = QLabel("Radius:")
        lbl_radius.setMinimumWidth(80)
        self._spin_radius = QSpinBox()
        self._spin_radius.setRange(1, 10)
        self._spin_radius.setValue(2)
        self._spin_radius.setAccessibleName("Sternreduktion Radius")
        row_radius.addWidget(lbl_radius)
        row_radius.addWidget(self._spin_radius, stretch=1)
        group_layout.addLayout(row_radius)

        # Threshold
        row_threshold = QHBoxLayout()
        lbl_threshold = QLabel("Schwellwert:")
        lbl_threshold.setMinimumWidth(80)
        self._spin_threshold = QDoubleSpinBox()
        self._spin_threshold.setRange(0.0, 1.0)
        self._spin_threshold.setSingleStep(0.05)
        self._spin_threshold.setDecimals(2)
        self._spin_threshold.setValue(0.5)
        self._spin_threshold.setAccessibleName("Sternreduktion Schwellwert")
        row_threshold.addWidget(lbl_threshold)
        row_threshold.addWidget(self._spin_threshold, stretch=1)
        group_layout.addLayout(row_threshold)

        self._reset_btn = QPushButton("Zurücksetzen")
        group_layout.addWidget(self._reset_btn)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._spin_amount.valueChanged.connect(self._on_amount_changed)
        self._spin_radius.valueChanged.connect(self._on_radius_changed)
        self._spin_threshold.valueChanged.connect(self._on_threshold_changed)
        self._reset_btn.clicked.connect(self._on_reset)
        self._model.star_reduction_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._spin_amount.blockSignals(True)
        self._spin_amount.setValue(self._model.star_reduction_amount)
        self._spin_amount.blockSignals(False)

        self._spin_radius.blockSignals(True)
        self._spin_radius.setValue(self._model.star_reduction_radius)
        self._spin_radius.blockSignals(False)

        self._spin_threshold.blockSignals(True)
        self._spin_threshold.setValue(self._model.star_reduction_threshold)
        self._spin_threshold.blockSignals(False)

    @Slot(float)
    def _on_amount_changed(self, value: float) -> None:
        self._model.star_reduction_amount = value
        self.star_reduction_changed.emit(self._build_config())

    @Slot(int)
    def _on_radius_changed(self, value: int) -> None:
        self._model.star_reduction_radius = value
        self.star_reduction_changed.emit(self._build_config())

    @Slot(float)
    def _on_threshold_changed(self, value: float) -> None:
        self._model.star_reduction_threshold = value
        self.star_reduction_changed.emit(self._build_config())

    @Slot()
    def _on_reset(self) -> None:
        defaults = StarReductionConfig()
        self._model.star_reduction_amount = defaults.amount
        self._model.star_reduction_radius = defaults.radius
        self._model.star_reduction_threshold = defaults.threshold

    def _build_config(self) -> StarReductionConfig:
        return StarReductionConfig(
            amount=self._model.star_reduction_amount,
            radius=self._model.star_reduction_radius,
            threshold=self._model.star_reduction_threshold,
        )
