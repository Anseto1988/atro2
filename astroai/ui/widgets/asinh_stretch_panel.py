"""Arcsinh Stretch panel widget for astrophotography stretch control."""
from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from astroai.processing.stretch.asinh_stretcher import AsinHConfig
from astroai.ui.models import PipelineModel

__all__ = ["AsinHStretchPanel"]


class AsinHStretchPanel(QWidget):
    """Panel for arcsinh stretch parameter adjustment."""

    asinh_stretch_changed = Signal(AsinHConfig)

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Arcsinh Stretch")
        group_layout = QVBoxLayout(group)

        # Stretch factor row
        row_sf = QHBoxLayout()
        lbl_sf = QLabel("Stretch-Faktor:")
        lbl_sf.setMinimumWidth(120)
        self._stretch_factor_spin = QDoubleSpinBox()
        self._stretch_factor_spin.setRange(0.001, 1000.0)
        self._stretch_factor_spin.setSingleStep(0.1)
        self._stretch_factor_spin.setDecimals(3)
        self._stretch_factor_spin.setValue(1.0)
        self._stretch_factor_spin.setAccessibleName("Arcsinh Stretch-Faktor")
        row_sf.addWidget(lbl_sf)
        row_sf.addWidget(self._stretch_factor_spin, stretch=1)
        group_layout.addLayout(row_sf)

        # Black point row
        row_bp = QHBoxLayout()
        lbl_bp = QLabel("Schwarzpunkt:")
        lbl_bp.setMinimumWidth(120)
        self._black_point_spin = QDoubleSpinBox()
        self._black_point_spin.setRange(0.0, 0.5)
        self._black_point_spin.setSingleStep(0.001)
        self._black_point_spin.setDecimals(3)
        self._black_point_spin.setValue(0.0)
        self._black_point_spin.setAccessibleName("Arcsinh Schwarzpunkt")
        row_bp.addWidget(lbl_bp)
        row_bp.addWidget(self._black_point_spin, stretch=1)
        group_layout.addLayout(row_bp)

        # Linked channels checkbox
        self._linked_cb = QCheckBox("Kanäle verknüpft")
        self._linked_cb.setChecked(True)
        group_layout.addWidget(self._linked_cb)

        # Reset button
        self._reset_btn = QPushButton("Zurücksetzen")
        group_layout.addWidget(self._reset_btn)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._stretch_factor_spin.valueChanged.connect(self._on_stretch_factor_changed)
        self._black_point_spin.valueChanged.connect(self._on_black_point_changed)
        self._linked_cb.toggled.connect(self._on_linked_changed)
        self._reset_btn.clicked.connect(self._on_reset)
        self._model.asinh_stretch_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._stretch_factor_spin.blockSignals(True)
        self._black_point_spin.blockSignals(True)
        self._linked_cb.blockSignals(True)

        self._stretch_factor_spin.setValue(self._model.asinh_stretch_factor)
        self._black_point_spin.setValue(self._model.asinh_black_point)
        self._linked_cb.setChecked(self._model.asinh_linked)

        self._stretch_factor_spin.blockSignals(False)
        self._black_point_spin.blockSignals(False)
        self._linked_cb.blockSignals(False)

    @Slot(float)
    def _on_stretch_factor_changed(self, value: float) -> None:
        self._model.asinh_stretch_factor = value
        self.asinh_stretch_changed.emit(self._build_config())

    @Slot(float)
    def _on_black_point_changed(self, value: float) -> None:
        self._model.asinh_black_point = value
        self.asinh_stretch_changed.emit(self._build_config())

    @Slot(bool)
    def _on_linked_changed(self, checked: bool) -> None:
        self._model.asinh_linked = checked
        self.asinh_stretch_changed.emit(self._build_config())

    @Slot()
    def _on_reset(self) -> None:
        self._model.asinh_stretch_factor = 1.0
        self._model.asinh_black_point = 0.0
        self._model.asinh_linked = True

    def _build_config(self) -> AsinHConfig:
        return AsinHConfig(
            stretch_factor=self._model.asinh_stretch_factor,
            black_point=self._model.asinh_black_point,
            linked_channels=self._model.asinh_linked,
        )
