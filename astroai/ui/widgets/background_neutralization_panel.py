"""Background Neutralization panel widget."""
from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from astroai.processing.color.background_neutralizer import (
    BackgroundNeutralizationConfig,
    SampleMode,
)
from astroai.ui.models import PipelineModel

__all__ = ["BackgroundNeutralizationPanel"]


class BackgroundNeutralizationPanel(QWidget):
    """Panel for background color neutralization parameter adjustment."""

    bg_neutralization_changed = Signal(BackgroundNeutralizationConfig)

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Hintergrundneutralisierung")
        group_layout = QVBoxLayout(group)

        # Mode row
        row_mode = QHBoxLayout()
        lbl_mode = QLabel("Modus:")
        lbl_mode.setMinimumWidth(130)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Auto", SampleMode.AUTO.value)
        self._mode_combo.addItem("Manuell (ROI)", SampleMode.MANUAL.value)
        self._mode_combo.setAccessibleName("Hintergrund-Sampling-Modus")
        row_mode.addWidget(lbl_mode)
        row_mode.addWidget(self._mode_combo, stretch=1)
        group_layout.addLayout(row_mode)

        # Target background row
        row_tgt = QHBoxLayout()
        lbl_tgt = QLabel("Ziel-Hintergrund:")
        lbl_tgt.setMinimumWidth(130)
        self._target_spin = QDoubleSpinBox()
        self._target_spin.setRange(0.0, 0.3)
        self._target_spin.setSingleStep(0.01)
        self._target_spin.setDecimals(3)
        self._target_spin.setValue(0.1)
        self._target_spin.setAccessibleName("Ziel-Hintergrundwert")
        row_tgt.addWidget(lbl_tgt)
        row_tgt.addWidget(self._target_spin, stretch=1)
        group_layout.addLayout(row_tgt)

        # ROI group (only active in manual mode)
        self._roi_group = QGroupBox("ROI (Zeile-Start / Ende, Spalte-Start / Ende)")
        roi_layout = QHBoxLayout(self._roi_group)
        self._roi_r0 = QSpinBox()
        self._roi_r0.setRange(0, 99999)
        self._roi_r0.setAccessibleName("ROI Zeile Start")
        self._roi_r1 = QSpinBox()
        self._roi_r1.setRange(1, 99999)
        self._roi_r1.setValue(100)
        self._roi_r1.setAccessibleName("ROI Zeile Ende")
        self._roi_c0 = QSpinBox()
        self._roi_c0.setRange(0, 99999)
        self._roi_c0.setAccessibleName("ROI Spalte Start")
        self._roi_c1 = QSpinBox()
        self._roi_c1.setRange(1, 99999)
        self._roi_c1.setValue(100)
        self._roi_c1.setAccessibleName("ROI Spalte Ende")
        roi_layout.addWidget(QLabel("R0:"))
        roi_layout.addWidget(self._roi_r0)
        roi_layout.addWidget(QLabel("R1:"))
        roi_layout.addWidget(self._roi_r1)
        roi_layout.addWidget(QLabel("C0:"))
        roi_layout.addWidget(self._roi_c0)
        roi_layout.addWidget(QLabel("C1:"))
        roi_layout.addWidget(self._roi_c1)
        self._roi_group.setEnabled(False)
        group_layout.addWidget(self._roi_group)

        # Reset button
        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton("Zurücksetzen")
        self._reset_btn.setAccessibleName("Hintergrundneutralisierung zurücksetzen")
        btn_row.addStretch()
        btn_row.addWidget(self._reset_btn)
        group_layout.addLayout(btn_row)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._target_spin.valueChanged.connect(self._on_target_changed)
        self._roi_r0.valueChanged.connect(self._on_roi_changed)
        self._roi_r1.valueChanged.connect(self._on_roi_changed)
        self._roi_c0.valueChanged.connect(self._on_roi_changed)
        self._roi_c1.valueChanged.connect(self._on_roi_changed)
        self._reset_btn.clicked.connect(self._on_reset)
        self._model.background_neutralization_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._mode_combo.blockSignals(True)
        self._target_spin.blockSignals(True)
        self._roi_r0.blockSignals(True)
        self._roi_r1.blockSignals(True)
        self._roi_c0.blockSignals(True)
        self._roi_c1.blockSignals(True)

        mode_idx = 0 if self._model.bg_neutralization_sample_mode == SampleMode.AUTO.value else 1
        self._mode_combo.setCurrentIndex(mode_idx)
        self._target_spin.setValue(self._model.bg_neutralization_target)
        roi = self._model.bg_neutralization_roi
        if roi is not None:
            self._roi_r0.setValue(roi[0])
            self._roi_r1.setValue(roi[1])
            self._roi_c0.setValue(roi[2])
            self._roi_c1.setValue(roi[3])
        self._roi_group.setEnabled(mode_idx == 1)

        self._mode_combo.blockSignals(False)
        self._target_spin.blockSignals(False)
        self._roi_r0.blockSignals(False)
        self._roi_r1.blockSignals(False)
        self._roi_c0.blockSignals(False)
        self._roi_c1.blockSignals(False)

    @Slot(int)
    def _on_mode_changed(self, index: int) -> None:
        mode = SampleMode.AUTO.value if index == 0 else SampleMode.MANUAL.value
        self._model.bg_neutralization_sample_mode = mode
        self._roi_group.setEnabled(index == 1)
        self.bg_neutralization_changed.emit(self._build_config())

    @Slot(float)
    def _on_target_changed(self, value: float) -> None:
        self._model.bg_neutralization_target = value
        self.bg_neutralization_changed.emit(self._build_config())

    @Slot(int)
    def _on_roi_changed(self, _: int) -> None:
        r0, r1 = self._roi_r0.value(), self._roi_r1.value()
        c0, c1 = self._roi_c0.value(), self._roi_c1.value()
        self._model.bg_neutralization_roi = (r0, r1, c0, c1)
        self.bg_neutralization_changed.emit(self._build_config())

    @Slot()
    def _on_reset(self) -> None:
        self._model.bg_neutralization_sample_mode = SampleMode.AUTO.value
        self._model.bg_neutralization_target = 0.1
        self._model.bg_neutralization_roi = None

    def _build_config(self) -> BackgroundNeutralizationConfig:
        mode = SampleMode(self._model.bg_neutralization_sample_mode)
        roi = self._model.bg_neutralization_roi if mode is SampleMode.MANUAL else None
        return BackgroundNeutralizationConfig(
            sample_mode=mode,
            target_background=self._model.bg_neutralization_target,
            roi=roi,
        )
