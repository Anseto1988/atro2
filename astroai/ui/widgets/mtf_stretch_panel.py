"""MTF (Midtone Transfer Function) stretch panel widget."""
from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from astroai.processing.stretch.mtf_stretch import MidtoneTransferConfig
from astroai.ui.models import PipelineModel

__all__ = ["MTFStretchPanel"]


class MTFStretchPanel(QWidget):
    """Panel for MTF histogram transformation stretch parameter adjustment."""

    mtf_changed = Signal(MidtoneTransferConfig)
    auto_btf_requested = Signal()

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("MTF-Stretch")
        group_layout = QVBoxLayout(group)

        # Midpoint row
        row_mp = QHBoxLayout()
        lbl_mp = QLabel("Mittelpunkt:")
        lbl_mp.setMinimumWidth(130)
        self._midpoint_spin = QDoubleSpinBox()
        self._midpoint_spin.setRange(0.001, 0.499)
        self._midpoint_spin.setSingleStep(0.005)
        self._midpoint_spin.setDecimals(3)
        self._midpoint_spin.setValue(0.25)
        self._midpoint_spin.setAccessibleName("MTF Mittelpunkt")
        row_mp.addWidget(lbl_mp)
        row_mp.addWidget(self._midpoint_spin, stretch=1)
        group_layout.addLayout(row_mp)

        # Shadows clipping row
        row_sc = QHBoxLayout()
        lbl_sc = QLabel("Schatten-Clipping:")
        lbl_sc.setMinimumWidth(130)
        self._shadows_spin = QDoubleSpinBox()
        self._shadows_spin.setRange(0.0, 0.1)
        self._shadows_spin.setSingleStep(0.001)
        self._shadows_spin.setDecimals(3)
        self._shadows_spin.setValue(0.0)
        self._shadows_spin.setAccessibleName("MTF Schatten-Clipping")
        row_sc.addWidget(lbl_sc)
        row_sc.addWidget(self._shadows_spin, stretch=1)
        group_layout.addLayout(row_sc)

        # Highlights row
        row_hl = QHBoxLayout()
        lbl_hl = QLabel("Lichter:")
        lbl_hl.setMinimumWidth(130)
        self._highlights_spin = QDoubleSpinBox()
        self._highlights_spin.setRange(0.98, 1.0)
        self._highlights_spin.setSingleStep(0.001)
        self._highlights_spin.setDecimals(3)
        self._highlights_spin.setValue(1.0)
        self._highlights_spin.setAccessibleName("MTF Lichter")
        row_hl.addWidget(lbl_hl)
        row_hl.addWidget(self._highlights_spin, stretch=1)
        group_layout.addLayout(row_hl)

        # Buttons row
        btn_row = QHBoxLayout()
        self._auto_btf_btn = QPushButton("Auto-BTF")
        self._auto_btf_btn.setAccessibleName("MTF Auto-BTF berechnen")
        self._reset_btn = QPushButton("Zurücksetzen")
        btn_row.addWidget(self._auto_btf_btn)
        btn_row.addWidget(self._reset_btn)
        group_layout.addLayout(btn_row)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._midpoint_spin.valueChanged.connect(self._on_midpoint_changed)
        self._shadows_spin.valueChanged.connect(self._on_shadows_changed)
        self._highlights_spin.valueChanged.connect(self._on_highlights_changed)
        self._auto_btf_btn.clicked.connect(self._on_auto_btf_clicked)
        self._reset_btn.clicked.connect(self._on_reset)
        self._model.mtf_stretch_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._midpoint_spin.blockSignals(True)
        self._shadows_spin.blockSignals(True)
        self._highlights_spin.blockSignals(True)

        self._midpoint_spin.setValue(self._model.mtf_midpoint)
        self._shadows_spin.setValue(self._model.mtf_shadows_clipping)
        self._highlights_spin.setValue(self._model.mtf_highlights)

        self._midpoint_spin.blockSignals(False)
        self._shadows_spin.blockSignals(False)
        self._highlights_spin.blockSignals(False)

    @Slot(float)
    def _on_midpoint_changed(self, value: float) -> None:
        self._model.mtf_midpoint = value
        self.mtf_changed.emit(self._build_config())

    @Slot(float)
    def _on_shadows_changed(self, value: float) -> None:
        self._model.mtf_shadows_clipping = value
        self.mtf_changed.emit(self._build_config())

    @Slot(float)
    def _on_highlights_changed(self, value: float) -> None:
        self._model.mtf_highlights = value
        self.mtf_changed.emit(self._build_config())

    @Slot()
    def _on_auto_btf_clicked(self) -> None:
        self.auto_btf_requested.emit()

    @Slot(float)
    def set_auto_btf_midpoint(self, midpoint: float) -> None:
        """Apply an externally computed Auto-BTF midpoint to model and spin."""
        self._model.mtf_midpoint = midpoint
        self._sync_from_model()
        self.mtf_changed.emit(self._build_config())

    @Slot()
    def _on_reset(self) -> None:
        self._model.mtf_midpoint = 0.25
        self._model.mtf_shadows_clipping = 0.0
        self._model.mtf_highlights = 1.0

    def _build_config(self) -> MidtoneTransferConfig:
        return MidtoneTransferConfig(
            midpoint=self._model.mtf_midpoint,
            shadows_clipping=self._model.mtf_shadows_clipping,
            highlights=self._model.mtf_highlights,
        )
