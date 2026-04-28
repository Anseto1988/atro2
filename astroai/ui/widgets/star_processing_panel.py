"""Star detection and reduction configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from astroai.ui.models import PipelineModel

__all__ = ["StarProcessingPanel"]


class StarProcessingPanel(QWidget):
    """Panel for configuring star detection parameters and star size reduction."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        reduce_group = QGroupBox("Sternreduktion")
        reduce_layout = QVBoxLayout(reduce_group)

        self._reduce_cb = QCheckBox("Sterne reduzieren statt entfernen")
        self._reduce_cb.setAccessibleName("Sterne verkleinern statt vollstaendig entfernen")
        reduce_layout.addWidget(self._reduce_cb)

        factor_row = QHBoxLayout()
        factor_label = QLabel("Faktor:")
        factor_label.setMinimumWidth(60)
        self._factor_slider = QSlider(Qt.Orientation.Horizontal)
        self._factor_slider.setRange(0, 100)
        self._factor_slider.setValue(50)
        self._factor_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._factor_slider.setTickInterval(10)
        self._factor_slider.setAccessibleName("Sternreduktionsfaktor 0 entfernen bis 100 behalten")
        self._factor_value = QLabel("50%")
        self._factor_value.setMinimumWidth(40)
        self._factor_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        factor_row.addWidget(factor_label)
        factor_row.addWidget(self._factor_slider, stretch=1)
        factor_row.addWidget(self._factor_value)
        reduce_layout.addLayout(factor_row)

        self._reduce_info = QLabel(
            "0% = vollständige Entfernung, 100% = Originalhelligkeit."
        )
        self._reduce_info.setWordWrap(True)
        self._reduce_info.setStyleSheet("color: #888; font-size: 11px;")
        reduce_layout.addWidget(self._reduce_info)

        layout.addWidget(reduce_group)

        detect_group = QGroupBox("Sternerkennung (Erweitert)")
        detect_layout = QVBoxLayout(detect_group)

        sigma_row = QHBoxLayout()
        sigma_label = QLabel("Erkennungs-Sigma:")
        sigma_label.setMinimumWidth(150)
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(1.0, 10.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setValue(4.0)
        self._sigma_spin.setAccessibleName("Sigma-Schwelle fuer Sternerkennung")
        sigma_row.addWidget(sigma_label)
        sigma_row.addWidget(self._sigma_spin, stretch=1)
        detect_layout.addLayout(sigma_row)

        min_area_row = QHBoxLayout()
        min_label = QLabel("Min. Sternfläche:")
        min_label.setMinimumWidth(150)
        self._min_area_spin = QSpinBox()
        self._min_area_spin.setRange(1, 500)
        self._min_area_spin.setSingleStep(1)
        self._min_area_spin.setSuffix(" px²")
        self._min_area_spin.setValue(3)
        self._min_area_spin.setAccessibleName("Minimale Sternflaeche in Pixeln quadrat")
        min_area_row.addWidget(min_label)
        min_area_row.addWidget(self._min_area_spin, stretch=1)
        detect_layout.addLayout(min_area_row)

        max_area_row = QHBoxLayout()
        max_label = QLabel("Max. Sternfläche:")
        max_label.setMinimumWidth(150)
        self._max_area_spin = QSpinBox()
        self._max_area_spin.setRange(100, 50000)
        self._max_area_spin.setSingleStep(100)
        self._max_area_spin.setSuffix(" px²")
        self._max_area_spin.setValue(5000)
        self._max_area_spin.setAccessibleName("Maximale Sternflaeche in Pixeln quadrat")
        max_area_row.addWidget(max_label)
        max_area_row.addWidget(self._max_area_spin, stretch=1)
        detect_layout.addLayout(max_area_row)

        dilation_row = QHBoxLayout()
        dilation_label = QLabel("Masken-Dilatation:")
        dilation_label.setMinimumWidth(150)
        self._dilation_spin = QSpinBox()
        self._dilation_spin.setRange(0, 20)
        self._dilation_spin.setSingleStep(1)
        self._dilation_spin.setSuffix(" px")
        self._dilation_spin.setValue(3)
        self._dilation_spin.setAccessibleName("Dilatation der Sternmaske in Pixeln")
        dilation_row.addWidget(dilation_label)
        dilation_row.addWidget(self._dilation_spin, stretch=1)
        detect_layout.addLayout(dilation_row)

        layout.addWidget(detect_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._reduce_cb.toggled.connect(self._on_reduce_enabled_changed)
        self._factor_slider.valueChanged.connect(self._on_factor_changed)
        self._sigma_spin.valueChanged.connect(self._on_sigma_changed)
        self._min_area_spin.valueChanged.connect(self._on_min_area_changed)
        self._max_area_spin.valueChanged.connect(self._on_max_area_changed)
        self._dilation_spin.valueChanged.connect(self._on_dilation_changed)
        self._model.star_processing_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._reduce_cb.blockSignals(True)
        self._reduce_cb.setChecked(self._model.star_reduce_enabled)
        self._reduce_cb.blockSignals(False)

        self._factor_slider.blockSignals(True)
        factor_pct = int(self._model.star_reduce_factor * 100)
        self._factor_slider.setValue(factor_pct)
        self._factor_slider.blockSignals(False)
        self._factor_value.setText(f"{factor_pct}%")

        self._sigma_spin.blockSignals(True)
        self._sigma_spin.setValue(self._model.star_detection_sigma)
        self._sigma_spin.blockSignals(False)

        self._min_area_spin.blockSignals(True)
        self._min_area_spin.setValue(self._model.star_min_area)
        self._min_area_spin.blockSignals(False)

        self._max_area_spin.blockSignals(True)
        self._max_area_spin.setValue(self._model.star_max_area)
        self._max_area_spin.blockSignals(False)

        self._dilation_spin.blockSignals(True)
        self._dilation_spin.setValue(self._model.star_mask_dilation)
        self._dilation_spin.blockSignals(False)

    @Slot(bool)
    def _on_reduce_enabled_changed(self, checked: bool) -> None:
        self._model.star_reduce_enabled = checked

    @Slot(int)
    def _on_factor_changed(self, value: int) -> None:
        self._factor_value.setText(f"{value}%")
        self._model.star_reduce_factor = value / 100.0

    @Slot(float)
    def _on_sigma_changed(self, value: float) -> None:
        self._model.star_detection_sigma = value

    @Slot(int)
    def _on_min_area_changed(self, value: int) -> None:
        self._model.star_min_area = value

    @Slot(int)
    def _on_max_area_changed(self, value: int) -> None:
        self._model.star_max_area = value

    @Slot(int)
    def _on_dilation_changed(self, value: int) -> None:
        self._model.star_mask_dilation = value
