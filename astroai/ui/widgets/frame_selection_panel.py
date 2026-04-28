"""Frame selection / sub-frame rejection configuration panel."""
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

__all__ = ["FrameSelectionPanel"]


class FrameSelectionPanel(QWidget):
    """Panel for configuring AI-based frame quality filtering."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._enabled_cb = QCheckBox("Frame-Selektion aktivieren")
        self._enabled_cb.setAccessibleName("KI-basierte Frame-Qualitaetsselektion aktivieren")
        layout.addWidget(self._enabled_cb)

        self._settings_group = QGroupBox("Einstellungen")
        group_layout = QVBoxLayout(self._settings_group)

        score_row = QHBoxLayout()
        score_label = QLabel("Mindestscore:")
        score_label.setMinimumWidth(150)
        self._score_spin = QDoubleSpinBox()
        self._score_spin.setRange(0.0, 1.0)
        self._score_spin.setSingleStep(0.05)
        self._score_spin.setDecimals(2)
        self._score_spin.setValue(0.5)
        self._score_spin.setAccessibleName("Minimaler KI-Qualitaetsscore fuer Frame-Akzeptanz")
        score_row.addWidget(score_label)
        score_row.addWidget(self._score_spin, stretch=1)
        group_layout.addLayout(score_row)

        reject_row = QHBoxLayout()
        reject_label = QLabel("Max. Ausschuss:")
        reject_label.setMinimumWidth(150)
        self._reject_spin = QDoubleSpinBox()
        self._reject_spin.setRange(0.0, 1.0)
        self._reject_spin.setSingleStep(0.05)
        self._reject_spin.setDecimals(2)
        self._reject_spin.setValue(0.8)
        self._reject_spin.setAccessibleName("Maximaler Anteil verworfener Frames")
        reject_row.addWidget(reject_label)
        reject_row.addWidget(self._reject_spin, stretch=1)
        group_layout.addLayout(reject_row)

        self._info_label = QLabel(
            "KI bewertet jeden Frame nach HFR, Stern-Rundheit\n"
            "und Bewölkung (0.0 = schlecht, 1.0 = perfekt)."
        )
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self._info_label)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._enabled_cb.toggled.connect(self._on_enabled_changed)
        self._score_spin.valueChanged.connect(self._on_min_score_changed)
        self._reject_spin.valueChanged.connect(self._on_max_rejected_changed)
        self._model.frame_selection_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._enabled_cb.blockSignals(True)
        self._enabled_cb.setChecked(self._model.frame_selection_enabled)
        self._enabled_cb.blockSignals(False)

        self._score_spin.blockSignals(True)
        self._score_spin.setValue(self._model.frame_selection_min_score)
        self._score_spin.blockSignals(False)

        self._reject_spin.blockSignals(True)
        self._reject_spin.setValue(self._model.frame_selection_max_rejected_fraction)
        self._reject_spin.blockSignals(False)

        self._settings_group.setEnabled(self._model.frame_selection_enabled)

    @Slot(bool)
    def _on_enabled_changed(self, checked: bool) -> None:
        self._model.frame_selection_enabled = checked
        self._settings_group.setEnabled(checked)

    @Slot(float)
    def _on_min_score_changed(self, value: float) -> None:
        self._model.frame_selection_min_score = value

    @Slot(float)
    def _on_max_rejected_changed(self, value: float) -> None:
        self._model.frame_selection_max_rejected_fraction = value
