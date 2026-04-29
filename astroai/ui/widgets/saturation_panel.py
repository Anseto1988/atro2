"""Selective HSV saturation adjustment panel."""
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

__all__ = ["SaturationPanel"]

_RANGE_LABELS: list[tuple[str, str]] = [
    ("saturation_global",  "Global:"),
    ("saturation_reds",    "Rot:"),
    ("saturation_oranges", "Orange:"),
    ("saturation_yellows", "Gelb:"),
    ("saturation_greens",  "Grün:"),
    ("saturation_cyans",   "Cyan:"),
    ("saturation_blues",   "Blau:"),
    ("saturation_purples", "Lila:"),
]


class SaturationPanel(QWidget):
    """Panel for per-hue-range saturation adjustment."""

    PREVIEW_STEP = "saturation"
    preview_requested = Signal(dict)

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Selektive Sättigung (HSV)")
        group_layout = QVBoxLayout(group)

        for attr, label_text in _RANGE_LABELS:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(100)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 4.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setValue(1.0)
            spin.setAccessibleName(f"Saettigung {label_text.rstrip(':')}")
            row.addWidget(lbl)
            row.addWidget(spin, stretch=1)
            group_layout.addLayout(row)
            self._spins[attr] = spin

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        for attr, spin in self._spins.items():
            spin.valueChanged.connect(lambda v, a=attr: self._on_spin_changed(a, v))
        self._model.saturation_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        for attr, spin in self._spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(self._model, attr))
            spin.blockSignals(False)

    def _emit_preview(self) -> None:
        self.preview_requested.emit({
            attr: self._spins[attr].value() for attr, _ in _RANGE_LABELS
        })

    @Slot(str, float)
    def _on_spin_changed(self, attr: str, value: float) -> None:
        setattr(self._model, attr, value)
        self._emit_preview()
