"""White balance panel widget for per-channel R/G/B multiplier adjustment."""
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

from astroai.processing.color.white_balance import WhiteBalanceConfig
from astroai.ui.models import PipelineModel

__all__ = ["WhiteBalancePanel"]

_CHANNEL_LABELS: list[tuple[str, str]] = [
    ("wb_red",   "Rot:"),
    ("wb_green", "Grün:"),
    ("wb_blue",  "Blau:"),
]


class WhiteBalancePanel(QWidget):
    """Panel for manual per-channel white balance adjustment."""

    white_balance_changed = Signal(WhiteBalanceConfig)

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

        group = QGroupBox("Weißabgleich")
        group_layout = QVBoxLayout(group)

        for attr, label_text in _CHANNEL_LABELS:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(60)
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 5.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(1.0)
            spin.setAccessibleName(f"Weißabgleich {label_text.rstrip(':')}")
            row.addWidget(lbl)
            row.addWidget(spin, stretch=1)
            group_layout.addLayout(row)
            self._spins[attr] = spin

        self._reset_btn = QPushButton("Zurücksetzen")
        group_layout.addWidget(self._reset_btn)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        for attr, spin in self._spins.items():
            spin.valueChanged.connect(lambda v, a=attr: self._on_spin_changed(a, v))
        self._reset_btn.clicked.connect(self._on_reset)
        self._model.white_balance_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        for attr, spin in self._spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(self._model, attr))
            spin.blockSignals(False)

    @Slot(str, float)
    def _on_spin_changed(self, attr: str, value: float) -> None:
        setattr(self._model, attr, value)
        self.white_balance_changed.emit(self._build_config())

    @Slot()
    def _on_reset(self) -> None:
        self._model.wb_red = 1.0
        self._model.wb_green = 1.0
        self._model.wb_blue = 1.0

    def _build_config(self) -> WhiteBalanceConfig:
        return WhiteBalanceConfig(
            red_factor=self._model.wb_red,
            green_factor=self._model.wb_green,
            blue_factor=self._model.wb_blue,
        )
