"""Channel Balance panel — per-channel R/G/B background offset adjustment."""
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

from astroai.processing.color.channel_balance import ChannelBalanceConfig, ChannelBalancer
from astroai.ui.models import PipelineModel

__all__ = ["ChannelBalancePanel"]

_CHANNELS: list[tuple[str, str]] = [
    ("cb_r_offset", "Rot:"),
    ("cb_g_offset", "Grün:"),
    ("cb_b_offset", "Blau:"),
]


class ChannelBalancePanel(QWidget):
    """Dock panel for per-channel R/G/B background-level balance."""

    channel_balance_changed = Signal(ChannelBalanceConfig)
    auto_balance_requested = Signal()

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Kanal-Balance")
        group_layout = QVBoxLayout(group)

        for attr, label_text in _CHANNELS:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(50)
            spin = QDoubleSpinBox()
            spin.setRange(-1.0, 1.0)
            spin.setSingleStep(0.001)
            spin.setDecimals(4)
            spin.setValue(0.0)
            spin.setAccessibleName(f"Kanal-Balance {label_text.rstrip(':')}")
            row.addWidget(lbl)
            row.addWidget(spin, stretch=1)
            group_layout.addLayout(row)
            self._spins[attr] = spin

        btn_row = QHBoxLayout()
        self._auto_btn = QPushButton("Auto-Balance")
        self._auto_btn.setToolTip(
            "Schätzt Hintergrund je Kanal und gleicht automatisch aus."
        )
        self._reset_btn = QPushButton("Zurücksetzen")
        btn_row.addWidget(self._auto_btn)
        btn_row.addWidget(self._reset_btn)
        group_layout.addLayout(btn_row)

        layout.addWidget(group)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        for attr, spin in self._spins.items():
            spin.valueChanged.connect(lambda v, a=attr: self._on_spin_changed(a, v))
        self._auto_btn.clicked.connect(self._on_auto_balance)
        self._reset_btn.clicked.connect(self._on_reset)
        self._model.channel_balance_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        for attr, spin in self._spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(self._model, attr))
            spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(str, float)
    def _on_spin_changed(self, attr: str, value: float) -> None:
        setattr(self._model, attr, value)
        self.channel_balance_changed.emit(self._build_config())

    @Slot()
    def _on_auto_balance(self) -> None:
        self.auto_balance_requested.emit()

    @Slot()
    def _on_reset(self) -> None:
        self._model.cb_r_offset = 0.0
        self._model.cb_g_offset = 0.0
        self._model.cb_b_offset = 0.0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def apply_auto_sample(self, config: ChannelBalanceConfig) -> None:
        """Update sliders from an auto-sampled ChannelBalanceConfig."""
        self._model.cb_r_offset = config.r_offset
        self._model.cb_g_offset = config.g_offset
        self._model.cb_b_offset = config.b_offset

    def _build_config(self) -> ChannelBalanceConfig:
        return ChannelBalanceConfig(
            r_offset=self._model.cb_r_offset,
            g_offset=self._model.cb_g_offset,
            b_offset=self._model.cb_b_offset,
        )
