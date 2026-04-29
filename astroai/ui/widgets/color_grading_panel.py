"""Color Grading panel widget — shadow/midtone/highlight RGB shifts."""
from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from astroai.processing.color.color_grading import ColorGradingConfig
from astroai.ui.models import PipelineModel

__all__ = ["ColorGradingPanel"]

# (model_attr, label)
_SHADOW_FIELDS: list[tuple[str, str]] = [
    ("cg_shadow_r", "Rot:"),
    ("cg_shadow_g", "Grün:"),
    ("cg_shadow_b", "Blau:"),
]

_MIDTONE_FIELDS: list[tuple[str, str]] = [
    ("cg_midtone_r", "Rot:"),
    ("cg_midtone_g", "Grün:"),
    ("cg_midtone_b", "Blau:"),
]

_HIGHLIGHT_FIELDS: list[tuple[str, str]] = [
    ("cg_highlight_r", "Rot:"),
    ("cg_highlight_g", "Grün:"),
    ("cg_highlight_b", "Blau:"),
]


def _make_zone_widget(
    fields: list[tuple[str, str]],
    spins: dict[str, QDoubleSpinBox],
) -> QWidget:
    """Build a zone (shadow/midtone/highlight) tab content widget."""
    zone_widget = QWidget()
    zone_layout = QVBoxLayout(zone_widget)
    zone_layout.setContentsMargins(6, 6, 6, 6)

    for attr, label_text in fields:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setMinimumWidth(50)
        spin = QDoubleSpinBox()
        spin.setRange(-0.5, 0.5)
        spin.setSingleStep(0.01)
        spin.setDecimals(2)
        spin.setValue(0.0)
        spin.setAccessibleName(f"Farbabstufung {label_text.rstrip(':')}")
        row.addWidget(lbl)
        row.addWidget(spin, stretch=1)
        zone_layout.addLayout(row)
        spins[attr] = spin

    zone_layout.addStretch()
    return zone_widget


class ColorGradingPanel(QWidget):
    """Panel for shadow/midtone/highlight color grading with per-zone RGB sliders."""

    color_grading_changed = Signal(ColorGradingConfig)

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

        group = QGroupBox("Farbabstufung")
        group_layout = QVBoxLayout(group)

        tabs = QTabWidget()
        tabs.addTab(_make_zone_widget(_SHADOW_FIELDS, self._spins), "Schatten")
        tabs.addTab(_make_zone_widget(_MIDTONE_FIELDS, self._spins), "Mitteltöne")
        tabs.addTab(_make_zone_widget(_HIGHLIGHT_FIELDS, self._spins), "Lichter")
        group_layout.addWidget(tabs)

        self._reset_btn = QPushButton("Alle zurücksetzen")
        group_layout.addWidget(self._reset_btn)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        for attr, spin in self._spins.items():
            spin.valueChanged.connect(lambda v, a=attr: self._on_spin_changed(a, v))
        self._reset_btn.clicked.connect(self._on_reset)
        self._model.color_grading_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        for attr, spin in self._spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(self._model, attr))
            spin.blockSignals(False)

    @Slot(str, float)
    def _on_spin_changed(self, attr: str, value: float) -> None:
        setattr(self._model, attr, value)
        self.color_grading_changed.emit(self._build_config())

    @Slot()
    def _on_reset(self) -> None:
        all_attrs = (
            [a for a, _ in _SHADOW_FIELDS]
            + [a for a, _ in _MIDTONE_FIELDS]
            + [a for a, _ in _HIGHLIGHT_FIELDS]
        )
        for attr in all_attrs:
            setattr(self._model, attr, 0.0)

    def _build_config(self) -> ColorGradingConfig:
        return ColorGradingConfig(
            shadow_r=self._model.cg_shadow_r,
            shadow_g=self._model.cg_shadow_g,
            shadow_b=self._model.cg_shadow_b,
            midtone_r=self._model.cg_midtone_r,
            midtone_g=self._model.cg_midtone_g,
            midtone_b=self._model.cg_midtone_b,
            highlight_r=self._model.cg_highlight_r,
            highlight_g=self._model.cg_highlight_g,
            highlight_b=self._model.cg_highlight_b,
        )
