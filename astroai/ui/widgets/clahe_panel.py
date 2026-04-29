"""CLAHE panel widget for local contrast enhancement configuration."""
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

from astroai.processing.contrast.clahe import CLAHEConfig
from astroai.ui.models import PipelineModel

__all__ = ["CLAHEPanel"]

_CHANNEL_MODE_LABELS: list[tuple[str, str]] = [
    ("luminance", "Luminanz"),
    ("each", "Pro Kanal"),
]


class CLAHEPanel(QWidget):
    """Panel for CLAHE local contrast enhancement configuration."""

    clahe_changed = Signal(CLAHEConfig)

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Lokale Kontrastverbesserung (CLAHE)")
        group_layout = QVBoxLayout(group)

        # clip_limit
        row_clip = QHBoxLayout()
        lbl_clip = QLabel("Clip-Limit:")
        lbl_clip.setMinimumWidth(100)
        self._spin_clip = QDoubleSpinBox()
        self._spin_clip.setRange(1.0, 10.0)
        self._spin_clip.setSingleStep(0.1)
        self._spin_clip.setDecimals(1)
        self._spin_clip.setValue(2.0)
        self._spin_clip.setAccessibleName("CLAHE Clip-Limit")
        row_clip.addWidget(lbl_clip)
        row_clip.addWidget(self._spin_clip, stretch=1)
        group_layout.addLayout(row_clip)

        # tile_size
        row_tile = QHBoxLayout()
        lbl_tile = QLabel("Kachelgröße:")
        lbl_tile.setMinimumWidth(100)
        self._spin_tile = QSpinBox()
        self._spin_tile.setRange(8, 512)
        self._spin_tile.setSingleStep(8)
        self._spin_tile.setValue(64)
        self._spin_tile.setAccessibleName("CLAHE Kachelgröße")
        row_tile.addWidget(lbl_tile)
        row_tile.addWidget(self._spin_tile, stretch=1)
        group_layout.addLayout(row_tile)

        # channel_mode
        row_mode = QHBoxLayout()
        lbl_mode = QLabel("Kanalmodus:")
        lbl_mode.setMinimumWidth(100)
        self._combo_mode = QComboBox()
        for _mode_key, mode_label in _CHANNEL_MODE_LABELS:
            self._combo_mode.addItem(mode_label)
        self._combo_mode.setAccessibleName("CLAHE Kanalmodus")
        row_mode.addWidget(lbl_mode)
        row_mode.addWidget(self._combo_mode, stretch=1)
        group_layout.addLayout(row_mode)

        # Reset button
        self._reset_btn = QPushButton("Zurücksetzen")
        group_layout.addWidget(self._reset_btn)

        layout.addWidget(group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._spin_clip.valueChanged.connect(self._on_clip_changed)
        self._spin_tile.valueChanged.connect(self._on_tile_changed)
        self._combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self._reset_btn.clicked.connect(self._on_reset)
        self._model.clahe_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._spin_clip.blockSignals(True)
        self._spin_clip.setValue(self._model.clahe_clip_limit)
        self._spin_clip.blockSignals(False)

        self._spin_tile.blockSignals(True)
        self._spin_tile.setValue(self._model.clahe_tile_size)
        self._spin_tile.blockSignals(False)

        mode = self._model.clahe_channel_mode
        idx = next(
            (i for i, (k, _) in enumerate(_CHANNEL_MODE_LABELS) if k == mode), 0
        )
        self._combo_mode.blockSignals(True)
        self._combo_mode.setCurrentIndex(idx)
        self._combo_mode.blockSignals(False)

    @Slot(float)
    def _on_clip_changed(self, value: float) -> None:
        self._model.clahe_clip_limit = value
        self.clahe_changed.emit(self._build_config())

    @Slot(int)
    def _on_tile_changed(self, value: int) -> None:
        self._model.clahe_tile_size = value
        self.clahe_changed.emit(self._build_config())

    @Slot(int)
    def _on_mode_changed(self, index: int) -> None:
        if 0 <= index < len(_CHANNEL_MODE_LABELS):
            mode_key = _CHANNEL_MODE_LABELS[index][0]
            self._model.clahe_channel_mode = mode_key
            self.clahe_changed.emit(self._build_config())

    @Slot()
    def _on_reset(self) -> None:
        self._model.clahe_clip_limit = 2.0
        self._model.clahe_tile_size = 64
        self._model.clahe_channel_mode = "luminance"

    def _build_config(self) -> CLAHEConfig:
        return CLAHEConfig(
            clip_limit=self._model.clahe_clip_limit,
            tile_size=self._model.clahe_tile_size,
            channel_mode=self._model.clahe_channel_mode,
        )
