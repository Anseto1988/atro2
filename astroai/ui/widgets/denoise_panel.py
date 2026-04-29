"""Denoising configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from astroai.core.noise_estimator import NoiseEstimate
from astroai.ui.models import PipelineModel

__all__ = ["DenoisePanel"]


class DenoisePanel(QWidget):
    """Panel for configuring the AI denoising step."""

    PREVIEW_STEP = "denoise"
    preview_requested = Signal(dict)
    noise_detect_requested = Signal()

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._settings_group = QGroupBox("KI-Entrauschung (NAFNet)")
        group_layout = QVBoxLayout(self._settings_group)

        strength_row = QHBoxLayout()
        strength_label = QLabel("Stärke:")
        strength_label.setMinimumWidth(130)
        self._strength_spin = QDoubleSpinBox()
        self._strength_spin.setRange(0.0, 1.0)
        self._strength_spin.setSingleStep(0.1)
        self._strength_spin.setDecimals(2)
        self._strength_spin.setValue(1.0)
        self._strength_spin.setAccessibleName("Entrauschungsstaerke 0 bis 1")
        strength_row.addWidget(strength_label)
        strength_row.addWidget(self._strength_spin, stretch=1)
        group_layout.addLayout(strength_row)

        tile_row = QHBoxLayout()
        tile_label = QLabel("Kachel-Größe:")
        tile_label.setMinimumWidth(130)
        self._tile_spin = QSpinBox()
        self._tile_spin.setRange(64, 1024)
        self._tile_spin.setSingleStep(64)
        self._tile_spin.setSuffix(" px")
        self._tile_spin.setValue(512)
        self._tile_spin.setAccessibleName("Kachelgroesse fuer KI-Inferenz in Pixeln")
        tile_row.addWidget(tile_label)
        tile_row.addWidget(self._tile_spin, stretch=1)
        group_layout.addLayout(tile_row)

        overlap_row = QHBoxLayout()
        overlap_label = QLabel("Überlappung:")
        overlap_label.setMinimumWidth(130)
        self._overlap_spin = QSpinBox()
        self._overlap_spin.setRange(0, 256)
        self._overlap_spin.setSingleStep(16)
        self._overlap_spin.setSuffix(" px")
        self._overlap_spin.setValue(64)
        self._overlap_spin.setAccessibleName("Kachelüberlappung in Pixeln")
        overlap_row.addWidget(overlap_label)
        overlap_row.addWidget(self._overlap_spin, stretch=1)
        group_layout.addLayout(overlap_row)

        auto_row = QHBoxLayout()
        self._auto_btn = QPushButton("Rauschen auto-ermitteln")
        self._auto_btn.setToolTip(
            "Analysiert das aktuelle Bild und schlägt optimale Entrauschungsstärke vor"
        )
        self._auto_btn.setAccessibleName("Rauschen automatisch ermitteln")
        auto_row.addWidget(self._auto_btn)
        group_layout.addLayout(auto_row)

        self._noise_label = QLabel("")
        self._noise_label.setWordWrap(True)
        self._noise_label.setStyleSheet("color: #aaa; font-size: 11px;")
        group_layout.addWidget(self._noise_label)

        self._info_label = QLabel(
            "NAFNet KI-Modell entfernt thermisches Rauschen\n"
            "und Photonenrauschen bei minimaler Detailverlust."
        )
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        group_layout.addWidget(self._info_label)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._strength_spin.valueChanged.connect(self._on_strength_changed)
        self._tile_spin.valueChanged.connect(self._on_tile_size_changed)
        self._overlap_spin.valueChanged.connect(self._on_overlap_changed)
        self._auto_btn.clicked.connect(self.noise_detect_requested)
        self._model.denoise_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._strength_spin.blockSignals(True)
        self._strength_spin.setValue(self._model.denoise_strength)
        self._strength_spin.blockSignals(False)

        self._tile_spin.blockSignals(True)
        self._tile_spin.setValue(self._model.denoise_tile_size)
        self._tile_spin.blockSignals(False)

        self._overlap_spin.blockSignals(True)
        self._overlap_spin.setValue(self._model.denoise_tile_overlap)
        self._overlap_spin.blockSignals(False)

    def _emit_preview(self) -> None:
        self.preview_requested.emit({
            "strength": self._model.denoise_strength,
            "tile_size": self._model.denoise_tile_size,
            "tile_overlap": self._model.denoise_tile_overlap,
        })

    @Slot(float)
    def _on_strength_changed(self, value: float) -> None:
        self._model.denoise_strength = value
        self._emit_preview()

    @Slot(int)
    def _on_tile_size_changed(self, value: int) -> None:
        self._model.denoise_tile_size = value
        self._emit_preview()

    @Slot(int)
    def _on_overlap_changed(self, value: int) -> None:
        self._model.denoise_tile_overlap = value
        self._emit_preview()

    def apply_estimate(self, estimate: NoiseEstimate) -> None:
        """Apply a NoiseEstimate: set strength and update status label."""
        self._strength_spin.setValue(estimate.suggested_strength)
        self._noise_label.setText(
            f"σ={estimate.sky_sigma:.4f}  SNR={estimate.snr_db:.1f} dB  "
            f"Rauschen={estimate.noise_level_pct:.1f}%"
        )
