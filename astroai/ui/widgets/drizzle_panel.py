"""Drizzle Super-Resolution configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["DrizzlePanel"]

_DROP_SIZE_OPTIONS = [0.5, 0.7, 1.0]


class DrizzlePanel(QWidget):
    """Panel for configuring the Drizzle super-resolution step."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._enabled_cb = QCheckBox("Drizzle aktivieren")
        self._enabled_cb.setAccessibleName("Drizzle-Verarbeitung aktivieren")
        layout.addWidget(self._enabled_cb)

        self._settings_group = QGroupBox("Einstellungen")
        group_layout = QVBoxLayout(self._settings_group)

        # DropSize selection via radio buttons
        drop_label = QLabel("Drop Size:")
        drop_label.setAccessibleName("Pixelfootprint im Output-Grid")
        group_layout.addWidget(drop_label)

        drop_row = QHBoxLayout()
        self._drop_group = QButtonGroup(self)
        self._drop_buttons: list[QRadioButton] = []
        for val in _DROP_SIZE_OPTIONS:
            btn = QRadioButton(str(val))
            btn.setAccessibleName(f"Drop Size {val}")
            self._drop_group.addButton(btn)
            self._drop_buttons.append(btn)
            drop_row.addWidget(btn)
        group_layout.addLayout(drop_row)

        # Scale spinbox
        scale_row = QHBoxLayout()
        scale_label = QLabel("Scale:")
        scale_label.setMinimumWidth(80)
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.5, 3.0)
        self._scale_spin.setSingleStep(0.5)
        self._scale_spin.setDecimals(1)
        self._scale_spin.setValue(1.0)
        self._scale_spin.setAccessibleName("Output-Aufloesung relativ zur Input-Aufloesung")
        scale_row.addWidget(scale_label)
        scale_row.addWidget(self._scale_spin, stretch=1)
        group_layout.addLayout(scale_row)

        # Pixfrac slider
        pixfrac_row = QHBoxLayout()
        pixfrac_label = QLabel("Pixfrac:")
        pixfrac_label.setMinimumWidth(80)
        self._pixfrac_slider = QSlider(Qt.Orientation.Horizontal)
        self._pixfrac_slider.setRange(10, 100)
        self._pixfrac_slider.setValue(100)
        self._pixfrac_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._pixfrac_slider.setTickInterval(10)
        self._pixfrac_slider.setAccessibleName("Pixel-Fraction fuer Drizzle-Algorithmus")
        self._pixfrac_value = QLabel("1.0")
        self._pixfrac_value.setMinimumWidth(32)
        self._pixfrac_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        pixfrac_row.addWidget(pixfrac_label)
        pixfrac_row.addWidget(self._pixfrac_slider, stretch=1)
        pixfrac_row.addWidget(self._pixfrac_value)
        group_layout.addLayout(pixfrac_row)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._enabled_cb.toggled.connect(self._on_enabled_changed)
        self._drop_group.buttonClicked.connect(self._on_drop_size_changed)
        self._scale_spin.valueChanged.connect(self._on_scale_changed)
        self._pixfrac_slider.valueChanged.connect(self._on_pixfrac_changed)
        self._model.drizzle_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._enabled_cb.blockSignals(True)
        self._enabled_cb.setChecked(self._model.drizzle_enabled)
        self._enabled_cb.blockSignals(False)

        # Sync drop size radio buttons
        for btn in self._drop_buttons:
            btn.blockSignals(True)
        target = self._model.drizzle_drop_size
        for btn in self._drop_buttons:
            if float(btn.text()) == target:
                btn.setChecked(True)
                break
        for btn in self._drop_buttons:
            btn.blockSignals(False)

        self._scale_spin.blockSignals(True)
        self._scale_spin.setValue(self._model.drizzle_scale)
        self._scale_spin.blockSignals(False)

        self._pixfrac_slider.blockSignals(True)
        self._pixfrac_slider.setValue(int(self._model.drizzle_pixfrac * 100))
        self._pixfrac_slider.blockSignals(False)
        self._pixfrac_value.setText(f"{self._model.drizzle_pixfrac:.1f}")

        self._settings_group.setEnabled(self._model.drizzle_enabled)

    @Slot(bool)
    def _on_enabled_changed(self, checked: bool) -> None:
        self._model.drizzle_enabled = checked
        self._settings_group.setEnabled(checked)

    @Slot()
    def _on_drop_size_changed(self) -> None:
        btn = self._drop_group.checkedButton()
        if btn:
            self._model.drizzle_drop_size = float(btn.text())

    @Slot(float)
    def _on_scale_changed(self, value: float) -> None:
        self._model.drizzle_scale = value

    @Slot(int)
    def _on_pixfrac_changed(self, value: int) -> None:
        frac = value / 100.0
        self._pixfrac_value.setText(f"{frac:.1f}")
        self._model.drizzle_pixfrac = frac
