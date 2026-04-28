"""Comet-Stacking Konfigurationspanel."""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["CometStackPanel"]

_TRACKING_MODES = [
    ("stars", "Sterne"),
    ("comet", "Kometenkopf"),
    ("blend", "Blend"),
]


class CometStackPanel(QWidget):
    """Panel fuer die Konfiguration des Komet-Stacking-Schritts."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._enabled_cb = QCheckBox("Komet-Stacking aktivieren")
        self._enabled_cb.setAccessibleName("Komet-Stacking-Verarbeitung aktivieren")
        layout.addWidget(self._enabled_cb)

        self._settings_group = QGroupBox("Einstellungen")
        group_layout = QVBoxLayout(self._settings_group)

        mode_label = QLabel("Tracking-Modus:")
        mode_label.setAccessibleName("Tracking-Modus Auswahl")
        group_layout.addWidget(mode_label)

        mode_row = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._mode_buttons: dict[str, QRadioButton] = {}
        for key, label in _TRACKING_MODES:
            btn = QRadioButton(label)
            btn.setAccessibleName(f"Tracking-Modus {label}")
            btn.setProperty("mode_key", key)
            self._mode_group.addButton(btn)
            self._mode_buttons[key] = btn
            mode_row.addWidget(btn)
        group_layout.addLayout(mode_row)

        self._blend_row = QWidget()
        blend_layout = QHBoxLayout(self._blend_row)
        blend_layout.setContentsMargins(0, 0, 0, 0)
        blend_label = QLabel("Blend-Faktor:")
        blend_label.setMinimumWidth(80)
        self._blend_slider = QSlider(Qt.Orientation.Horizontal)
        self._blend_slider.setRange(0, 100)
        self._blend_slider.setValue(50)
        self._blend_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._blend_slider.setTickInterval(10)
        self._blend_slider.setAccessibleName(
            "Blend-Faktor: 0 nur Sterne, 100 nur Kometenkopf"
        )
        self._blend_value = QLabel("0.50")
        self._blend_value.setMinimumWidth(32)
        self._blend_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        blend_layout.addWidget(blend_label)
        blend_layout.addWidget(self._blend_slider, stretch=1)
        blend_layout.addWidget(self._blend_value)
        group_layout.addWidget(self._blend_row)

        layout.addWidget(self._settings_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._enabled_cb.toggled.connect(self._on_enabled_changed)
        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        self._blend_slider.valueChanged.connect(self._on_blend_changed)
        self._model.comet_stack_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._enabled_cb.blockSignals(True)
        self._enabled_cb.setChecked(self._model.comet_stack_enabled)
        self._enabled_cb.blockSignals(False)

        for btn in self._mode_buttons.values():
            btn.blockSignals(True)
        target = self._model.comet_tracking_mode
        if target in self._mode_buttons:
            self._mode_buttons[target].setChecked(True)
        for btn in self._mode_buttons.values():
            btn.blockSignals(False)

        self._blend_slider.blockSignals(True)
        self._blend_slider.setValue(int(self._model.comet_blend_factor * 100))
        self._blend_slider.blockSignals(False)
        self._blend_value.setText(f"{self._model.comet_blend_factor:.2f}")

        self._blend_row.setVisible(self._model.comet_tracking_mode == "blend")
        self._settings_group.setEnabled(self._model.comet_stack_enabled)

    @Slot(bool)
    def _on_enabled_changed(self, checked: bool) -> None:
        self._model.comet_stack_enabled = checked
        self._settings_group.setEnabled(checked)

    @Slot()
    def _on_mode_changed(self) -> None:
        btn = self._mode_group.checkedButton()
        if btn is None:
            return
        mode = btn.property("mode_key")
        self._model.comet_tracking_mode = mode
        self._blend_row.setVisible(mode == "blend")

    @Slot(int)
    def _on_blend_changed(self, value: int) -> None:
        factor = value / 100.0
        self._blend_value.setText(f"{factor:.2f}")
        self._model.comet_blend_factor = factor
