"""Mosaic workflow configuration panel."""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["MosaicPanel"]

_BLEND_MODES = ["average", "median", "max", "min", "overlay"]


class MosaicPanel(QWidget):
    """Panel for configuring the Mosaic workflow step."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._setup_ui()
        self._connect_signals()
        self._sync_from_model()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._enabled_cb = QCheckBox("Mosaic aktivieren")
        self._enabled_cb.setAccessibleName("Mosaic-Verarbeitung aktivieren")
        layout.addWidget(self._enabled_cb)

        self._settings_group = QGroupBox("Einstellungen")
        group_layout = QVBoxLayout(self._settings_group)

        # Blend mode dropdown
        blend_row = QHBoxLayout()
        blend_label = QLabel("Blend Mode:")
        blend_label.setMinimumWidth(80)
        self._blend_combo = QComboBox()
        self._blend_combo.addItems(_BLEND_MODES)
        self._blend_combo.setAccessibleName("Blending-Modus fuer Mosaic-Ueberlappungen")
        blend_row.addWidget(blend_label)
        blend_row.addWidget(self._blend_combo, stretch=1)
        group_layout.addLayout(blend_row)

        # Gradient correction checkbox
        self._gradient_cb = QCheckBox("Gradient-Korrektur")
        self._gradient_cb.setAccessibleName("Automatische Gradientenkorrektur an Nahtbereichen")
        group_layout.addWidget(self._gradient_cb)

        # Output scale slider
        scale_row = QHBoxLayout()
        scale_label = QLabel("Output Scale:")
        scale_label.setMinimumWidth(80)
        self._scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._scale_slider.setRange(25, 400)
        self._scale_slider.setValue(100)
        self._scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._scale_slider.setTickInterval(25)
        self._scale_slider.setAccessibleName("Ausgabe-Skalierung fuer das Mosaic")
        self._scale_value = QLabel("1.00x")
        self._scale_value.setMinimumWidth(40)
        self._scale_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        scale_row.addWidget(scale_label)
        scale_row.addWidget(self._scale_slider, stretch=1)
        scale_row.addWidget(self._scale_value)
        group_layout.addLayout(scale_row)

        layout.addWidget(self._settings_group)

        # Panel list
        panels_group = QGroupBox("Panels")
        panels_layout = QVBoxLayout(panels_group)

        self._panel_list = QListWidget()
        self._panel_list.setAccessibleName("Liste der Mosaic-Dateipfade")
        panels_layout.addWidget(self._panel_list)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Hinzufuegen...")
        self._add_btn.setAccessibleName("Dateien zum Mosaic hinzufuegen")
        self._remove_btn = QPushButton("Entfernen")
        self._remove_btn.setAccessibleName("Ausgewaehlte Datei aus dem Mosaic entfernen")
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._remove_btn)
        panels_layout.addLayout(btn_row)

        layout.addWidget(panels_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._enabled_cb.toggled.connect(self._on_enabled_changed)
        self._blend_combo.currentTextChanged.connect(self._on_blend_mode_changed)
        self._gradient_cb.toggled.connect(self._on_gradient_changed)
        self._scale_slider.valueChanged.connect(self._on_scale_changed)
        self._add_btn.clicked.connect(self._on_add_panels)
        self._remove_btn.clicked.connect(self._on_remove_panel)
        self._model.mosaic_config_changed.connect(self._sync_from_model)
        self._model.pipeline_reset.connect(self._sync_from_model)

    def _sync_from_model(self) -> None:
        self._enabled_cb.blockSignals(True)
        self._enabled_cb.setChecked(self._model.mosaic_enabled)
        self._enabled_cb.blockSignals(False)

        self._blend_combo.blockSignals(True)
        idx = self._blend_combo.findText(self._model.mosaic_blend_mode)
        if idx >= 0:
            self._blend_combo.setCurrentIndex(idx)
        self._blend_combo.blockSignals(False)

        self._gradient_cb.blockSignals(True)
        self._gradient_cb.setChecked(self._model.mosaic_gradient_correct)
        self._gradient_cb.blockSignals(False)

        self._scale_slider.blockSignals(True)
        self._scale_slider.setValue(int(self._model.mosaic_output_scale * 100))
        self._scale_slider.blockSignals(False)
        self._scale_value.setText(f"{self._model.mosaic_output_scale:.2f}x")

        self._panel_list.clear()
        for path in self._model.mosaic_panels:
            self._panel_list.addItem(path)

        enabled = self._model.mosaic_enabled
        self._settings_group.setEnabled(enabled)
        self._panel_list.setEnabled(enabled)
        self._add_btn.setEnabled(enabled)
        self._remove_btn.setEnabled(enabled)

    @Slot(bool)
    def _on_enabled_changed(self, checked: bool) -> None:
        self._model.mosaic_enabled = checked
        self._settings_group.setEnabled(checked)
        self._panel_list.setEnabled(checked)
        self._add_btn.setEnabled(checked)
        self._remove_btn.setEnabled(checked)

    @Slot(str)
    def _on_blend_mode_changed(self, text: str) -> None:
        self._model.mosaic_blend_mode = text

    @Slot(bool)
    def _on_gradient_changed(self, checked: bool) -> None:
        self._model.mosaic_gradient_correct = checked

    @Slot(int)
    def _on_scale_changed(self, value: int) -> None:
        scale = value / 100.0
        self._scale_value.setText(f"{scale:.2f}x")
        self._model.mosaic_output_scale = scale

    @Slot()
    def _on_add_panels(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Mosaic-Dateien hinzufuegen",
            "",
            "FITS (*.fits *.fit *.fts);;TIFF (*.tif *.tiff);;PNG (*.png);;Alle (*)",
        )
        for p in paths:
            self._model.add_mosaic_panel(p)

    @Slot()
    def _on_remove_panel(self) -> None:
        item = self._panel_list.currentItem()
        if item is not None:
            self._model.remove_mosaic_panel(item.text())
