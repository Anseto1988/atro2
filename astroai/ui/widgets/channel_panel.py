"""Channel Combiner panel for LRGB and Narrowband compositing."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from astroai.ui.models import PipelineModel

__all__ = ["ChannelCombinerPanel"]

_LRGB_CHANNELS = ["L", "R", "G", "B"]
_NB_CHANNELS = ["Ha", "OIII", "SII"]
_PALETTE_LABELS = {"SHO (Hubble)": "SHO", "HOO": "HOO", "NHO": "NHO"}


class ChannelCombinerPanel(QWidget):
    """Panel for configuring and triggering multi-channel combination."""

    def __init__(self, model: PipelineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._channel_edits: dict[str, QLineEdit] = {}
        self._setup_ui()
        self._connect_signals()
        self._on_mode_changed()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Mode toggle
        mode_group = QGroupBox("Modus")
        mode_layout = QHBoxLayout(mode_group)
        self._rb_lrgb = QRadioButton("LRGB")
        self._rb_lrgb.setChecked(True)
        self._rb_nb = QRadioButton("Narrowband")
        self._mode_buttons = QButtonGroup(self)
        self._mode_buttons.addButton(self._rb_lrgb)
        self._mode_buttons.addButton(self._rb_nb)
        mode_layout.addWidget(self._rb_lrgb)
        mode_layout.addWidget(self._rb_nb)
        layout.addWidget(mode_group)

        # Palette selector (Narrowband only)
        palette_row = QHBoxLayout()
        palette_row.addWidget(QLabel("Palette:"))
        self._palette_combo = QComboBox()
        for label in _PALETTE_LABELS:
            self._palette_combo.addItem(label)
        palette_row.addWidget(self._palette_combo, stretch=1)
        self._palette_widget = QWidget()
        self._palette_widget.setLayout(palette_row)
        layout.addWidget(self._palette_widget)

        # Channel file pickers
        self._lrgb_group = self._build_channel_group("LRGB-Kanäle", _LRGB_CHANNELS)
        self._nb_group = self._build_channel_group("Narrowband-Kanäle", _NB_CHANNELS)
        layout.addWidget(self._lrgb_group)
        layout.addWidget(self._nb_group)

        # Combine button + status
        self._combine_btn = QPushButton("Kombinieren")
        self._combine_btn.setAccessibleName("Kanäle kombinieren")
        layout.addWidget(self._combine_btn)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        layout.addStretch()

    def _build_channel_group(self, title: str, channels: list[str]) -> QGroupBox:
        group = QGroupBox(title)
        grp_layout = QVBoxLayout(group)
        for ch in channels:
            row = QHBoxLayout()
            lbl = QLabel(f"{ch}:")
            lbl.setMinimumWidth(40)
            edit = QLineEdit()
            edit.setPlaceholderText("FITS-Datei…")
            edit.setAccessibleName(f"Kanal {ch} Dateipfad")
            btn = QPushButton("…")
            btn.setFixedWidth(28)
            btn.setAccessibleName(f"Kanal {ch} Datei auswählen")
            btn.clicked.connect(lambda checked=False, e=edit, c=ch: self._pick_file(e, c))
            row.addWidget(lbl)
            row.addWidget(edit, stretch=1)
            row.addWidget(btn)
            grp_layout.addLayout(row)
            self._channel_edits[ch] = edit
        return group

    def _connect_signals(self) -> None:
        self._rb_lrgb.toggled.connect(self._on_mode_changed)
        self._combine_btn.clicked.connect(self._on_combine)
        self._model.pipeline_reset.connect(self._reset_fields)

    @Slot()
    def _on_mode_changed(self) -> None:
        is_lrgb = self._rb_lrgb.isChecked()
        self._lrgb_group.setVisible(is_lrgb)
        self._nb_group.setVisible(not is_lrgb)
        self._palette_widget.setVisible(not is_lrgb)

    @Slot()
    def _on_combine(self) -> None:
        import numpy as np

        from astroai.core.io import read_fits, read_tiff
        from astroai.processing.channels import (
            ChannelCombiner,
            NarrowbandMapper,
            NarrowbandPalette,
        )

        channels: dict[str, np.ndarray] = {}
        keys = _LRGB_CHANNELS if self._rb_lrgb.isChecked() else _NB_CHANNELS
        for key in keys:
            path_str = self._channel_edits[key].text().strip()
            if path_str:
                p = Path(path_str)
                try:
                    if p.suffix.lower() in (".tif", ".tiff"):
                        data, _ = read_tiff(p)
                    else:
                        data, _ = read_fits(p)
                    channels[key] = data
                except Exception as exc:
                    self._status_label.setText(f"Lesefehler {key}: {exc}")
                    return

        if not channels:
            self._status_label.setText("Keine Kanäle geladen.")
            return

        try:
            if self._rb_lrgb.isChecked():
                result = ChannelCombiner().combine_lrgb(
                    L=channels.get("L"),
                    R=channels.get("R"),
                    G=channels.get("G"),
                    B=channels.get("B"),
                )
            else:
                label = self._palette_combo.currentText()
                palette = NarrowbandPalette[_PALETTE_LABELS[label]]
                result = NarrowbandMapper().map(
                    Ha=channels.get("Ha"),
                    OIII=channels.get("OIII"),
                    SII=channels.get("SII"),
                    palette=palette,
                )
            self._status_label.setText(f"OK — Shape: {result.shape}")
            self.model_result_ready(result)
        except Exception as exc:
            self._status_label.setText(f"Fehler: {exc}")

    def model_result_ready(self, result: object) -> None:
        """Override or connect externally to receive the combined image."""

    def _pick_file(self, edit: QLineEdit, channel: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Kanal {channel} öffnen", "",
            "FITS (*.fits *.fit *.fts);;TIFF (*.tif *.tiff);;Alle (*)",
        )
        if path:
            edit.setText(path)

    @Slot()
    def _reset_fields(self) -> None:
        for edit in self._channel_edits.values():
            edit.clear()
        self._status_label.clear()
