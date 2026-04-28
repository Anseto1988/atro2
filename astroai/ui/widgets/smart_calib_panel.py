"""Smart Calibration Scanner panel (FR-2.3)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from astroai.core.calibration.matcher import BatchMatchResult, batch_match, suggest_calibration_config
from astroai.core.calibration.scanner import (
    ScannedFrame,
    build_calibration_library,
    partition_by_type,
    scan_directory,
)
from astroai.core.io.fits_io import ImageMetadata

if TYPE_CHECKING:
    from astroai.project.project_file import AstroProject

__all__ = ["SmartCalibPanel"]

_TYPE_LABELS: dict[str, str] = {
    "dark": "Dark",
    "flat": "Flat",
    "bias": "Bias",
    "light": "Light",
    "unknown": "Unbekannt",
}


class SmartCalibPanel(QWidget):
    """Scan a calibration directory and auto-match frames to the current project."""

    def __init__(
        self,
        project_getter: Callable[[], AstroProject | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._get_project = project_getter
        self._scanned_frames: list[ScannedFrame] = []
        self._last_result: BatchMatchResult | None = None
        self._setup_ui()
        self._connect_signals()

    # -- Public API ----------------------------------------------------------

    def set_project_getter(self, getter: Callable[[], AstroProject | None]) -> None:
        self._get_project = getter

    # -- UI ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        dir_group = QGroupBox("Kalibrierungs-Verzeichnis")
        dir_layout = QVBoxLayout(dir_group)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Verzeichnis auswählen…")
        self._dir_edit.setAccessibleName("Verzeichnis für Kalibrierungs-Frames")
        dir_row.addWidget(self._dir_edit, stretch=1)

        self._browse_btn = QPushButton("…")
        self._browse_btn.setMaximumWidth(32)
        self._browse_btn.setAccessibleName("Verzeichnis durchsuchen")
        dir_row.addWidget(self._browse_btn)
        dir_layout.addLayout(dir_row)

        self._recursive_cb = QCheckBox("Unterverzeichnisse einschließen")
        self._recursive_cb.setAccessibleName("Rekursiver Verzeichnis-Scan")
        dir_layout.addWidget(self._recursive_cb)

        self._scan_btn = QPushButton("Verzeichnis scannen")
        self._scan_btn.setAccessibleName("Kalibrierungs-Verzeichnis scannen")
        dir_layout.addWidget(self._scan_btn)
        layout.addWidget(dir_group)

        result_group = QGroupBox("Scan-Ergebnisse")
        result_layout = QVBoxLayout(result_group)

        self._result_table = QTableWidget(0, 2)
        self._result_table.setHorizontalHeaderLabels(["Typ", "Anzahl"])
        self._result_table.horizontalHeader().setStretchLastSection(True)
        self._result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._result_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._result_table.setMaximumHeight(150)
        result_layout.addWidget(self._result_table)
        layout.addWidget(result_group)

        match_group = QGroupBox("Auto-Match")
        match_layout = QVBoxLayout(match_group)

        self._match_btn = QPushButton("Auto-Match anwenden")
        self._match_btn.setEnabled(False)
        self._match_btn.setAccessibleName("Kalibrierungs-Frames automatisch zuordnen")
        match_layout.addWidget(self._match_btn)

        dark_row = QHBoxLayout()
        dark_row.addWidget(QLabel("Dark-Coverage:"))
        self._dark_coverage_label = QLabel("—")
        self._dark_coverage_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        dark_row.addWidget(self._dark_coverage_label)
        match_layout.addLayout(dark_row)

        flat_row = QHBoxLayout()
        flat_row.addWidget(QLabel("Flat-Coverage:"))
        self._flat_coverage_label = QLabel("—")
        self._flat_coverage_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        flat_row.addWidget(self._flat_coverage_label)
        match_layout.addLayout(flat_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        match_layout.addWidget(self._status_label)

        layout.addWidget(match_group)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self._browse_btn.clicked.connect(self._on_browse)
        self._scan_btn.clicked.connect(self._on_scan)
        self._match_btn.clicked.connect(self._on_apply_match)

    # -- Slots ---------------------------------------------------------------

    @Slot()
    def _on_browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Kalibrierungs-Verzeichnis auswählen",
            self._dir_edit.text() or "",
        )
        if directory:
            self._dir_edit.setText(directory)

    @Slot()
    def _on_scan(self) -> None:
        raw = self._dir_edit.text().strip()
        directory = Path(raw) if raw else Path(".")
        if not directory.is_dir():
            self._status_label.setText("Ungültiges Verzeichnis.")
            self._result_table.setRowCount(0)
            self._match_btn.setEnabled(False)
            return

        recursive = self._recursive_cb.isChecked()
        self._scanned_frames = scan_directory(directory, recursive=recursive)
        self._populate_table()
        has_calib = any(f.frame_type in {"dark", "flat", "bias"} for f in self._scanned_frames)
        self._match_btn.setEnabled(has_calib)
        self._status_label.setText(f"{len(self._scanned_frames)} Frame(s) gefunden.")
        self._dark_coverage_label.setText("—")
        self._flat_coverage_label.setText("—")
        self._last_result = None

    def _populate_table(self) -> None:
        groups = partition_by_type(self._scanned_frames)
        self._result_table.setRowCount(0)
        for frame_type in ("dark", "flat", "bias", "light", "unknown"):
            frames = groups.get(frame_type, [])
            if not frames:
                continue
            row = self._result_table.rowCount()
            self._result_table.insertRow(row)
            self._result_table.setItem(
                row, 0, QTableWidgetItem(_TYPE_LABELS.get(frame_type, frame_type))
            )
            count_item = QTableWidgetItem(str(len(frames)))
            count_item.setData(
                Qt.ItemDataRole.TextAlignmentRole,
                int(Qt.AlignmentFlag.AlignCenter),
            )
            self._result_table.setItem(row, 1, count_item)

    @Slot()
    def _on_apply_match(self) -> None:
        from astroai.project.project_file import AstroProject

        project = self._get_project() if self._get_project is not None else None
        if not isinstance(project, AstroProject):
            self._status_label.setText("Kein Projekt geöffnet.")
            return

        library = build_calibration_library(self._scanned_frames)

        lights: list[tuple[Path, ImageMetadata]] = []
        for entry in project.input_frames:
            meta = ImageMetadata(
                exposure=entry.exposure,
                gain_iso=entry.gain_iso,
                temperature=entry.temperature,
            )
            lights.append((Path(entry.path), meta))

        if not lights:
            self._status_label.setText("Keine Light-Frames im Projekt.")
            return

        result = batch_match(lights, library)
        self._last_result = result

        config = suggest_calibration_config(result)
        project.calibration = config
        project.touch()

        self._dark_coverage_label.setText(f"{result.dark_coverage:.0%}")
        self._flat_coverage_label.setText(f"{result.flat_coverage:.0%}")
        self._status_label.setText(
            f"Kalibrierung angewendet: {len(config.dark_frames)} Dark(s),"
            f" {len(config.flat_frames)} Flat(s)."
        )
