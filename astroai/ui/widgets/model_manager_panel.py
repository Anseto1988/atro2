"""ONNX model management panel – download, verify, delete."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from astroai.inference.models.downloader import MODELS_DIR, ModelDownloader

__all__ = ["ModelManagerPanel"]

_COL_NAME = 0
_COL_DESC = 1
_COL_STATUS = 2
_COL_SIZE = 3
_COL_PROGRESS = 4
_COL_ACTIONS = 5
_NUM_COLS = 6

_STATUS_PRESENT = "Heruntergeladen"
_STATUS_MISSING = "Nicht vorhanden"
_STATUS_CORRUPT = "Prüfsummen-Fehler"
_STATUS_UNAVAILABLE = "Nicht verfügbar"
_STATUS_DOWNLOADING = "Wird heruntergeladen…"

_COLOR_OK = "#44cc44"
_COLOR_MISSING = "#cc4444"
_COLOR_CORRUPT = "#cc8800"
_COLOR_UNAVAIL = "#888888"
_COLOR_ACTIVE = "#4488cc"


def _fmt_size(path: Path) -> str:
    if not path.exists():
        return "—"
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"
    if size < 1024 ** 3:
        return f"{size / 1024 ** 2:.1f} MB"
    return f"{size / 1024 ** 3:.2f} GB"


class _DownloadWorker(QObject):
    """Background worker: runs ModelDownloader.ensure_model() in a QThread."""

    progress_pct = Signal(int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, name: str, models_dir: Path) -> None:
        super().__init__()
        self._name = name
        self._models_dir = models_dir

    @Slot()
    def run(self) -> None:
        from astroai.core.pipeline.base import PipelineProgress

        def _cb(p: PipelineProgress) -> None:
            if p.total > 0:
                self.progress_pct.emit(min(int(p.current * 100 / p.total), 100))

        try:
            dl = ModelDownloader(models_dir=self._models_dir, progress=_cb)
            dl.ensure_model(self._name)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
        else:
            self.finished.emit()


class ModelManagerPanel(QWidget):
    """Qt panel for managing downloadable ONNX models.

    Shows download status, file size, and per-model action buttons
    (Download / Verify / Delete). Downloads run in a QThread so the UI
    stays responsive.
    """

    status_message = Signal(str)

    def __init__(
        self,
        models_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._models_dir = models_dir or MODELS_DIR
        self._downloader = ModelDownloader(models_dir=self._models_dir)
        self._active_threads: dict[str, QThread] = {}
        self._progress_bars: dict[str, QProgressBar] = {}
        self._setup_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("KI-Modelle verwalten")
        group_layout = QVBoxLayout(group)

        self._table = QTableWidget(0, _NUM_COLS)
        self._table.setHorizontalHeaderLabels(
            ["Modell", "Beschreibung", "Status", "Größe", "Fortschritt", "Aktionen"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setAccessibleName("Modell-Verwaltungstabelle")
        group_layout.addWidget(self._table)

        refresh_btn = QPushButton("Status aktualisieren")
        refresh_btn.setAccessibleName("Modellstatus aktualisieren")
        refresh_btn.clicked.connect(self.refresh)
        group_layout.addWidget(refresh_btn)

        info = QLabel(
            "Modelle werden in <b>~/.astroai/models/</b> gespeichert."
        )
        info.setStyleSheet("color: #888; font-size: 11px;")
        info.setWordWrap(True)
        group_layout.addWidget(info)

        layout.addWidget(group)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_download(self, name: str) -> None:
        """Public API: start downloading *name* (no-op if already active)."""
        self._start_download(name)

    @Slot()
    def refresh(self) -> None:
        """Reload model statuses and rebuild the table rows."""
        manifest = self._downloader.get_manifest()
        self._table.setRowCount(len(manifest))

        for row, (name, entry) in enumerate(manifest.items()):
            model_path = self._models_dir / entry.filename
            downloadable = self._downloader.is_downloadable(name)
            status, color = self._compute_status(name, model_path, downloadable)

            name_item = QTableWidgetItem(name)
            self._table.setItem(row, _COL_NAME, name_item)

            self._table.setItem(row, _COL_DESC, QTableWidgetItem(entry.description))

            status_item = QTableWidgetItem(status)
            status_item.setForeground(QColor(color))
            self._table.setItem(row, _COL_STATUS, status_item)

            self._table.setItem(row, _COL_SIZE, QTableWidgetItem(_fmt_size(model_path)))

            if name not in self._progress_bars:
                pb = QProgressBar()
                pb.setRange(0, 100)
                pb.setValue(0)
                pb.setVisible(False)
                pb.setAccessibleName(f"Fortschritt {name}")
                self._progress_bars[name] = pb
            self._table.setCellWidget(row, _COL_PROGRESS, self._progress_bars[name])

            actions = self._make_action_widget(name, downloadable, model_path, entry.filename)
            self._table.setCellWidget(row, _COL_ACTIONS, actions)

        self._table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_status(
        self, name: str, model_path: Path, downloadable: bool
    ) -> tuple[str, str]:
        if not downloadable:
            return _STATUS_UNAVAILABLE, _COLOR_UNAVAIL
        if name in self._active_threads:
            return _STATUS_DOWNLOADING, _COLOR_ACTIVE
        if not model_path.exists():
            return _STATUS_MISSING, _COLOR_MISSING
        if self._downloader.is_available(name):
            return _STATUS_PRESENT, _COLOR_OK
        return _STATUS_CORRUPT, _COLOR_CORRUPT

    def _make_action_widget(
        self, name: str, downloadable: bool, model_path: Path, filename: str
    ) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)

        dl_btn = QPushButton("Herunterladen")
        dl_btn.setEnabled(downloadable and name not in self._active_threads)
        dl_btn.setToolTip(f"Modell '{name}' herunterladen")
        dl_btn.setAccessibleName(f"{name} herunterladen")
        dl_btn.clicked.connect(lambda _checked=False, n=name: self._start_download(n))

        verify_btn = QPushButton("Verifizieren")
        verify_btn.setEnabled(model_path.exists() and downloadable)
        verify_btn.setToolTip("Prüfsumme des Modells prüfen")
        verify_btn.setAccessibleName(f"{name} verifizieren")
        verify_btn.clicked.connect(lambda _checked=False, n=name: self._verify_model(n))

        del_btn = QPushButton("Löschen")
        del_btn.setEnabled(model_path.exists())
        del_btn.setToolTip(f"Modelldatei '{filename}' löschen")
        del_btn.setAccessibleName(f"{name} löschen")
        del_btn.clicked.connect(lambda _checked=False, n=name: self._delete_model(n))

        layout.addWidget(dl_btn)
        layout.addWidget(verify_btn)
        layout.addWidget(del_btn)
        return container

    def _start_download(self, name: str) -> None:
        if name in self._active_threads:
            return

        pb = self._progress_bars.get(name)
        if pb:
            pb.setValue(0)
            pb.setVisible(True)

        thread = QThread(self)
        worker = _DownloadWorker(name, self._models_dir)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress_pct.connect(lambda pct, n=name: self._on_download_progress(n, pct))
        worker.finished.connect(lambda n=name: self._on_download_finished(n))
        worker.error.connect(lambda msg, n=name: self._on_download_error(n, msg))
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)

        self._active_threads[name] = thread
        self.refresh()
        thread.start()
        self.status_message.emit(f"Download gestartet: {name}")

    @Slot(str, int)
    def _on_download_progress(self, name: str, pct: int) -> None:
        pb = self._progress_bars.get(name)
        if pb:
            pb.setValue(pct)

    @Slot(str)
    def _on_download_finished(self, name: str) -> None:
        self._active_threads.pop(name, None)
        pb = self._progress_bars.get(name)
        if pb:
            pb.setVisible(False)
        self.refresh()
        self.status_message.emit(f"Download abgeschlossen: {name}")

    @Slot(str, str)
    def _on_download_error(self, name: str, message: str) -> None:
        self._active_threads.pop(name, None)
        pb = self._progress_bars.get(name)
        if pb:
            pb.setVisible(False)
        self.refresh()
        self.status_message.emit(f"Download-Fehler {name}: {message}")
        QMessageBox.critical(
            self,
            "Download-Fehler",
            f"Fehler beim Download von '{name}':\n{message}",
        )

    def _verify_model(self, name: str) -> None:
        ok = self._downloader.is_available(name)
        if ok:
            QMessageBox.information(
                self, "Verifizierung", f"Modell '{name}' ist korrekt (Prüfsumme OK)."
            )
            self.status_message.emit(f"Modell {name} verifiziert: OK")
        else:
            QMessageBox.warning(
                self,
                "Verifizierung fehlgeschlagen",
                f"Prüfsummenfehler für '{name}'.\nDie Datei könnte beschädigt sein.",
            )
            self.status_message.emit(f"Modell {name}: Prüfsummen-Fehler")
        self.refresh()

    def _delete_model(self, name: str) -> None:
        manifest = self._downloader.get_manifest()
        if name not in manifest:
            return
        model_path = self._models_dir / manifest[name].filename
        if not model_path.exists():
            return
        reply = QMessageBox.question(
            self,
            "Modell löschen",
            f"Modell '{name}' wirklich löschen?\nDie Datei '{manifest[name].filename}' wird entfernt.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            model_path.unlink()
            self.refresh()
            self.status_message.emit(f"Modell {name} gelöscht")
