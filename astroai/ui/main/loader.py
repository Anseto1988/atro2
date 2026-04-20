"""Background file loader using QThread to avoid GUI freeze."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal


class _LoadWorker(QObject):
    finished = Signal(object, str)  # (ndarray | None, filename)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def run(self) -> None:
        path = self._path
        suffix = path.suffix.lower()
        self.status.emit(f"Lade {path.name}...")
        try:
            if suffix in (".fits", ".fit", ".fts"):
                from astropy.io import fits

                with fits.open(str(path)) as hdul:
                    data = hdul[0].data  # type: ignore[index]
                    if data is None:
                        self.error.emit(f"Keine Bilddaten in {path.name}")
                        return
                    img = data.astype(np.float32)
            elif suffix in (".tif", ".tiff"):
                from PIL import Image

                pil_img = Image.open(path)
                img = np.array(pil_img, dtype=np.float32)
                if img.ndim == 3:
                    img = np.mean(img, axis=2).astype(np.float32)
            else:
                from PIL import Image

                pil_img = Image.open(path).convert("L")
                img = np.array(pil_img, dtype=np.float32)

            self.finished.emit(img, path.name)
        except Exception as exc:
            self.error.emit(str(exc))


class FileLoader(QObject):
    """Manages background file loading via a worker thread."""

    image_loaded = Signal(object, str)  # (ndarray, filename)
    load_error = Signal(str)
    load_status = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _LoadWorker | None = None

    @property
    def is_loading(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def load(self, path: Path) -> None:
        if self.is_loading:
            return
        self._thread = QThread()
        self._worker = _LoadWorker(path)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.status.connect(self.load_status.emit)

        self._thread.start()

    def _on_finished(self, data: object, name: str) -> None:
        self.image_loaded.emit(data, name)
        self._cleanup()

    def _on_error(self, msg: str) -> None:
        self.load_error.emit(msg)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
