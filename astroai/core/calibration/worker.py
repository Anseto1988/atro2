"""Background worker for GPU batch calibration with benchmark metrics emission."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QThread, Signal

from astroai.core.calibration.gpu_engine import GPUCalibrationEngine, LoadDataFn
from astroai.core.calibration.matcher import CalibrationLibrary
from astroai.core.calibration.metrics import BenchmarkMetrics
from astroai.core.io.fits_io import ImageMetadata

__all__ = ["CalibrationWorker"]


class _WorkerRunner(QObject):
    finished = Signal(object)
    error = Signal(str)
    metrics = Signal(object)

    def __init__(
        self,
        frames: list[NDArray[np.floating[Any]]],
        light_meta: ImageMetadata,
        library: CalibrationLibrary,
        load_data: LoadDataFn | None,
    ) -> None:
        super().__init__()
        self._frames = frames
        self._meta = light_meta
        self._library = library
        self._load_data = load_data

    def run(self) -> None:
        try:
            engine = GPUCalibrationEngine()
            results = engine.calibrate_batch_gpu(
                self._frames,
                self._meta,
                self._library,
                load_data=self._load_data,
                on_metrics=self._emit_metrics,
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))

    def _emit_metrics(self, m: BenchmarkMetrics) -> None:
        self.metrics.emit(m)


class CalibrationWorker(QObject):
    """Manages background GPU batch calibration with live metrics signals."""

    finished = Signal(object)
    error = Signal(str)
    metrics = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._runner: _WorkerRunner | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(
        self,
        frames: list[NDArray[np.floating[Any]]],
        light_meta: ImageMetadata,
        library: CalibrationLibrary,
        load_data: LoadDataFn | None = None,
    ) -> None:
        if self.is_running:
            return

        self._thread = QThread()
        self._runner = _WorkerRunner(frames, light_meta, library, load_data)
        self._runner.moveToThread(self._thread)

        self._thread.started.connect(self._runner.run)
        self._runner.finished.connect(self._on_finished)
        self._runner.error.connect(self._on_error)
        self._runner.metrics.connect(self._on_metrics)

        self._thread.start()

    def _on_finished(self, results: object) -> None:
        self.finished.emit(results)
        self._cleanup()

    def _on_error(self, msg: str) -> None:
        self.error.emit(msg)
        self._cleanup()

    def _on_metrics(self, m: object) -> None:
        self.metrics.emit(m)

    def _cleanup(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
            self._thread = None
        if self._runner is not None:
            self._runner.deleteLater()
            self._runner = None
