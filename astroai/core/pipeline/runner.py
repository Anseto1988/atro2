"""Background QThread executor for running a configured Pipeline."""
from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, QThread, Signal

from astroai.core.pipeline.base import (
    Pipeline,
    PipelineCancelledError,
    PipelineContext,
    PipelineProgress,
    PipelineStage,
)

__all__ = ["PipelineWorker"]

logger = logging.getLogger(__name__)


class _RunnerWorker(QObject):
    finished = Signal(object)   # PipelineContext
    cancelled = Signal()
    error = Signal(str)
    progress = Signal(object)   # PipelineProgress

    def __init__(
        self,
        pipeline: Pipeline,
        context: PipelineContext,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._context = context
        self._cancel_event = cancel_event

    def run(self) -> None:
        try:
            result = self._pipeline.run(
                self._context,
                self._emit_progress,
                cancel_check=self._cancel_event.is_set,
            )
            self.finished.emit(result)
        except PipelineCancelledError:
            logger.info("Pipeline cancelled by user")
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("Pipeline execution failed: %s", exc)
            self.error.emit(str(exc))

    def _emit_progress(self, p: PipelineProgress) -> None:
        self.progress.emit(p)


class PipelineWorker(QObject):
    """Manages background pipeline execution with live progress signals.

    Mirrors the CalibrationWorker pattern: one active thread at a time,
    auto-cleanup on finish/error/cancel.
    """

    finished = Signal(object)          # PipelineContext
    cancelled = Signal()
    error = Signal(str)
    progress = Signal(float, str)      # (fraction 0-1, status message)
    stage_active = Signal(str)         # PipelineStage.name when stage changes

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _RunnerWorker | None = None
        self._last_stage: PipelineStage | None = None
        self._cancel_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, pipeline: Pipeline, context: PipelineContext) -> None:
        if self.is_running:
            logger.warning("PipelineWorker: already running, ignoring start()")
            return

        self._last_stage = None
        self._cancel_event.clear()
        self._thread = QThread()
        self._worker = _RunnerWorker(pipeline, context, self._cancel_event)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self._on_progress)

        self._thread.start()

    def cancel(self) -> None:
        if self.is_running:
            self._cancel_event.set()
            logger.info("PipelineWorker: cancel requested")

    def _on_finished(self, context: object) -> None:
        self.finished.emit(context)
        self._cleanup()

    def _on_cancelled(self) -> None:
        self.cancelled.emit()
        self._cleanup()

    def _on_error(self, msg: str) -> None:
        self.error.emit(msg)
        self._cleanup()

    def _on_progress(self, p: object) -> None:
        prog = p
        assert isinstance(prog, PipelineProgress)
        if prog.stage is not self._last_stage:
            self._last_stage = prog.stage
            self.stage_active.emit(prog.stage.name)
        self.progress.emit(prog.fraction, prog.message)

    def _cleanup(self) -> None:
        self._last_stage = None
        self._cancel_event.clear()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
