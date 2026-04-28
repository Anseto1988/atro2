"""Shared debounced preview engine for processing panels."""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage

__all__ = ["PreviewWorker"]

logger = logging.getLogger(__name__)

MAX_PREVIEW_SIZE = 1024
DEBOUNCE_MS = 300


def create_thumbnail(
    image: NDArray[np.floating[Any]],
    max_size: int = MAX_PREVIEW_SIZE,
) -> NDArray[np.floating[Any]]:
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image.copy()
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    row_idx = np.linspace(0, h - 1, new_h, dtype=int)
    col_idx = np.linspace(0, w - 1, new_w, dtype=int)
    return image[np.ix_(row_idx, col_idx)].copy()


def numpy_to_qimage(arr: NDArray[np.floating[Any]]) -> QImage:
    clipped = np.clip(arr, 0.0, 1.0)
    uint8 = (clipped * 255).astype(np.uint8)
    if uint8.ndim == 2:
        h, w = uint8.shape
        uint8 = np.ascontiguousarray(uint8)
        return QImage(uint8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    h, w, c = uint8.shape
    if c >= 3:
        rgb = np.ascontiguousarray(uint8[:, :, :3])
        return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    uint8 = np.ascontiguousarray(uint8[:, :, 0])
    return QImage(uint8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


def _apply_step(
    step_name: str,
    thumbnail: NDArray[np.floating[Any]],
    params: dict[str, Any],
    cancel: threading.Event,
) -> NDArray[np.floating[Any]]:
    if cancel.is_set():
        return thumbnail

    if step_name == "stretch":
        from astroai.processing.stretch.stretcher import IntelligentStretcher

        stretcher = IntelligentStretcher(
            target_background=params.get("target_background", 0.25),
            shadow_clipping_sigmas=params.get("shadow_clipping_sigmas", -2.8),
            linked_channels=params.get("linked_channels", True),
        )
        return stretcher.stretch(thumbnail)

    if step_name == "curves":
        from astroai.processing.curves.pipeline_step import CurvesStep

        step = CurvesStep(
            rgb_points=params.get("rgb_points"),
            r_points=params.get("r_points"),
            g_points=params.get("g_points"),
            b_points=params.get("b_points"),
        )
        return step._apply_curves(thumbnail.copy())

    if step_name == "denoise":
        from astroai.processing.denoise.denoiser import Denoiser

        denoiser = Denoiser(
            strength=params.get("strength", 1.0),
            tile_size=params.get("tile_size", 256),
            tile_overlap=params.get("tile_overlap", 32),
        )
        return denoiser.denoise(thumbnail)

    if step_name == "background_removal":
        from astroai.processing.background.extractor import (
            BackgroundExtractor,
            ModelMethod,
        )
        from astroai.processing.background.gradient_remover import GradientRemover

        method_str = params.get("method", "rbf")
        method = ModelMethod.RBF if method_str == "rbf" else ModelMethod.POLYNOMIAL
        extractor = BackgroundExtractor(
            tile_size=params.get("tile_size", 64),
            method=method,
        )
        remover = GradientRemover(
            extractor=extractor,
            preserve_median=params.get("preserve_median", True),
        )
        return remover.remove(thumbnail)

    return thumbnail


class _PreviewTask(QObject):
    finished = Signal(object, str)
    error = Signal(str)

    def __init__(
        self,
        step_name: str,
        thumbnail: NDArray[np.floating[Any]],
        params: dict[str, Any],
        cancel_event: threading.Event,
        *,
        diff_original: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        super().__init__()
        self._step_name = step_name
        self._thumbnail = thumbnail
        self._params = params
        self._cancel = cancel_event
        self._diff_original = diff_original

    @Slot()
    def run(self) -> None:
        try:
            if self._cancel.is_set():
                return
            result = _apply_step(
                self._step_name, self._thumbnail, self._params, self._cancel,
            )
            if self._cancel.is_set():
                return
            qimg = numpy_to_qimage(result)
            if self._diff_original is not None:
                diff = np.abs(result.astype(np.float64) - self._diff_original.astype(np.float64))
                diff = np.clip(diff * 3.0, 0.0, 1.0).astype(np.float32)
                diff_img = numpy_to_qimage(diff)
                self.finished.emit((qimg, diff_img), self._step_name)
            else:
                self.finished.emit(qimg, self._step_name)
        except Exception as exc:
            if not self._cancel.is_set():
                logger.exception("Preview failed for %s: %s", self._step_name, exc)
                self.error.emit(str(exc))


class PreviewWorker(QObject):
    """Debounced preview engine running processing steps on thumbnails."""

    preview_ready = Signal(object, str)
    preview_error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._source_image: NDArray[np.floating[Any]] | None = None
        self._diff_enabled: bool = False
        self._pending: tuple[str, dict[str, Any]] | None = None
        self._thread: QThread | None = None
        self._task: _PreviewTask | None = None
        self._cancel_event = threading.Event()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._run_preview)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def set_source_image(self, image: NDArray[np.floating[Any]] | None) -> None:
        self._source_image = image

    def set_diff_enabled(self, enabled: bool) -> None:
        self._diff_enabled = enabled

    @Slot(str, dict)
    def schedule_preview(self, step_name: str, params: dict[str, Any]) -> None:
        self._pending = (step_name, params)
        if self.is_running:
            self._cancel_event.set()
        self._timer.start()

    def cancel(self) -> None:
        self._timer.stop()
        self._cancel_running()

    def _cancel_running(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._cancel_event.set()
            self._thread.quit()
            self._thread.wait(2000)
        self._cleanup()

    @Slot()
    def _run_preview(self) -> None:
        if self._source_image is None or self._pending is None:
            return

        self._cancel_running()

        step_name, params = self._pending
        self._pending = None

        thumbnail = create_thumbnail(self._source_image)
        diff_original = thumbnail.copy() if self._diff_enabled else None

        self._cancel_event.clear()
        self._thread = QThread()
        self._task = _PreviewTask(
            step_name, thumbnail, params, self._cancel_event,
            diff_original=diff_original,
        )
        self._task.moveToThread(self._thread)

        self._thread.started.connect(self._task.run)
        self._task.finished.connect(self._on_finished)
        self._task.error.connect(self._on_error)

        self._thread.start()

    def _on_finished(self, result: object, step_name: str) -> None:
        self.preview_ready.emit(result, step_name)
        self._cleanup()

    def _on_error(self, msg: str) -> None:
        self.preview_error.emit(msg)
        self._cleanup()

    def _cleanup(self) -> None:
        self._cancel_event.clear()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread.deleteLater()
            self._thread = None
        if self._task is not None:
            self._task.deleteLater()
            self._task = None

    def dispose(self) -> None:
        self.cancel()
        self._source_image = None
