"""Tests for the live-preview engine (PreviewWorker) and panel integration."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage

from astroai.ui.models import PipelineModel
from astroai.ui.preview_worker import (
    DEBOUNCE_MS,
    MAX_PREVIEW_SIZE,
    PreviewWorker,
    _PreviewTask,
    _apply_step,
    create_thumbnail,
    numpy_to_qimage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(h: int = 256, w: int = 256, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(42)
    if channels == 0:
        return rng.random((h, w), dtype=np.float32)
    return rng.random((h, w, channels), dtype=np.float32)


# ---------------------------------------------------------------------------
# create_thumbnail
# ---------------------------------------------------------------------------

class TestCreateThumbnail:
    def test_small_image_unchanged(self):
        img = _make_image(100, 100)
        thumb = create_thumbnail(img)
        assert thumb.shape == img.shape
        np.testing.assert_array_equal(thumb, img)

    def test_large_image_downscaled(self):
        img = _make_image(2048, 4096)
        thumb = create_thumbnail(img, max_size=1024)
        assert max(thumb.shape[:2]) <= 1024

    def test_returns_copy(self):
        img = _make_image(64, 64)
        thumb = create_thumbnail(img)
        assert thumb is not img

    def test_preserves_channels(self):
        img = _make_image(2000, 1000, 3)
        thumb = create_thumbnail(img, max_size=512)
        assert thumb.ndim == 3
        assert thumb.shape[2] == 3

    def test_grayscale_image(self):
        img = _make_image(2048, 2048, 0)
        thumb = create_thumbnail(img, max_size=1024)
        assert thumb.ndim == 2
        assert max(thumb.shape) <= 1024


# ---------------------------------------------------------------------------
# numpy_to_qimage
# ---------------------------------------------------------------------------

class TestNumpyToQImage:
    def test_rgb_image(self):
        img = _make_image(64, 64, 3)
        qimg = numpy_to_qimage(img)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 64
        assert qimg.height() == 64

    def test_grayscale_image(self):
        img = _make_image(32, 32, 0)
        qimg = numpy_to_qimage(img)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 32

    def test_clipping(self):
        img = np.full((10, 10, 3), 2.0, dtype=np.float32)
        qimg = numpy_to_qimage(img)
        assert isinstance(qimg, QImage)


# ---------------------------------------------------------------------------
# _apply_step
# ---------------------------------------------------------------------------

class TestApplyStep:
    def test_cancel_returns_input(self):
        img = _make_image(64, 64)
        cancel = threading.Event()
        cancel.set()
        result = _apply_step("stretch", img, {}, cancel)
        np.testing.assert_array_equal(result, img)

    def test_unknown_step_returns_input(self):
        img = _make_image(64, 64)
        cancel = threading.Event()
        result = _apply_step("unknown_step", img, {}, cancel)
        np.testing.assert_array_equal(result, img)

    def test_stretch_step_runs(self):
        img = _make_image(64, 64)
        cancel = threading.Event()
        params = {"target_background": 0.25, "shadow_clipping_sigmas": -2.8, "linked_channels": True}
        result = _apply_step("stretch", img, params, cancel)
        assert result.shape == img.shape
        assert not np.array_equal(result, img)

    def test_curves_step_runs(self):
        img = _make_image(64, 64)
        cancel = threading.Event()
        params = {
            "rgb_points": [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
            "r_points": [(0.0, 0.0), (1.0, 1.0)],
            "g_points": [(0.0, 0.0), (1.0, 1.0)],
            "b_points": [(0.0, 0.0), (1.0, 1.0)],
        }
        result = _apply_step("curves", img, params, cancel)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# _PreviewTask
# ---------------------------------------------------------------------------

class TestPreviewTask:
    def test_emits_finished(self, qtbot):
        img = _make_image(32, 32)
        cancel = threading.Event()
        task = _PreviewTask("stretch", img, {
            "target_background": 0.25,
            "shadow_clipping_sigmas": -2.8,
            "linked_channels": True,
        }, cancel)

        results = []
        task.finished.connect(lambda r, s: results.append((r, s)))
        task.run()

        assert len(results) == 1
        assert results[0][1] == "stretch"
        assert isinstance(results[0][0], QImage)

    def test_cancelled_no_emit(self, qtbot):
        img = _make_image(32, 32)
        cancel = threading.Event()
        cancel.set()
        task = _PreviewTask("stretch", img, {}, cancel)

        results = []
        task.finished.connect(lambda r, s: results.append((r, s)))
        task.run()

        assert len(results) == 0

    def test_diff_mode(self, qtbot):
        img = _make_image(32, 32)
        cancel = threading.Event()
        params = {
            "rgb_points": [(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)],
            "r_points": [(0.0, 0.0), (1.0, 1.0)],
            "g_points": [(0.0, 0.0), (1.0, 1.0)],
            "b_points": [(0.0, 0.0), (1.0, 1.0)],
        }
        task = _PreviewTask("curves", img, params, cancel, diff_original=img.copy())

        results = []
        task.finished.connect(lambda r, s: results.append((r, s)))
        task.run()

        assert len(results) == 1
        result_pair, step = results[0]
        assert isinstance(result_pair, tuple)
        assert len(result_pair) == 2
        assert isinstance(result_pair[0], QImage)
        assert isinstance(result_pair[1], QImage)

    def test_error_emits_error_signal(self, qtbot):
        img = _make_image(32, 32)
        cancel = threading.Event()
        task = _PreviewTask("stretch", img, {
            "target_background": 0.25,
            "shadow_clipping_sigmas": -2.8,
            "linked_channels": True,
        }, cancel)

        errors = []
        task.error.connect(lambda msg: errors.append(msg))

        with patch(
            "astroai.ui.preview_worker._apply_step",
            side_effect=RuntimeError("test error"),
        ):
            task.run()

        assert len(errors) == 1
        assert "test error" in errors[0]


# ---------------------------------------------------------------------------
# PreviewWorker
# ---------------------------------------------------------------------------

class TestPreviewWorker:
    def test_initial_state(self, qtbot):
        worker = PreviewWorker()
        assert not worker.is_running
        assert worker._source_image is None
        assert worker._diff_enabled is False

    def test_set_source_image(self, qtbot):
        worker = PreviewWorker()
        img = _make_image(128, 128)
        worker.set_source_image(img)
        assert worker._source_image is not None

    def test_set_diff_enabled(self, qtbot):
        worker = PreviewWorker()
        worker.set_diff_enabled(True)
        assert worker._diff_enabled is True

    def test_schedule_without_source_no_crash(self, qtbot):
        worker = PreviewWorker()
        worker.schedule_preview("stretch", {"target_background": 0.25})
        worker._timer.stop()

    def test_cancel_stops_timer(self, qtbot):
        worker = PreviewWorker()
        worker.set_source_image(_make_image(64, 64))
        worker.schedule_preview("stretch", {})
        assert worker._timer.isActive()
        worker.cancel()
        assert not worker._timer.isActive()

    def test_debounce_resets_on_rapid_calls(self, qtbot):
        worker = PreviewWorker()
        worker.set_source_image(_make_image(64, 64))

        worker.schedule_preview("stretch", {"target_background": 0.1})
        worker.schedule_preview("stretch", {"target_background": 0.2})
        worker.schedule_preview("stretch", {"target_background": 0.3})

        assert worker._pending is not None
        assert worker._pending[1]["target_background"] == 0.3
        worker.cancel()

    def test_dispose_cleans_up(self, qtbot):
        worker = PreviewWorker()
        worker.set_source_image(_make_image(64, 64))
        worker.dispose()
        assert worker._source_image is None


# ---------------------------------------------------------------------------
# Panel integration: preview_requested signal
# ---------------------------------------------------------------------------

class TestStretchPanelPreview:
    def test_emits_preview_on_bg_change(self, qtbot):
        from astroai.ui.widgets.stretch_panel import StretchPanel

        model = PipelineModel()
        panel = StretchPanel(model)
        qtbot.addWidget(panel)

        signals = []
        panel.preview_requested.connect(lambda p: signals.append(p))

        panel._bg_spin.setValue(0.4)

        assert len(signals) >= 1
        assert signals[-1]["target_background"] == pytest.approx(0.4)

    def test_emits_preview_on_sigma_change(self, qtbot):
        from astroai.ui.widgets.stretch_panel import StretchPanel

        model = PipelineModel()
        panel = StretchPanel(model)
        qtbot.addWidget(panel)

        signals = []
        panel.preview_requested.connect(lambda p: signals.append(p))

        panel._sigma_spin.setValue(-5.0)

        assert len(signals) >= 1
        assert signals[-1]["shadow_clipping_sigmas"] == pytest.approx(-5.0)

    def test_has_preview_step(self):
        from astroai.ui.widgets.stretch_panel import StretchPanel
        assert StretchPanel.PREVIEW_STEP == "stretch"


class TestCurvesPanelPreview:
    def test_emits_preview_on_point_change(self, qtbot):
        from astroai.ui.widgets.curves_panel import CurvesPanel

        model = PipelineModel()
        panel = CurvesPanel(model)
        qtbot.addWidget(panel)

        signals = []
        panel.preview_requested.connect(lambda p: signals.append(p))

        panel._on_points_changed([(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)])

        assert len(signals) >= 1
        assert "rgb_points" in signals[-1]

    def test_has_preview_step(self):
        from astroai.ui.widgets.curves_panel import CurvesPanel
        assert CurvesPanel.PREVIEW_STEP == "curves"


class TestDenoisePanelPreview:
    def test_emits_preview_on_strength_change(self, qtbot):
        from astroai.ui.widgets.denoise_panel import DenoisePanel

        model = PipelineModel()
        panel = DenoisePanel(model)
        qtbot.addWidget(panel)

        signals = []
        panel.preview_requested.connect(lambda p: signals.append(p))

        panel._strength_spin.setValue(0.5)

        assert len(signals) >= 1
        assert signals[-1]["strength"] == pytest.approx(0.5)

    def test_has_preview_step(self):
        from astroai.ui.widgets.denoise_panel import DenoisePanel
        assert DenoisePanel.PREVIEW_STEP == "denoise"


class TestBackgroundRemovalPanelPreview:
    def test_emits_preview_on_tile_change(self, qtbot):
        from astroai.ui.widgets.background_removal_panel import BackgroundRemovalPanel

        model = PipelineModel()
        panel = BackgroundRemovalPanel(model)
        qtbot.addWidget(panel)

        signals = []
        panel.preview_requested.connect(lambda p: signals.append(p))

        panel._tile_spin.setValue(128)

        assert len(signals) >= 1
        assert signals[-1]["tile_size"] == 128

    def test_has_preview_step(self):
        from astroai.ui.widgets.background_removal_panel import BackgroundRemovalPanel
        assert BackgroundRemovalPanel.PREVIEW_STEP == "background_removal"


# ---------------------------------------------------------------------------
# numpy_to_qimage — single-channel 3D array (HxWx1) → grayscale path (lines 47-48)
# ---------------------------------------------------------------------------

class TestNumpyToQImageSingleChannel:
    def test_single_channel_3d_array(self):
        img = _make_image(32, 32, 1)  # shape (32, 32, 1)
        qimg = numpy_to_qimage(img)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 32
        assert qimg.height() == 32


# ---------------------------------------------------------------------------
# _apply_step — denoise and background_removal branches (lines 81-108)
# ---------------------------------------------------------------------------

class TestApplyStepExtraBranches:
    def test_denoise_step_runs(self):
        img = _make_image(64, 64)
        cancel = threading.Event()
        params = {"strength": 0.5, "tile_size": 64, "tile_overlap": 8}
        result = _apply_step("denoise", img, params, cancel)
        assert result.shape == img.shape

    def test_background_removal_rbf_step_runs(self):
        img = _make_image(64, 64)
        cancel = threading.Event()
        params = {"method": "rbf", "tile_size": 32, "preserve_median": True}
        result = _apply_step("background_removal", img, params, cancel)
        assert result.shape == img.shape

    def test_background_removal_polynomial_step_runs(self):
        img = _make_image(64, 64)
        cancel = threading.Event()
        params = {"method": "polynomial", "tile_size": 32, "preserve_median": False}
        result = _apply_step("background_removal", img, params, cancel)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# _PreviewTask — cancel after _apply_step returns (line 142)
# ---------------------------------------------------------------------------

class TestPreviewTaskCancelAfterResult:
    def test_cancel_after_apply_step_no_emit(self, qtbot):
        img = _make_image(32, 32)
        cancel = threading.Event()

        def cancelling_apply(step, thumb, params, evt):
            cancel.set()  # set cancel AFTER apply_step is called
            return thumb

        task = _PreviewTask("stretch", img, {}, cancel)
        results = []
        task.finished.connect(lambda r, s: results.append((r, s)))

        with patch("astroai.ui.preview_worker._apply_step", side_effect=cancelling_apply):
            task.run()

        assert len(results) == 0


# ---------------------------------------------------------------------------
# PreviewWorker — schedule_preview while running, _cancel_running with thread
# ---------------------------------------------------------------------------

class TestPreviewWorkerRunningState:
    def test_schedule_while_running_sets_cancel(self, qtbot):
        worker = PreviewWorker()
        img = _make_image(64, 64)
        worker.set_source_image(img)

        # Simulate a running thread
        mock_thread = MagicMock()
        mock_thread.isRunning.return_value = True
        worker._thread = mock_thread

        worker.schedule_preview("stretch", {"target_background": 0.3})

        assert worker._cancel_event.is_set()
        worker._thread = None  # cleanup

    def test_cancel_running_with_active_thread(self, qtbot):
        worker = PreviewWorker()
        mock_thread = MagicMock()
        mock_thread.isRunning.return_value = True
        worker._thread = mock_thread

        worker._cancel_running()

        # quit/wait called in _cancel_running AND _cleanup — at least once each
        assert mock_thread.quit.call_count >= 1
        assert mock_thread.wait.call_count >= 1


# ---------------------------------------------------------------------------
# PreviewWorker — full QThread-based execution via _run_preview (lines 207-230)
# ---------------------------------------------------------------------------

class TestPreviewWorkerFullRun:
    def test_run_preview_emits_preview_ready(self, qtbot):
        worker = PreviewWorker()
        worker.set_source_image(_make_image(64, 64))
        worker._pending = ("stretch", {
            "target_background": 0.25,
            "shadow_clipping_sigmas": -2.8,
            "linked_channels": True,
        })

        results = []
        worker.preview_ready.connect(lambda r, s: results.append((r, s)))

        with qtbot.waitSignal(worker.preview_ready, timeout=10000):
            worker._run_preview()

        assert len(results) == 1
        assert results[0][1] == "stretch"

    def test_run_preview_error_emits_preview_error(self, qtbot):
        worker = PreviewWorker()
        worker.set_source_image(_make_image(64, 64))
        worker._pending = ("stretch", {
            "target_background": 0.25,
            "shadow_clipping_sigmas": -2.8,
            "linked_channels": True,
        })

        errors = []
        worker.preview_error.connect(lambda msg: errors.append(msg))

        with patch(
            "astroai.ui.preview_worker._apply_step",
            side_effect=RuntimeError("boom"),
        ):
            with qtbot.waitSignal(worker.preview_error, timeout=10000):
                worker._run_preview()

        assert len(errors) == 1
        assert "boom" in errors[0]

    def test_run_preview_no_source_returns_early(self, qtbot):
        worker = PreviewWorker()
        worker._pending = ("stretch", {})
        # No source image set — should return without crash
        worker._run_preview()
        assert worker._thread is None

    def test_run_preview_no_pending_returns_early(self, qtbot):
        worker = PreviewWorker()
        worker.set_source_image(_make_image(64, 64))
        # _pending is None — should return without crash
        worker._run_preview()
        assert worker._thread is None

    def test_run_preview_diff_mode_emits_tuple(self, qtbot):
        worker = PreviewWorker()
        worker.set_source_image(_make_image(64, 64))
        worker.set_diff_enabled(True)
        worker._pending = ("curves", {
            "rgb_points": [(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)],
            "r_points": [(0.0, 0.0), (1.0, 1.0)],
            "g_points": [(0.0, 0.0), (1.0, 1.0)],
            "b_points": [(0.0, 0.0), (1.0, 1.0)],
        })

        results = []
        worker.preview_ready.connect(lambda r, s: results.append((r, s)))

        with qtbot.waitSignal(worker.preview_ready, timeout=10000):
            worker._run_preview()

        assert len(results) == 1
        result_pair, step = results[0]
        assert isinstance(result_pair, tuple)
        assert len(result_pair) == 2

    def test_cleanup_with_thread_and_task(self, qtbot):
        worker = PreviewWorker()
        mock_thread = MagicMock()
        mock_thread.isRunning.return_value = False
        mock_task = MagicMock()
        worker._thread = mock_thread
        worker._task = mock_task

        worker._cleanup()

        mock_thread.quit.assert_called_once()
        mock_thread.wait.assert_called_once_with(2000)
        mock_thread.deleteLater.assert_called_once()
        mock_task.deleteLater.assert_called_once()
        assert worker._thread is None
        assert worker._task is None
