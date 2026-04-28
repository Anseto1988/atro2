"""Tests for CalibrationWorker Qt threading module.

Coverage targets: lines 30-34, 37-48, 51, 68, 77-89, 92-93, 96-97, 100, 103-110
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.core.calibration.matcher import CalibrationLibrary
from astroai.core.calibration.metrics import BenchmarkMetrics, BenchmarkBackend
from astroai.core.calibration.worker import CalibrationWorker, _WorkerRunner
from astroai.core.io.fits_io import ImageMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta() -> ImageMetadata:
    return ImageMetadata(exposure=120.0, gain_iso=800, temperature=-10.0, width=64, height=64)


def _frames() -> list:
    rng = np.random.default_rng(0)
    return [rng.random((64, 64), dtype=np.float32) for _ in range(2)]


def _make_mock_engine(results=None) -> MagicMock:
    engine = MagicMock()
    engine.calibrate_batch_gpu.return_value = results if results is not None else []
    return engine


# ---------------------------------------------------------------------------
# _WorkerRunner.__init__  (lines 30-34)
# ---------------------------------------------------------------------------

class TestWorkerRunnerInit:
    def test_stores_frames(self) -> None:
        frames = _frames()
        runner = _WorkerRunner(frames, _meta(), CalibrationLibrary.empty(), None)
        assert runner._frames is frames

    def test_stores_meta(self) -> None:
        meta = _meta()
        runner = _WorkerRunner(_frames(), meta, CalibrationLibrary.empty(), None)
        assert runner._meta is meta

    def test_stores_library(self) -> None:
        lib = CalibrationLibrary.empty()
        runner = _WorkerRunner(_frames(), _meta(), lib, None)
        assert runner._library is lib

    def test_stores_load_data(self) -> None:
        load_fn = MagicMock()
        runner = _WorkerRunner(_frames(), _meta(), CalibrationLibrary.empty(), load_fn)
        assert runner._load_data is load_fn

    def test_load_data_none_is_accepted(self) -> None:
        runner = _WorkerRunner(_frames(), _meta(), CalibrationLibrary.empty(), None)
        assert runner._load_data is None


# ---------------------------------------------------------------------------
# _WorkerRunner.run()  (lines 37-48)
# ---------------------------------------------------------------------------

class TestWorkerRunnerRun:
    def test_run_success_emits_finished(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        expected = [np.zeros((64, 64), dtype=np.float32)]
        mock_engine = _make_mock_engine(expected)
        runner = _WorkerRunner(_frames(), _meta(), CalibrationLibrary.empty(), None)
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=mock_engine,
        ):
            with qtbot.waitSignal(runner.finished, timeout=2000) as blocker:
                runner.run()
        assert blocker.args[0] is expected

    def test_run_error_emits_error_signal(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        mock_engine = MagicMock()
        mock_engine.calibrate_batch_gpu.side_effect = RuntimeError("GPU exploded")
        runner = _WorkerRunner(_frames(), _meta(), CalibrationLibrary.empty(), None)
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=mock_engine,
        ):
            with qtbot.waitSignal(runner.error, timeout=2000) as blocker:
                runner.run()
        assert "GPU exploded" in blocker.args[0]

    def test_run_passes_load_data_to_engine(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        mock_engine = _make_mock_engine([])
        load_fn = MagicMock()
        runner = _WorkerRunner(_frames(), _meta(), CalibrationLibrary.empty(), load_fn)
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=mock_engine,
        ):
            with qtbot.waitSignal(runner.finished, timeout=2000):
                runner.run()
        call_kwargs = mock_engine.calibrate_batch_gpu.call_args
        ld = call_kwargs.kwargs.get("load_data") or (
            call_kwargs.args[3] if len(call_kwargs.args) > 3 else None
        )
        assert ld is load_fn

    def test_run_does_not_emit_finished_on_error(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        mock_engine = MagicMock()
        mock_engine.calibrate_batch_gpu.side_effect = ValueError("bad")
        runner = _WorkerRunner(_frames(), _meta(), CalibrationLibrary.empty(), None)
        finished_received: list = []
        runner.finished.connect(finished_received.append)
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=mock_engine,
        ):
            with qtbot.waitSignal(runner.error, timeout=2000):
                runner.run()
        assert len(finished_received) == 0


# ---------------------------------------------------------------------------
# _WorkerRunner._emit_metrics  (line 51)
# ---------------------------------------------------------------------------

class TestWorkerRunnerEmitMetrics:
    def _make_metrics(self) -> BenchmarkMetrics:
        return BenchmarkMetrics(
            backend=BenchmarkBackend.CPU,
            device_name="CPU",
            frames_per_second=10.0,
            speedup_factor=1.0,
            current_frame=1,
            total_frames=1,
            eta_seconds=0.0,
        )

    def test_emit_metrics_forwards_object(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        runner = _WorkerRunner(_frames(), _meta(), CalibrationLibrary.empty(), None)
        m = self._make_metrics()
        with qtbot.waitSignal(runner.metrics, timeout=1000) as blocker:
            runner._emit_metrics(m)
        assert blocker.args[0] is m

    def test_on_metrics_callback_invokes_emit_metrics(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        received: list = []
        m = self._make_metrics()

        def fake_calibrate(frames, meta, lib, *, load_data=None, on_metrics=None):  # type: ignore[no-untyped-def]
            if on_metrics is not None:
                on_metrics(m)
            return []

        mock_engine = MagicMock()
        mock_engine.calibrate_batch_gpu.side_effect = fake_calibrate
        runner = _WorkerRunner(_frames(), _meta(), CalibrationLibrary.empty(), None)
        runner.metrics.connect(received.append)
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=mock_engine,
        ):
            with qtbot.waitSignal(runner.finished, timeout=2000):
                runner.run()
        assert len(received) == 1
        assert received[0] is m


# ---------------------------------------------------------------------------
# CalibrationWorker.is_running  (line 68)
# ---------------------------------------------------------------------------

class TestCalibrationWorkerIsRunning:
    def test_is_running_false_initially(self) -> None:
        worker = CalibrationWorker()
        assert worker.is_running is False

    def test_is_running_false_when_thread_none(self) -> None:
        worker = CalibrationWorker()
        worker._thread = None
        assert worker.is_running is False

    def test_is_running_true_when_thread_active(self) -> None:
        from PySide6.QtCore import QThread
        worker = CalibrationWorker()
        fake_thread = MagicMock(spec=QThread)
        fake_thread.isRunning.return_value = True
        worker._thread = fake_thread
        assert worker.is_running is True


# ---------------------------------------------------------------------------
# CalibrationWorker.start()  (lines 77-89)
# ---------------------------------------------------------------------------

class TestCalibrationWorkerStart:
    def test_start_emits_finished_signal(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        expected_results = [np.zeros((64, 64), dtype=np.float32)]
        mock_engine = _make_mock_engine(expected_results)
        worker = CalibrationWorker()
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=mock_engine,
        ):
            with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
                worker.start(_frames(), _meta(), CalibrationLibrary.empty())
        assert blocker.args[0] is expected_results

    def test_start_emits_error_signal(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        bad_engine = MagicMock()
        bad_engine.calibrate_batch_gpu.side_effect = RuntimeError("VRAM OOM")
        worker = CalibrationWorker()
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=bad_engine,
        ):
            with qtbot.waitSignal(worker.error, timeout=5000) as blocker:
                worker.start(_frames(), _meta(), CalibrationLibrary.empty())
        assert "VRAM OOM" in blocker.args[0]

    def test_start_emits_metrics_signal(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        m = BenchmarkMetrics(
            backend=BenchmarkBackend.CPU,
            device_name="CPU",
            frames_per_second=1.0,
            speedup_factor=1.0,
            current_frame=1,
            total_frames=1,
            eta_seconds=0.0,
        )

        def fake_calibrate(frames, meta, lib, *, load_data=None, on_metrics=None):  # type: ignore[no-untyped-def]
            if on_metrics is not None:
                on_metrics(m)
            return []

        engine_mock = MagicMock()
        engine_mock.calibrate_batch_gpu.side_effect = fake_calibrate
        worker = CalibrationWorker()
        received: list = []
        worker.metrics.connect(received.append)
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=engine_mock,
        ):
            with qtbot.waitSignal(worker.finished, timeout=5000):
                worker.start(_frames(), _meta(), CalibrationLibrary.empty())
        assert len(received) >= 1
        assert received[0] is m

    def test_start_when_already_running_is_noop(self) -> None:
        from PySide6.QtCore import QThread
        worker = CalibrationWorker()
        fake_thread = MagicMock(spec=QThread)
        fake_thread.isRunning.return_value = True
        worker._thread = fake_thread
        worker.start(_frames(), _meta(), CalibrationLibrary.empty())
        assert worker._thread is fake_thread
        fake_thread.start.assert_not_called()

    def test_start_cleans_up_after_finished(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        mock_engine = _make_mock_engine([])
        worker = CalibrationWorker()
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=mock_engine,
        ):
            with qtbot.waitSignal(worker.finished, timeout=5000):
                worker.start(_frames(), _meta(), CalibrationLibrary.empty())
        assert worker._thread is None
        assert worker._runner is None


# ---------------------------------------------------------------------------
# CalibrationWorker._cleanup()  (lines 103-110)
# ---------------------------------------------------------------------------

class TestCalibrationWorkerCleanup:
    def test_cleanup_nullifies_thread_and_runner(self) -> None:
        from PySide6.QtCore import QThread
        worker = CalibrationWorker()
        fake_thread = MagicMock(spec=QThread)
        fake_runner = MagicMock()
        worker._thread = fake_thread
        worker._runner = fake_runner
        worker._cleanup()
        assert worker._thread is None
        assert worker._runner is None

    def test_cleanup_calls_thread_quit_and_wait(self) -> None:
        from PySide6.QtCore import QThread
        worker = CalibrationWorker()
        fake_thread = MagicMock(spec=QThread)
        worker._thread = fake_thread
        worker._runner = MagicMock()
        worker._cleanup()
        fake_thread.quit.assert_called_once()
        fake_thread.wait.assert_called_once_with(5000)

    def test_cleanup_calls_delete_later_on_both(self) -> None:
        from PySide6.QtCore import QThread
        worker = CalibrationWorker()
        fake_thread = MagicMock(spec=QThread)
        fake_runner = MagicMock()
        worker._thread = fake_thread
        worker._runner = fake_runner
        worker._cleanup()
        fake_thread.deleteLater.assert_called_once()
        fake_runner.deleteLater.assert_called_once()

    def test_cleanup_safe_when_both_none(self) -> None:
        worker = CalibrationWorker()
        worker._thread = None
        worker._runner = None
        worker._cleanup()

    def test_cleanup_safe_when_only_runner_none(self) -> None:
        from PySide6.QtCore import QThread
        worker = CalibrationWorker()
        fake_thread = MagicMock(spec=QThread)
        worker._thread = fake_thread
        worker._runner = None
        worker._cleanup()
        assert worker._thread is None
        assert worker._runner is None

    def test_cleanup_called_after_finished(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        mock_engine = _make_mock_engine([])
        worker = CalibrationWorker()
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=mock_engine,
        ):
            with qtbot.waitSignal(worker.finished, timeout=5000):
                worker.start(_frames(), _meta(), CalibrationLibrary.empty())
        assert worker._thread is None
        assert worker._runner is None

    def test_cleanup_called_after_error(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        bad_engine = MagicMock()
        bad_engine.calibrate_batch_gpu.side_effect = RuntimeError("fail")
        worker = CalibrationWorker()
        with patch(
            "astroai.core.calibration.worker.GPUCalibrationEngine",
            return_value=bad_engine,
        ):
            with qtbot.waitSignal(worker.error, timeout=5000):
                worker.start(_frames(), _meta(), CalibrationLibrary.empty())
        assert worker._thread is None
        assert worker._runner is None
