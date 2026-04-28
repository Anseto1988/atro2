"""Tests for HistogramView, HistogramWorker, and _HistogramCanvas."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from astroai.ui.widgets.live_histogram_view import (
    _ComputeWorker,
    _HistogramCanvas,
    _HistogramData,
    _NUM_BINS,
    HistogramView,
    HistogramWorker,
)


# ---------------------------------------------------------------------------
# _HistogramData
# ---------------------------------------------------------------------------

class TestHistogramData:
    def test_slots(self) -> None:
        r = np.zeros(_NUM_BINS)
        g = np.ones(_NUM_BINS)
        b = np.zeros(_NUM_BINS)
        lum = np.zeros(_NUM_BINS)
        hd = _HistogramData(r=r, g=g, b=b, lum=lum)
        assert hd.r is r
        assert hd.g is g
        assert hd.b is b
        assert hd.lum is lum


# ---------------------------------------------------------------------------
# _ComputeWorker
# ---------------------------------------------------------------------------

class TestComputeWorker:
    @pytest.fixture()
    def worker(self) -> _ComputeWorker:
        return _ComputeWorker()

    def test_rgb_image_emits_result(self, worker: _ComputeWorker, qtbot) -> None:
        img = np.random.rand(64, 64, 3).astype(np.float32)
        results: list[_HistogramData] = []
        worker.result_ready.connect(lambda d: results.append(d))

        with qtbot.waitSignal(worker.result_ready, timeout=3000):
            worker.compute(img)

        assert len(results) == 1
        hd = results[0]
        assert isinstance(hd, _HistogramData)
        assert len(hd.r) == _NUM_BINS
        assert len(hd.g) == _NUM_BINS
        assert len(hd.b) == _NUM_BINS
        assert len(hd.lum) == _NUM_BINS

    def test_grayscale_channels_are_equal(self, worker: _ComputeWorker, qtbot) -> None:
        img = np.random.rand(32, 32).astype(np.float32)
        results: list[_HistogramData] = []
        worker.result_ready.connect(lambda d: results.append(d))

        with qtbot.waitSignal(worker.result_ready, timeout=3000):
            worker.compute(img)

        hd = results[0]
        np.testing.assert_array_equal(hd.r, hd.g)
        np.testing.assert_array_equal(hd.g, hd.b)
        np.testing.assert_array_equal(hd.b, hd.lum)

    def test_flat_image_gives_zeros(self, worker: _ComputeWorker, qtbot) -> None:
        img = np.full((10, 10), 0.5, dtype=np.float32)
        results: list[_HistogramData] = []
        worker.result_ready.connect(lambda d: results.append(d))

        with qtbot.waitSignal(worker.result_ready, timeout=3000):
            worker.compute(img)

        hd = results[0]
        assert float(np.sum(hd.r)) == 0.0
        assert float(np.sum(hd.lum)) == 0.0

    def test_4channel_uses_first_3(self, worker: _ComputeWorker, qtbot) -> None:
        img = np.random.rand(32, 32, 4).astype(np.float32)
        results: list[_HistogramData] = []
        worker.result_ready.connect(lambda d: results.append(d))

        with qtbot.waitSignal(worker.result_ready, timeout=3000):
            worker.compute(img)

        hd = results[0]
        assert len(hd.r) == _NUM_BINS


# ---------------------------------------------------------------------------
# HistogramWorker
# ---------------------------------------------------------------------------

class TestHistogramWorker:
    @pytest.fixture()
    def worker(self, qtbot) -> HistogramWorker:
        w = HistogramWorker()
        yield w
        w.stop()

    def test_thread_starts(self, worker: HistogramWorker) -> None:
        assert worker._thread.isRunning()

    def test_compute_emits_result(self, worker: HistogramWorker, qtbot) -> None:
        img = np.random.rand(64, 64, 3).astype(np.float32)
        results: list = []
        worker.result_ready.connect(lambda d: results.append(d))

        with qtbot.waitSignal(worker.result_ready, timeout=5000):
            worker.compute(img)

        assert len(results) == 1
        assert isinstance(results[0], _HistogramData)

    def test_stop_quits_thread(self, worker: HistogramWorker) -> None:
        worker.stop()
        assert not worker._thread.isRunning()


# ---------------------------------------------------------------------------
# _HistogramCanvas
# ---------------------------------------------------------------------------

@pytest.fixture()
def canvas(qtbot) -> _HistogramCanvas:
    w = _HistogramCanvas()
    w.resize(300, 200)
    qtbot.addWidget(w)
    return w


@pytest.fixture()
def sample_data() -> _HistogramData:
    img = np.random.rand(64, 64, 3).astype(np.float32)
    r = img[:, :, 0].ravel()
    g = img[:, :, 1].ravel()
    b = img[:, :, 2].ravel()
    lum = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).ravel()

    def _hist(ch: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(ch, bins=_NUM_BINS, range=(0.0, 1.0))
        return counts.astype(np.float64)

    return _HistogramData(r=_hist(r), g=_hist(g), b=_hist(b), lum=_hist(lum))


class TestHistogramCanvas:
    def test_paint_empty_no_crash(self, canvas: _HistogramCanvas) -> None:
        canvas.repaint()

    def test_paint_with_data_no_crash(self, canvas: _HistogramCanvas, sample_data: _HistogramData) -> None:
        canvas.set_data(sample_data)
        canvas.repaint()

    def test_paint_linear_scale(self, canvas: _HistogramCanvas, sample_data: _HistogramData) -> None:
        canvas.set_data(sample_data)
        canvas.set_log_scale(False)
        canvas.repaint()

    def test_clear_removes_data(self, canvas: _HistogramCanvas, sample_data: _HistogramData) -> None:
        canvas.set_data(sample_data)
        canvas.clear()
        assert canvas._data is None

    def test_log_scale_toggle(self, canvas: _HistogramCanvas) -> None:
        canvas.set_log_scale(False)
        assert not canvas._log_scale
        canvas.set_log_scale(True)
        assert canvas._log_scale

    def test_tiny_widget_no_crash(self, qtbot) -> None:
        w = _HistogramCanvas()
        w.resize(1, 1)
        qtbot.addWidget(w)
        w.repaint()

    def test_set_data_triggers_update(self, canvas: _HistogramCanvas, sample_data: _HistogramData) -> None:
        canvas.set_data(sample_data)
        assert canvas._data is sample_data


# ---------------------------------------------------------------------------
# HistogramView
# ---------------------------------------------------------------------------

@pytest.fixture()
def view(qtbot) -> HistogramView:
    w = HistogramView()
    w.resize(300, 200)
    qtbot.addWidget(w)
    yield w
    w._worker.stop()


class TestHistogramView:
    def test_accessible_name(self, view: HistogramView) -> None:
        assert view.accessibleName() == "Live-Histogramm"

    def test_minimum_width(self, view: HistogramView) -> None:
        assert view.minimumWidth() == 200

    def test_log_checkbox_default(self, view: HistogramView) -> None:
        assert view._log_cb.isChecked()

    def test_log_checkbox_updates_canvas(self, view: HistogramView) -> None:
        view._log_cb.setChecked(False)
        assert not view._canvas._log_scale
        view._log_cb.setChecked(True)
        assert view._canvas._log_scale

    def test_set_image_data_rgb(self, view: HistogramView, qtbot) -> None:
        img = np.random.rand(64, 64, 3).astype(np.float32)
        with qtbot.waitSignal(view._worker.result_ready, timeout=5000):
            view.set_image_data(img)
        assert view._canvas._data is not None

    def test_set_image_data_grayscale(self, view: HistogramView, qtbot) -> None:
        img = np.random.rand(32, 32).astype(np.float32)
        with qtbot.waitSignal(view._worker.result_ready, timeout=5000):
            view.set_image_data(img)
        assert view._canvas._data is not None

    def test_set_image_data_non_ndarray_ignored(self, view: HistogramView) -> None:
        view.set_image_data("not-an-array")
        assert view._canvas._data is None

    def test_set_image_data_none_ignored(self, view: HistogramView) -> None:
        view.set_image_data(None)
        assert view._canvas._data is None

    def test_clear(self, view: HistogramView, qtbot) -> None:
        img = np.random.rand(64, 64, 3).astype(np.float32)
        with qtbot.waitSignal(view._worker.result_ready, timeout=5000):
            view.set_image_data(img)
        view.clear()
        assert view._canvas._data is None

    def test_paint_after_data(self, view: HistogramView, qtbot) -> None:
        img = np.random.rand(64, 64, 3).astype(np.float32)
        with qtbot.waitSignal(view._worker.result_ready, timeout=5000):
            view.set_image_data(img)
        view.repaint()

    def test_worker_is_child(self, view: HistogramView) -> None:
        assert view._worker.parent() is view


# ---------------------------------------------------------------------------
# PipelineModel.histogram_changed integration
# ---------------------------------------------------------------------------

class TestPipelineModelHistogramSignal:
    def test_histogram_changed_signal_exists(self) -> None:
        from astroai.ui.models import PipelineModel
        model = PipelineModel()
        assert hasattr(model, "histogram_changed")

    def test_histogram_changed_emits_ndarray(self, qtbot) -> None:
        from astroai.ui.models import PipelineModel
        model = PipelineModel()
        received: list = []
        model.histogram_changed.connect(lambda d: received.append(d))
        img = np.random.rand(32, 32, 3).astype(np.float32)
        with qtbot.waitSignal(model.histogram_changed, timeout=1000):
            model.histogram_changed.emit(img)
        assert len(received) == 1
        assert isinstance(received[0], np.ndarray)
