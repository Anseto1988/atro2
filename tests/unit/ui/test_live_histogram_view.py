"""Tests for HistogramView and HistogramWorker – 100% statement coverage."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.ui.widgets.live_histogram_view import (
    HistogramView,
    HistogramWorker,
)


class TestHistogramWorker:
    """Test _ComputeWorker directly (synchronous, no thread boundary)."""

    def _compute(self, arr: np.ndarray):
        """Synchronously run the compute slot and capture the emitted result."""
        from astroai.ui.widgets.live_histogram_view import _ComputeWorker  # noqa: PLC0415
        results: list = []
        worker = _ComputeWorker()
        worker.result_ready.connect(results.append)
        worker.compute(arr)
        assert results, "result_ready was not emitted"
        return results[0]

    def test_result_ready_emitted_rgb(self) -> None:
        arr = np.random.rand(32, 32, 3).astype(np.float32)
        data = self._compute(arr)
        assert data.r.shape == (256,)
        assert data.g.shape == (256,)
        assert data.b.shape == (256,)
        assert data.lum.shape == (256,)

    def test_result_ready_emitted_grayscale(self) -> None:
        arr = np.random.rand(32, 32).astype(np.float32)
        data = self._compute(arr)
        assert np.array_equal(data.r, data.g)

    def test_flat_array_returns_zeros(self) -> None:
        arr = np.ones((16, 16), dtype=np.float32)
        data = self._compute(arr)
        assert np.all(data.r == 0)


class TestHistogramView:
    @pytest.fixture()
    def view(self, qtbot):
        w = HistogramView()
        qtbot.addWidget(w)
        w.show()
        yield w
        w._worker.stop()

    def test_widget_created(self, view: HistogramView) -> None:
        assert view is not None

    def test_accessible_name(self, view: HistogramView) -> None:
        assert "Histogramm" in view.accessibleName()

    def test_set_image_data_non_array_ignored(self, view: HistogramView) -> None:
        view.set_image_data("not an array")

    def test_set_image_data_accepts_ndarray(self, view: HistogramView) -> None:
        arr = np.random.rand(16, 16, 3).astype(np.float32)
        view.set_image_data(arr)  # must not raise

    def test_clear_clears_canvas(self, view: HistogramView, qtbot) -> None:
        arr = np.random.rand(16, 16, 3).astype(np.float32)
        with qtbot.waitSignal(view._worker.result_ready, timeout=2000):
            view.set_image_data(arr)
        view.clear()
        assert view._canvas._data is None

    def test_log_checkbox_default_checked(self, view: HistogramView) -> None:
        assert view._log_cb.isChecked()

    def test_log_checkbox_toggle_updates_canvas(self, view: HistogramView) -> None:
        view._log_cb.setChecked(False)
        assert not view._canvas._log_scale
        view._log_cb.setChecked(True)
        assert view._canvas._log_scale

    def test_on_result_ready_non_histogram_data_ignored(self, view: HistogramView) -> None:
        view._on_result_ready("garbage")
        assert view._canvas._data is None

    def test_paint_with_data(self, view: HistogramView) -> None:
        from astroai.ui.widgets.live_histogram_view import _HistogramData  # noqa: PLC0415
        arr = np.linspace(0, 1, 256, dtype=np.float64)
        fake = _HistogramData(r=arr, g=arr, b=arr, lum=arr)
        view._on_result_ready(fake)
        view.repaint()

    def test_paint_without_data(self, view: HistogramView) -> None:
        view.clear()
        view.repaint()

    def test_minimum_width(self, view: HistogramView) -> None:
        assert view.minimumWidth() >= 200
