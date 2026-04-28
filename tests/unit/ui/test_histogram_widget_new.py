"""Tests for HistogramWidget."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.ui.widgets.histogram_widget import HistogramWidget


@pytest.fixture()
def widget(qtbot):
    w = HistogramWidget()
    w.resize(200, 100)
    qtbot.addWidget(w)
    w.show()
    return w


class TestInitialState:
    def test_creates_without_error(self, widget):
        assert widget is not None

    def test_bins_none_initially(self, widget):
        assert widget._bins is None

    def test_minimum_size(self, widget):
        assert widget.minimumHeight() == 80
        assert widget.minimumWidth() == 120


class TestSetImageData:
    def test_set_image_data_populates_bins(self, widget):
        arr = np.random.default_rng(1).random((16, 16), dtype=np.float64).astype(np.float32)
        widget.set_image_data(arr)
        assert widget._bins is not None
        assert len(widget._bins) == 256

    def test_max_count_set_correctly(self, widget):
        arr = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
        widget.set_image_data(arr)
        assert widget._max_count >= 1.0

    def test_constant_image_clears_bins(self, widget):
        arr = np.full((8, 8), 0.5, dtype=np.float32)
        widget.set_image_data(arr)
        assert widget._bins is None

    def test_color_image_3d_flattened(self, widget):
        arr = np.random.default_rng(2).random((8, 8, 3), dtype=np.float64).astype(np.float32)
        widget.set_image_data(arr)
        assert widget._bins is not None

    def test_set_image_data_triggers_repaint(self, widget, qtbot):
        arr = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        widget.set_image_data(arr)
        assert widget._bins is not None


class TestClear:
    def test_clear_removes_bins(self, widget):
        arr = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        widget.set_image_data(arr)
        assert widget._bins is not None
        widget.clear()
        assert widget._bins is None


class TestPaintEvent:
    def test_paint_with_no_bins_no_crash(self, widget):
        widget._bins = None
        widget.grab()

    def test_paint_with_bins_no_crash(self, widget):
        arr = np.linspace(0, 1, 256, dtype=np.float32)
        widget.set_image_data(arr)
        widget.grab()

    def test_paint_small_widget_no_crash(self, qtbot):
        w = HistogramWidget()
        qtbot.addWidget(w)
        arr = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        w.set_image_data(arr)
        w.show()
        w.grab()

    def test_paint_with_uniform_log_bins_no_crash(self, widget):
        arr = np.zeros(256, dtype=np.float32)
        arr[0] = 0.0
        arr[-1] = 1.0
        widget.set_image_data(arr)
        widget.grab()
