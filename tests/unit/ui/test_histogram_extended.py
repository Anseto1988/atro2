"""Extended tests for HistogramWidget — paint event coverage."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.ui.widgets.histogram_widget import HistogramWidget


@pytest.fixture()
def histogram(qtbot) -> HistogramWidget:
    w = HistogramWidget()
    w.resize(256, 100)
    qtbot.addWidget(w)
    return w


class TestHistogramPaint:
    def test_paint_empty_no_crash(self, histogram: HistogramWidget) -> None:
        histogram.repaint()

    def test_paint_with_data(self, histogram: HistogramWidget) -> None:
        data = np.random.rand(100, 100).astype(np.float32)
        histogram.set_image_data(data)
        histogram.repaint()

    def test_paint_with_large_range(self, histogram: HistogramWidget) -> None:
        data = np.linspace(0.0, 65535.0, 10000).reshape(100, 100).astype(np.float32)
        histogram.set_image_data(data)
        histogram.repaint()

    def test_paint_after_clear(self, histogram: HistogramWidget) -> None:
        data = np.random.rand(50, 50).astype(np.float32)
        histogram.set_image_data(data)
        histogram.clear()
        histogram.repaint()

    def test_paint_tiny_widget(self, histogram: HistogramWidget) -> None:
        histogram.resize(2, 2)
        data = np.random.rand(10, 10).astype(np.float32)
        histogram.set_image_data(data)
        histogram.repaint()

    def test_bins_correct_count(self, histogram: HistogramWidget) -> None:
        data = np.random.rand(64, 64).astype(np.float32)
        histogram.set_image_data(data)
        assert histogram._bins is not None
        assert len(histogram._bins) == 256

    def test_max_count_positive(self, histogram: HistogramWidget) -> None:
        data = np.random.rand(32, 32).astype(np.float32)
        histogram.set_image_data(data)
        assert histogram._max_count >= 1.0

    def test_accessible_name(self, histogram: HistogramWidget) -> None:
        assert histogram.accessibleName() == "Histogramm"

    def test_tooltip(self, histogram: HistogramWidget) -> None:
        assert "logarithmisch" in histogram.toolTip()
