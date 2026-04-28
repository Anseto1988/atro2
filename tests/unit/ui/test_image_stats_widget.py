"""Tests for ImageStatsWidget — per-channel mean, median, std, SNR, min, max."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.ui.widgets.image_stats_widget import ImageStatsWidget, _HEADERS, _PLACEHOLDER


@pytest.fixture()
def widget(qtbot):
    w = ImageStatsWidget()
    qtbot.addWidget(w)
    return w


def _cell(widget: ImageStatsWidget, row: int, col: int) -> str:
    item = widget._table.item(row, col)
    assert item is not None, f"No item at ({row}, {col})"
    return item.text()


class TestHeaders:
    def test_column_count(self, widget):
        assert widget._table.columnCount() == 7

    def test_header_labels(self, widget):
        expected = ["Kanal", "Mittel", "Median", "Std", "SNR", "Min", "Max"]
        actual = [
            widget._table.horizontalHeaderItem(i).text()
            for i in range(widget._table.columnCount())
        ]
        assert actual == expected


class TestClear:
    def test_clear_shows_one_row(self, widget):
        widget.clear()
        assert widget._table.rowCount() == 1

    def test_clear_shows_placeholder_for_stats(self, widget):
        widget.clear()
        for col in range(1, len(_HEADERS)):
            assert _cell(widget, 0, col) == _PLACEHOLDER

    def test_clear_channel_label(self, widget):
        widget.clear()
        assert _cell(widget, 0, 0) == "L"


class TestMonoImage:
    def _make_mono(self) -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.random((16, 16), dtype=np.float64).astype(np.float32)

    def test_row_count_is_one(self, widget):
        widget.set_image_data(self._make_mono())
        assert widget._table.rowCount() == 1

    def test_channel_label(self, widget):
        widget.set_image_data(self._make_mono())
        assert _cell(widget, 0, 0) == "L"

    def test_mean_not_placeholder(self, widget):
        widget.set_image_data(self._make_mono())
        assert _cell(widget, 0, 1) != _PLACEHOLDER

    def test_median_not_placeholder(self, widget):
        widget.set_image_data(self._make_mono())
        assert _cell(widget, 0, 2) != _PLACEHOLDER

    def test_std_not_placeholder(self, widget):
        widget.set_image_data(self._make_mono())
        assert _cell(widget, 0, 3) != _PLACEHOLDER

    def test_snr_not_placeholder(self, widget):
        widget.set_image_data(self._make_mono())
        assert _cell(widget, 0, 4) != _PLACEHOLDER

    def test_mean_value_correct(self, widget):
        arr = self._make_mono()
        widget.set_image_data(arr)
        expected = f"{float(np.mean(arr)):.4f}"
        assert _cell(widget, 0, 1) == expected

    def test_median_value_correct(self, widget):
        arr = self._make_mono()
        widget.set_image_data(arr)
        expected = f"{float(np.median(arr)):.4f}"
        assert _cell(widget, 0, 2) == expected

    def test_snr_value_correct(self, widget):
        arr = self._make_mono()
        widget.set_image_data(arr)
        std = float(np.std(arr))
        expected = f"{float(np.mean(arr)) / std:.1f}"
        assert _cell(widget, 0, 4) == expected

    def test_min_max_correct(self, widget):
        arr = self._make_mono()
        widget.set_image_data(arr)
        assert _cell(widget, 0, 5) == f"{float(np.min(arr)):.4f}"
        assert _cell(widget, 0, 6) == f"{float(np.max(arr)):.4f}"


class TestColorImage:
    def _make_color(self) -> np.ndarray:
        rng = np.random.default_rng(13)
        return rng.random((16, 16, 3), dtype=np.float64).astype(np.float32)

    def test_row_count_is_three(self, widget):
        widget.set_image_data(self._make_color())
        assert widget._table.rowCount() == 3

    def test_channel_labels(self, widget):
        widget.set_image_data(self._make_color())
        assert _cell(widget, 0, 0) == "R"
        assert _cell(widget, 1, 0) == "G"
        assert _cell(widget, 2, 0) == "B"

    def test_per_channel_median(self, widget):
        arr = self._make_color()
        widget.set_image_data(arr)
        for i in range(3):
            expected = f"{float(np.median(arr[:, :, i])):.4f}"
            assert _cell(widget, i, 2) == expected

    def test_per_channel_snr(self, widget):
        arr = self._make_color()
        widget.set_image_data(arr)
        for i in range(3):
            ch = arr[:, :, i].ravel()
            std = float(np.std(ch))
            expected = f"{float(np.mean(ch)) / std:.1f}"
            assert _cell(widget, i, 4) == expected


class TestConstantArraySNR:
    def test_constant_mono_snr_is_placeholder(self, widget):
        arr = np.full((8, 8), 0.5, dtype=np.float32)
        widget.set_image_data(arr)
        assert _cell(widget, 0, 4) == _PLACEHOLDER

    def test_constant_color_snr_is_placeholder(self, widget):
        arr = np.full((8, 8, 3), 0.5, dtype=np.float32)
        widget.set_image_data(arr)
        for row in range(3):
            assert _cell(widget, row, 4) == _PLACEHOLDER


class TestInvalidShape:
    def test_4d_array_clears(self, widget):
        widget.set_image_data(np.zeros((8, 8, 3)))
        arr_4d = np.zeros((4, 4, 4, 3), dtype=np.float32)
        widget.set_image_data(arr_4d)
        assert widget._table.rowCount() == 1
        assert _cell(widget, 0, 1) == _PLACEHOLDER

    def test_wrong_channel_count_clears(self, widget):
        widget.set_image_data(np.zeros((8, 8, 3)))
        arr_2ch = np.zeros((8, 8, 2), dtype=np.float32)
        widget.set_image_data(arr_2ch)
        assert _cell(widget, 0, 1) == _PLACEHOLDER
