"""Tests for SkyOverlay widget – 100 % coverage target."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from astroai.astrometry.catalog import WcsSolution
from astroai.ui.widgets.sky_overlay import SkyOverlay, _format_dec, _format_ra


def _make_solution(**overrides) -> WcsSolution:
    defaults = dict(
        ra_center=180.0,
        dec_center=45.0,
        pixel_scale_arcsec=1.5,
        rotation_deg=0.0,
        fov_width_deg=2.0,
        fov_height_deg=1.5,
        cd_matrix=(1.0, 0.0, 0.0, 1.0),
        crpix1=500.0,
        crpix2=400.0,
    )
    defaults.update(overrides)
    return WcsSolution(**defaults)


@pytest.fixture()
def image_widget(qtbot):
    w = QWidget()
    w.setGeometry(0, 0, 800, 600)
    qtbot.addWidget(w)
    return w


@pytest.fixture()
def overlay(qtbot, image_widget):
    o = SkyOverlay(image_widget)
    qtbot.addWidget(o)
    return o


# ------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------


class TestInit:
    def test_creates_without_error(self, overlay):
        assert overlay is not None

    def test_parented_to_image_widget(self, overlay, image_widget):
        assert overlay.parent() is image_widget

    def test_transparent_for_mouse_events(self, overlay):
        assert overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def test_no_system_background(self, overlay):
        assert overlay.testAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def test_translucent_background(self, overlay):
        assert overlay.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def test_geometry_matches_image_widget(self, overlay, image_widget):
        assert overlay.geometry() == image_widget.geometry()

    def test_solution_initially_none(self, overlay):
        assert overlay._solution is None

    def test_grid_initially_visible(self, overlay):
        assert overlay._visible_grid is True

    def test_explicit_parent(self, qtbot, image_widget):
        parent = QWidget()
        qtbot.addWidget(parent)
        o = SkyOverlay(image_widget, parent=parent)
        qtbot.addWidget(o)
        assert o.parent() is parent


# ------------------------------------------------------------------
# set_solution
# ------------------------------------------------------------------


class TestSetSolution:
    def test_stores_solution(self, overlay):
        sol = _make_solution()
        overlay.set_solution(sol)
        assert overlay._solution is sol

    def test_stores_none(self, overlay):
        overlay.set_solution(_make_solution())
        overlay.set_solution(None)
        assert overlay._solution is None

    def test_calls_update(self, overlay):
        with patch.object(overlay, "update") as mock_update:
            overlay.set_solution(_make_solution())
            mock_update.assert_called_once()


# ------------------------------------------------------------------
# set_grid_visible
# ------------------------------------------------------------------


class TestSetGridVisible:
    def test_sets_false(self, overlay):
        overlay.set_grid_visible(False)
        assert overlay._visible_grid is False

    def test_sets_true(self, overlay):
        overlay.set_grid_visible(False)
        overlay.set_grid_visible(True)
        assert overlay._visible_grid is True

    def test_calls_update(self, overlay):
        with patch.object(overlay, "update") as mock_update:
            overlay.set_grid_visible(False)
            mock_update.assert_called_once()


# ------------------------------------------------------------------
# paintEvent
# ------------------------------------------------------------------


class TestPaintEvent:
    def test_no_solution_returns_early(self, overlay):
        with patch("astroai.ui.widgets.sky_overlay.QPainter") as mock_cls:
            overlay.paintEvent(None)
            mock_cls.assert_not_called()

    def test_with_solution_creates_painter(self, overlay):
        overlay.set_solution(_make_solution())
        overlay.resize(800, 600)
        with patch("astroai.ui.widgets.sky_overlay.QPainter") as mock_cls:
            mock_painter = MagicMock()
            mock_cls.return_value = mock_painter
            overlay.paintEvent(None)
            mock_cls.assert_called_once_with(overlay)
            mock_painter.setRenderHint.assert_called_once()
            mock_painter.end.assert_called_once()

    def test_grid_visible_draws_grid(self, overlay):
        overlay.set_solution(_make_solution())
        overlay.resize(800, 600)
        with patch("astroai.ui.widgets.sky_overlay.QPainter") as mock_cls:
            mock_painter = MagicMock()
            mock_cls.return_value = mock_painter
            overlay.paintEvent(None)
            assert mock_painter.drawLine.call_count >= _grid_line_count()

    def test_grid_hidden_skips_grid_lines(self, overlay):
        overlay.set_solution(_make_solution())
        overlay.set_grid_visible(False)
        overlay.resize(800, 600)
        with patch("astroai.ui.widgets.sky_overlay.QPainter") as mock_cls:
            mock_painter = MagicMock()
            mock_cls.return_value = mock_painter
            overlay.paintEvent(None)
            crosshair_lines = 2
            assert mock_painter.drawLine.call_count == crosshair_lines

    def test_painter_end_called_on_exception(self, overlay):
        overlay.set_solution(_make_solution())
        overlay.resize(800, 600)
        with patch("astroai.ui.widgets.sky_overlay.QPainter") as mock_cls:
            mock_painter = MagicMock()
            mock_cls.return_value = mock_painter
            mock_painter.setRenderHint.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError, match="boom"):
                overlay.paintEvent(None)
            mock_painter.end.assert_called_once()

    def test_corner_labels_drawn(self, overlay):
        overlay.set_solution(_make_solution())
        overlay.resize(800, 600)
        with patch("astroai.ui.widgets.sky_overlay.QPainter") as mock_cls:
            mock_painter = MagicMock()
            mock_cls.return_value = mock_painter
            overlay.paintEvent(None)
            text_calls = mock_painter.drawText.call_count
            assert text_calls >= 1


def _grid_line_count() -> int:
    from astroai.ui.widgets.sky_overlay import _GRID_STEPS
    return _GRID_STEPS * 2 + 2  # RA lines + Dec lines + crosshair


# ------------------------------------------------------------------
# _format_ra
# ------------------------------------------------------------------


class TestFormatRa:
    def test_zero(self):
        assert _format_ra(0.0) == "00h00m00.0s"

    def test_180_degrees(self):
        assert _format_ra(180.0) == "12h00m00.0s"

    def test_negative_wraps(self):
        result = _format_ra(-15.0)
        assert result == _format_ra(345.0)

    def test_360_wraps_to_zero(self):
        assert _format_ra(360.0) == "00h00m00.0s"

    def test_fractional(self):
        result = _format_ra(45.0)
        assert result.startswith("03h")


# ------------------------------------------------------------------
# _format_dec
# ------------------------------------------------------------------


class TestFormatDec:
    def test_zero(self):
        assert _format_dec(0.0).startswith("+")

    def test_positive(self):
        result = _format_dec(45.0)
        assert result.startswith("+45")

    def test_negative(self):
        result = _format_dec(-30.5)
        assert result.startswith("-30")

    def test_precise_value(self):
        result = _format_dec(90.0)
        assert result == "+90°00'00.0\""

    def test_fractional_minutes(self):
        result = _format_dec(45.5)
        assert "30'" in result
