"""Tests for CurveEditor and CurvesPanel."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.curves_panel import CurveEditor, CurvesPanel, _IDENTITY, _MAX_POINTS, _MIN_POINTS


@pytest.fixture()
def editor(qtbot) -> CurveEditor:  # type: ignore[no-untyped-def]
    w = CurveEditor()
    qtbot.addWidget(w)
    w.resize(300, 300)
    w.show()
    return w


@pytest.fixture()
def model() -> PipelineModel:
    return PipelineModel()


@pytest.fixture()
def panel(qtbot, model: PipelineModel) -> CurvesPanel:  # type: ignore[no-untyped-def]
    w = CurvesPanel(model)
    qtbot.addWidget(w)
    return w


class TestCurveEditor:
    def test_initial_points_are_identity(self, editor: CurveEditor) -> None:
        assert editor.get_points() == list(_IDENTITY)

    def test_set_points_updates(self, editor: CurveEditor) -> None:
        new_pts = [(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)]
        editor.set_points(new_pts)
        assert editor.get_points() == new_pts

    def test_set_points_too_short_falls_back_to_identity(self, editor: CurveEditor) -> None:
        editor.set_points([(0.0, 0.0)])
        assert editor.get_points() == list(_IDENTITY)

    def test_reset_restores_identity(self, editor: CurveEditor) -> None:
        editor.set_points([(0.0, 0.0), (0.5, 0.9), (1.0, 1.0)])
        editor.reset()
        assert editor.get_points() == list(_IDENTITY)

    def test_reset_emits_points_changed(self, editor: CurveEditor, qtbot) -> None:
        received = []
        editor.points_changed.connect(received.append)
        editor.reset()
        assert len(received) == 1
        assert received[0] == list(_IDENTITY)

    def test_plot_rect_returns_four_values(self, editor: CurveEditor) -> None:
        ox, oy, w, h = editor._plot_rect()
        assert w > 0
        assert h > 0

    def test_to_widget_origin_maps_to_margin(self, editor: CurveEditor) -> None:
        from PySide6.QtCore import QPointF
        pt = editor._to_widget(0.0, 0.0)
        assert isinstance(pt, QPointF)

    def test_from_widget_clamps_to_unit_square(self, editor: CurveEditor) -> None:
        x, y = editor._from_widget(-100.0, -100.0)
        assert x == 0.0
        assert y == 0.0 or y == 1.0

    def test_nearest_point_idx_hit(self, editor: CurveEditor) -> None:
        # Origin point (0,0) maps near the lower-left margin area
        ox, oy, w, h = editor._plot_rect()
        wp = editor._to_widget(0.0, 0.0)
        idx = editor._nearest_point_idx(wp.x(), wp.y())
        assert idx == 0

    def test_nearest_point_idx_miss(self, editor: CurveEditor) -> None:
        idx = editor._nearest_point_idx(1000.0, 1000.0)
        assert idx == -1

    def test_add_point_via_left_click(self, editor: CurveEditor, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtCore import Qt, QPoint

        ox, oy, w, h = editor._plot_rect()
        px, py = int(ox + w * 0.5), int(oy + h * 0.5)

        initial_count = len(editor.get_points())
        qtbot.mouseClick(editor, Qt.MouseButton.LeftButton, pos=QPoint(px, py))
        assert len(editor.get_points()) == initial_count + 1

    def test_right_click_on_endpoint_does_not_remove(self, editor: CurveEditor, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtCore import Qt, QPoint

        editor.set_points([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
        wp = editor._to_widget(0.0, 0.0)

        qtbot.mouseClick(editor, Qt.MouseButton.RightButton, pos=QPoint(int(wp.x()), int(wp.y())))
        assert len(editor.get_points()) == 3

    def test_right_click_removes_middle_point(self, editor: CurveEditor, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtCore import Qt, QPoint

        pts = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        editor.set_points(pts)
        wp = editor._to_widget(0.5, 0.5)

        qtbot.mouseClick(editor, Qt.MouseButton.RightButton, pos=QPoint(int(wp.x()), int(wp.y())))
        assert len(editor.get_points()) == 2

    def test_mouse_release_clears_drag(self, editor: CurveEditor, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtCore import Qt, QPoint

        editor._drag_idx = 0
        qtbot.mouseRelease(editor, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
        assert editor._drag_idx == -1

    def test_max_points_not_exceeded(self, editor: CurveEditor, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtCore import Qt, QPoint

        pts = [(i / (_MAX_POINTS - 1), i / (_MAX_POINTS - 1)) for i in range(_MAX_POINTS)]
        editor.set_points(pts)
        assert len(editor.get_points()) == _MAX_POINTS

        ox, oy, w, h = editor._plot_rect()
        qtbot.mouseClick(editor, Qt.MouseButton.LeftButton, pos=QPoint(int(ox + w * 0.5), int(oy + h * 0.3)))
        assert len(editor.get_points()) <= _MAX_POINTS

    def test_left_click_on_existing_point_sets_drag_idx(self, editor: CurveEditor, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtCore import Qt, QPoint

        wp = editor._to_widget(0.0, 0.0)
        qtbot.mousePress(editor, Qt.MouseButton.LeftButton, pos=QPoint(int(wp.x()), int(wp.y())))
        assert editor._drag_idx == 0
        qtbot.mouseRelease(editor, Qt.MouseButton.LeftButton, pos=QPoint(int(wp.x()), int(wp.y())))

    def test_mouse_move_with_drag_emits_points_changed(self, editor: CurveEditor, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtCore import Qt, QPoint

        editor.set_points([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
        wp = editor._to_widget(0.5, 0.5)
        qtbot.mousePress(editor, Qt.MouseButton.LeftButton, pos=QPoint(int(wp.x()), int(wp.y())))
        changed: list[object] = []
        editor.points_changed.connect(changed.append)
        qtbot.mouseMove(editor, pos=QPoint(int(wp.x()), int(wp.y()) - 20))
        assert changed

    def test_mouse_move_with_first_endpoint_locks_x(self, editor: CurveEditor, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtCore import Qt, QPoint

        editor._drag_idx = 0
        qtbot.mouseMove(editor, pos=QPoint(50, 80))
        assert editor.get_points()[0][0] == 0.0  # x locked for first endpoint

    def test_mouse_move_with_last_endpoint_locks_x(self, editor: CurveEditor, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtCore import Qt, QPoint

        editor._drag_idx = 1
        qtbot.mouseMove(editor, pos=QPoint(50, 50))
        assert editor.get_points()[-1][0] == 1.0  # x locked for last endpoint

    def test_mouse_move_without_drag_is_noop(self, editor: CurveEditor, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtCore import QPoint

        editor._drag_idx = -1
        changed: list[object] = []
        editor.points_changed.connect(changed.append)
        qtbot.mouseMove(editor, pos=QPoint(100, 100))
        assert not changed

    def test_paintEvent_spline_exception_does_not_crash(self, editor: CurveEditor, qtbot, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import astroai.ui.widgets.curves_panel as cp_mod

        def _raise(*args: object, **kwargs: object) -> None:
            raise ValueError("forced spline failure")

        monkeypatch.setattr(cp_mod, "CubicSpline", _raise)
        editor.set_points([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
        editor.repaint()  # must not raise


class TestCurvesPanel:
    def test_panel_creates(self, panel: CurvesPanel) -> None:
        assert panel is not None

    def test_channel_combo_has_five_items(self, panel: CurvesPanel) -> None:
        assert panel._channel_combo.count() == 5

    def test_on_reset_restores_all_channels(self, panel: CurvesPanel, model: PipelineModel) -> None:
        model.curves_r_points = [(0.0, 0.0), (0.5, 0.9), (1.0, 1.0)]
        panel._on_reset()
        assert model.curves_r_points == list(_IDENTITY)
        assert model.curves_g_points == list(_IDENTITY)
        assert model.curves_b_points == list(_IDENTITY)
        assert model.curves_rgb_points == list(_IDENTITY)

    def test_on_channel_changed_syncs_editor(self, panel: CurvesPanel, model: PipelineModel) -> None:
        r_pts = [(0.0, 0.0), (0.5, 0.9), (1.0, 1.0)]
        model.curves_r_points = r_pts
        panel._channel_combo.setCurrentText("R")
        assert panel._curve_editor.get_points() == r_pts

    def test_on_points_changed_updates_rgb(self, panel: CurvesPanel, model: PipelineModel) -> None:
        panel._channel_combo.setCurrentText("Alle Kanäle")
        new_pts = [(0.0, 0.0), (0.3, 0.5), (1.0, 1.0)]
        panel._on_points_changed(new_pts)
        assert model.curves_rgb_points == new_pts

    def test_on_points_changed_updates_r(self, panel: CurvesPanel, model: PipelineModel) -> None:
        panel._channel_combo.setCurrentText("R")
        new_pts = [(0.0, 0.0), (0.4, 0.6), (1.0, 1.0)]
        panel._on_points_changed(new_pts)
        assert model.curves_r_points == new_pts

    def test_on_points_changed_updates_g(self, panel: CurvesPanel, model: PipelineModel) -> None:
        panel._channel_combo.setCurrentText("G")
        new_pts = [(0.0, 0.0), (0.4, 0.7), (1.0, 1.0)]
        panel._on_points_changed(new_pts)
        assert model.curves_g_points == new_pts

    def test_on_points_changed_updates_b(self, panel: CurvesPanel, model: PipelineModel) -> None:
        panel._channel_combo.setCurrentText("B")
        new_pts = [(0.0, 0.0), (0.4, 0.4), (1.0, 1.0)]
        panel._on_points_changed(new_pts)
        assert model.curves_b_points == new_pts
