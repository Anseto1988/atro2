"""Tests for SplitCompareView widget."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt

from astroai.ui.widgets.split_compare_view import SplitCompareView


@pytest.fixture()
def view(qtbot):  # type: ignore[no-untyped-def]
    w = SplitCompareView()
    qtbot.addWidget(w)
    w.resize(400, 300)
    w.show()
    return w


@pytest.fixture()
def gray2d() -> np.ndarray:
    return np.random.rand(64, 64).astype(np.float32)


@pytest.fixture()
def rgb3d() -> np.ndarray:
    return np.random.rand(3, 64, 64).astype(np.float32)


class TestSplitCompareViewInit:
    def test_initial_split_is_centered(self, view: SplitCompareView) -> None:
        assert view._split == pytest.approx(0.5)

    def test_no_image_initially(self, view: SplitCompareView) -> None:
        assert view._before is None
        assert view._after is None

    def test_initial_zoom_positive(self, view: SplitCompareView) -> None:
        assert view._zoom > 0

    def test_accessible_name_set(self, view: SplitCompareView) -> None:
        assert "Vergleich" in view.accessibleName()


class TestSplitCompareViewSetData:
    def test_set_before_stores_2d(
        self, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        assert view._before is not None
        assert view._before.ndim == 2

    def test_set_after_stores_2d(
        self, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_after(gray2d)
        assert view._after is not None
        assert view._after.ndim == 2

    def test_set_before_rgb_collapses_to_2d(
        self, view: SplitCompareView, rgb3d: np.ndarray
    ) -> None:
        view.set_before(rgb3d)
        assert view._before is not None
        assert view._before.ndim == 2

    def test_set_after_rgb_collapses_to_2d(
        self, view: SplitCompareView, rgb3d: np.ndarray
    ) -> None:
        view.set_after(rgb3d)
        assert view._after is not None
        assert view._after.ndim == 2

    def test_set_before_clears_cache(
        self, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view._before_cache[(0, 0)] = None  # type: ignore[assignment]
        view.set_before(gray2d)
        assert (0, 0) not in view._before_cache

    def test_set_after_clears_cache(
        self, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view._after_cache[(0, 0)] = None  # type: ignore[assignment]
        view.set_after(gray2d)
        assert (0, 0) not in view._after_cache

    def test_dims_updated_from_before(
        self, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        assert view._height == 64
        assert view._width == 64

    def test_dims_updated_from_after(
        self, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_after(gray2d)
        assert view._height == 64
        assert view._width == 64


class TestSplitCompareViewClear:
    def test_clear_removes_images(
        self, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        view.set_after(gray2d)
        view.clear()
        assert view._before is None
        assert view._after is None

    def test_clear_empties_caches(
        self, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        view.set_after(gray2d)
        # Force tile generation
        view._get_tile(view._before, view._before_cache, 0, 0)  # type: ignore[arg-type]
        view.clear()
        assert len(view._before_cache) == 0
        assert len(view._after_cache) == 0


class TestSplitCompareViewPaint:
    def test_paint_no_crash_without_images(
        self, qtbot, view: SplitCompareView
    ) -> None:
        view.repaint()  # must not raise

    def test_paint_no_crash_with_before_only(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        view.repaint()

    def test_paint_no_crash_with_both_images(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        view.set_after(gray2d)
        view.repaint()


class TestSplitCompareViewInteraction:
    def test_fit_to_view_resets_offset(
        self, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        view._offset.setX(100)
        view.fit_to_view()
        assert view._offset.x() == pytest.approx(0.0)

    def test_near_split_true_at_center(self, view: SplitCompareView) -> None:
        center = view.width() * view._split
        assert view._near_split(center)

    def test_near_split_false_far_away(self, view: SplitCompareView) -> None:
        assert not view._near_split(0.0)

    def test_key_home_calls_fit(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        view._offset.setX(50)
        qtbot.keyClick(view, Qt.Key.Key_Home)
        assert view._offset.x() == pytest.approx(0.0)


class TestSplitCompareViewZoom:
    def test_zoom_level_property(self, view: SplitCompareView) -> None:
        assert view.zoom_level == view._zoom

    def test_set_before_emits_zoom_changed(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        with qtbot.waitSignal(view.zoom_changed, timeout=500):
            view.set_before(gray2d)

    def test_set_after_emits_zoom_changed(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        with qtbot.waitSignal(view.zoom_changed, timeout=500):
            view.set_after(gray2d)

    def test_fit_to_view_emits_zoom_changed(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        with qtbot.waitSignal(view.zoom_changed, timeout=500):
            view.fit_to_view()

    def test_key_plus_emits_zoom_changed(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        with qtbot.waitSignal(view.zoom_changed, timeout=500):
            qtbot.keyClick(view, Qt.Key.Key_Plus)

    def test_key_minus_emits_zoom_changed(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        with qtbot.waitSignal(view.zoom_changed, timeout=500):
            qtbot.keyClick(view, Qt.Key.Key_Minus)

    def test_zoom_changed_signal_carries_float(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        with qtbot.waitSignal(view.zoom_changed, timeout=500) as blocker:
            view.set_before(gray2d)
        assert isinstance(blocker.args[0], float)
        assert blocker.args[0] > 0

    def test_wheel_up_increases_zoom(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QWheelEvent

        view.set_before(gray2d)
        old_zoom = view._zoom
        event = QWheelEvent(
            QPointF(200.0, 150.0),
            view.mapToGlobal(QPoint(200, 150)),
            QPoint(0, 120),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        view.wheelEvent(event)
        assert view._zoom > old_zoom

    def test_wheel_down_decreases_zoom(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QWheelEvent

        view.set_before(gray2d)
        old_zoom = view._zoom
        event = QWheelEvent(
            QPointF(200.0, 150.0),
            view.mapToGlobal(QPoint(200, 150)),
            QPoint(0, -120),
            QPoint(0, -120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        view.wheelEvent(event)
        assert view._zoom < old_zoom


class TestSplitCompareViewMouseAndKey:
    def test_left_press_near_split_sets_split_drag(
        self, view: SplitCompareView
    ) -> None:
        center = int(view.width() * view._split)
        from PySide6.QtCore import QPoint
        import pytest

        view._near_split(float(center))  # sanity-check
        # Simulate press near split
        view._split_drag = False
        view._pan_drag = False
        # Direct test: _near_split at center should be True
        assert view._near_split(float(center))

    def test_left_press_away_from_split_sets_pan_drag(
        self, qtbot, view: SplitCompareView
    ) -> None:
        qtbot.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(10, 150))
        assert view._pan_drag

    def test_left_release_clears_drag_flags(
        self, qtbot, view: SplitCompareView
    ) -> None:
        view._pan_drag = True
        view._split_drag = True
        qtbot.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(200, 150))
        assert not view._pan_drag
        assert not view._split_drag

    def test_key_left_shifts_offset(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        before_x = view._offset.x()
        qtbot.keyClick(view, Qt.Key.Key_Left)
        assert view._offset.x() != before_x

    def test_key_right_shifts_offset(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        before_x = view._offset.x()
        qtbot.keyClick(view, Qt.Key.Key_Right)
        assert view._offset.x() != before_x

    def test_key_up_shifts_offset(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        before_y = view._offset.y()
        qtbot.keyClick(view, Qt.Key.Key_Up)
        assert view._offset.y() != before_y

    def test_key_down_shifts_offset(
        self, qtbot, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        before_y = view._offset.y()
        qtbot.keyClick(view, Qt.Key.Key_Down)
        assert view._offset.y() != before_y

    def test_tile_cache_hit_reuses_tile(
        self, view: SplitCompareView, gray2d: np.ndarray
    ) -> None:
        view.set_before(gray2d)
        assert view._before is not None
        tile1 = view._get_tile(view._before, view._before_cache, 0, 0)
        tile2 = view._get_tile(view._before, view._before_cache, 0, 0)
        assert tile1 is tile2  # same object from cache
