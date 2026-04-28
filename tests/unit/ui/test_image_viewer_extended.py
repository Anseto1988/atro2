"""Extended tests for ImageViewer — paint, mouse/key events, fit_to_view."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QWidget

from astroai.ui.widgets.image_viewer import ImageViewer, _MAX_ZOOM, _MIN_ZOOM, _PAN_STEP


@pytest.fixture()
def viewer(qtbot) -> ImageViewer:
    w = ImageViewer()
    qtbot.addWidget(w)
    w.resize(400, 300)
    return w


@pytest.fixture()
def loaded_viewer(viewer: ImageViewer) -> ImageViewer:
    data = np.random.rand(200, 300).astype(np.float32)
    viewer.set_image_data(data)
    return viewer


class TestImageViewerPaint:
    def test_paint_empty_no_crash(self, viewer: ImageViewer) -> None:
        viewer.repaint()

    def test_paint_with_data(self, loaded_viewer: ImageViewer) -> None:
        loaded_viewer.repaint()

    def test_paint_with_small_tile(self, viewer: ImageViewer) -> None:
        data = np.random.rand(64, 64).astype(np.float32)
        viewer.set_image_data(data)
        viewer.repaint()

    def test_tile_cache_populated_after_render(self, loaded_viewer: ImageViewer) -> None:
        loaded_viewer._render_tile(0, 0)
        assert len(loaded_viewer._tile_cache) > 0

    def test_tile_cache_cleared_on_zoom(self, loaded_viewer: ImageViewer) -> None:
        loaded_viewer.repaint()
        loaded_viewer.set_zoom(2.0)
        assert len(loaded_viewer._tile_cache) == 0


class TestImageViewerFitToView:
    def test_fit_to_view_with_data(self, loaded_viewer: ImageViewer) -> None:
        loaded_viewer.set_zoom(5.0)
        loaded_viewer.fit_to_view()
        assert loaded_viewer.zoom_level < 5.0
        assert loaded_viewer._offset == QPointF(0, 0)

    def test_fit_to_view_no_data(self, viewer: ImageViewer) -> None:
        viewer.set_zoom(3.0)
        viewer.fit_to_view()
        assert viewer.zoom_level == 3.0

    def test_fit_emits_zoom_changed(self, loaded_viewer: ImageViewer, qtbot) -> None:
        loaded_viewer.set_zoom(5.0)
        with qtbot.waitSignal(loaded_viewer.zoom_changed, timeout=500):
            loaded_viewer.fit_to_view()


class TestImageViewerKeyboard:
    def test_key_plus_zooms_in(self, loaded_viewer: ImageViewer) -> None:
        old = loaded_viewer.zoom_level
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Plus, Qt.KeyboardModifier.NoModifier)
        loaded_viewer.keyPressEvent(event)
        assert loaded_viewer.zoom_level > old

    def test_key_minus_zooms_out(self, loaded_viewer: ImageViewer) -> None:
        loaded_viewer.set_zoom(2.0)
        old = loaded_viewer.zoom_level
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Minus, Qt.KeyboardModifier.NoModifier)
        loaded_viewer.keyPressEvent(event)
        assert loaded_viewer.zoom_level < old

    def test_key_home_fits(self, loaded_viewer: ImageViewer) -> None:
        loaded_viewer.set_zoom(10.0)
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Home, Qt.KeyboardModifier.NoModifier)
        loaded_viewer.keyPressEvent(event)
        assert loaded_viewer.zoom_level < 10.0

    def test_key_left_pans(self, loaded_viewer: ImageViewer) -> None:
        old_x = loaded_viewer._offset.x()
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier)
        loaded_viewer.keyPressEvent(event)
        assert loaded_viewer._offset.x() == old_x + _PAN_STEP

    def test_key_right_pans(self, loaded_viewer: ImageViewer) -> None:
        old_x = loaded_viewer._offset.x()
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
        loaded_viewer.keyPressEvent(event)
        assert loaded_viewer._offset.x() == old_x - _PAN_STEP

    def test_key_up_pans(self, loaded_viewer: ImageViewer) -> None:
        old_y = loaded_viewer._offset.y()
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier)
        loaded_viewer.keyPressEvent(event)
        assert loaded_viewer._offset.y() == old_y + _PAN_STEP

    def test_key_down_pans(self, loaded_viewer: ImageViewer) -> None:
        old_y = loaded_viewer._offset.y()
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier)
        loaded_viewer.keyPressEvent(event)
        assert loaded_viewer._offset.y() == old_y - _PAN_STEP

    def test_key_equal_zooms_in(self, loaded_viewer: ImageViewer) -> None:
        old = loaded_viewer.zoom_level
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Equal, Qt.KeyboardModifier.NoModifier)
        loaded_viewer.keyPressEvent(event)
        assert loaded_viewer.zoom_level > old

    def test_unhandled_key_no_crash(self, loaded_viewer: ImageViewer) -> None:
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
        loaded_viewer.keyPressEvent(event)


class TestImageViewerMouse:
    def test_mouse_press_starts_drag(self, loaded_viewer: ImageViewer, qtbot) -> None:
        from PySide6.QtGui import QMouseEvent

        pos = QPointF(100, 100)
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            pos, pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        loaded_viewer.mousePressEvent(event)
        assert loaded_viewer._dragging is True

    def test_mouse_release_stops_drag(self, loaded_viewer: ImageViewer) -> None:
        from PySide6.QtGui import QMouseEvent

        pos = QPointF(100, 100)
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            pos, pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        loaded_viewer.mousePressEvent(press)

        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            pos, pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        loaded_viewer.mouseReleaseEvent(release)
        assert loaded_viewer._dragging is False

    def test_mouse_move_drag_updates_offset(self, loaded_viewer: ImageViewer) -> None:
        from PySide6.QtGui import QMouseEvent

        start = QPointF(100, 100)
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            start, start,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        loaded_viewer.mousePressEvent(press)

        move_pos = QPointF(120, 110)
        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            move_pos, move_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        old_offset = QPointF(loaded_viewer._offset)
        loaded_viewer.mouseMoveEvent(move)
        assert loaded_viewer._offset != old_offset

    def test_mouse_move_hover_emits_pixel(self, loaded_viewer: ImageViewer, qtbot) -> None:
        from PySide6.QtGui import QMouseEvent

        loaded_viewer._offset = QPointF(0, 0)
        loaded_viewer._zoom = 1.0
        pos = QPointF(10, 10)
        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            pos, pos,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        with qtbot.waitSignal(loaded_viewer.pixel_hovered, timeout=500):
            loaded_viewer.mouseMoveEvent(move)

    def test_mouse_move_hover_out_of_bounds(self, loaded_viewer: ImageViewer) -> None:
        from PySide6.QtGui import QMouseEvent

        loaded_viewer._offset = QPointF(0, 0)
        loaded_viewer._zoom = 1.0
        pos = QPointF(9999, 9999)
        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            pos, pos,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        loaded_viewer.mouseMoveEvent(move)


class TestImageViewerExtra:
    def test_set_image_data_3d_collapses_to_2d(self, viewer: ImageViewer) -> None:
        data = np.random.rand(3, 200, 300).astype(np.float32)
        viewer.set_image_data(data)
        assert viewer._raw_data is not None
        assert viewer._raw_data.ndim == 2

    def test_clear_removes_data(self, viewer: ImageViewer) -> None:
        viewer.set_image_data(np.random.rand(100, 100).astype(np.float32))
        viewer.clear()
        assert viewer._raw_data is None

    def test_clear_empties_tile_cache(self, viewer: ImageViewer) -> None:
        viewer.set_image_data(np.random.rand(100, 100).astype(np.float32))
        viewer._render_tile(0, 0)
        viewer.clear()
        assert len(viewer._tile_cache) == 0

    def test_render_full_qimage_returns_none_without_data(self, viewer: ImageViewer) -> None:
        assert viewer.render_full_qimage() is None

    def test_render_full_qimage_returns_qimage(self, viewer: ImageViewer) -> None:
        viewer.set_image_data(np.random.rand(50, 80).astype(np.float32))
        img = viewer.render_full_qimage()
        assert img is not None
        assert img.width() == 80
        assert img.height() == 50

    def test_render_full_qimage_uniform_data(self, viewer: ImageViewer) -> None:
        viewer.set_image_data(np.ones((40, 40), dtype=np.float32))
        img = viewer.render_full_qimage()
        assert img is not None

    def test_tile_cache_hit_returns_same_image(self, viewer: ImageViewer) -> None:
        viewer.set_image_data(np.random.rand(100, 100).astype(np.float32))
        first = viewer._render_tile(0, 0)
        second = viewer._render_tile(0, 0)
        assert first is second


class TestImageViewerPaintForced:
    """Uses show()+grab() to force synchronous paintEvent execution."""

    @pytest.fixture()
    def shown_viewer(self, qtbot) -> ImageViewer:
        w = ImageViewer()
        w.resize(400, 300)
        qtbot.addWidget(w)
        w.show()
        return w

    @pytest.fixture()
    def shown_loaded(self, shown_viewer: ImageViewer) -> ImageViewer:
        shown_viewer.set_image_data(np.random.rand(200, 300).astype(np.float32))
        return shown_viewer

    def test_grab_empty_no_crash(self, shown_viewer: ImageViewer) -> None:
        shown_viewer.grab()

    def test_grab_with_data(self, shown_loaded: ImageViewer) -> None:
        shown_loaded.grab()

    def test_grab_after_clear(self, shown_loaded: ImageViewer) -> None:
        shown_loaded.clear()
        shown_loaded.grab()

    def test_grab_3d_data(self, shown_viewer: ImageViewer) -> None:
        shown_viewer.set_image_data(np.random.rand(3, 50, 80).astype(np.float32))
        shown_viewer.grab()

    def test_grab_uses_tile_cache_on_second_paint(self, shown_loaded: ImageViewer) -> None:
        shown_loaded.grab()
        cache_size_after_first = len(shown_loaded._tile_cache)
        shown_loaded.grab()
        assert len(shown_loaded._tile_cache) == cache_size_after_first

    def test_grab_high_zoom_single_tile(self, shown_loaded: ImageViewer) -> None:
        shown_loaded.set_zoom(0.1)
        shown_loaded.grab()

    def test_grab_zoomed_out(self, shown_loaded: ImageViewer) -> None:
        shown_loaded.set_zoom(0.5)
        shown_loaded.grab()


class TestImageViewerWheel:
    def test_wheel_zoom_in(self, loaded_viewer: ImageViewer) -> None:
        old = loaded_viewer.zoom_level
        from PySide6.QtCore import QPoint

        event = QWheelEvent(
            QPointF(100, 100), QPointF(100, 100),
            QPoint(0, 120), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        loaded_viewer.wheelEvent(event)
        assert loaded_viewer.zoom_level > old

    def test_wheel_zoom_out(self, loaded_viewer: ImageViewer) -> None:
        loaded_viewer.set_zoom(2.0)
        old = loaded_viewer.zoom_level
        from PySide6.QtCore import QPoint

        event = QWheelEvent(
            QPointF(100, 100), QPointF(100, 100),
            QPoint(0, -120), QPoint(0, -120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        loaded_viewer.wheelEvent(event)
        assert loaded_viewer.zoom_level < old

    def test_wheel_updates_offset(self, loaded_viewer: ImageViewer) -> None:
        from PySide6.QtCore import QPoint

        loaded_viewer._offset = QPointF(50, 50)
        event = QWheelEvent(
            QPointF(100, 100), QPointF(100, 100),
            QPoint(0, 120), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        loaded_viewer.wheelEvent(event)
        assert loaded_viewer._offset != QPointF(50, 50)
