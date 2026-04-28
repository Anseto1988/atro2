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
