"""Tests for comet stack preview integration in PipelineModel."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.ui.models import PipelineModel


@pytest.fixture()
def model() -> PipelineModel:
    return PipelineModel()


@pytest.fixture()
def star_stack() -> np.ndarray:
    return np.ones((64, 64), dtype=np.float32)


@pytest.fixture()
def nucleus_stack() -> np.ndarray:
    return np.full((64, 64), 0.5, dtype=np.float32)


class TestCometPreviewImage:
    def test_no_stacks_returns_none(self, model: PipelineModel) -> None:
        assert model.comet_preview_image is None

    def test_stars_mode_returns_star_stack(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray
    ) -> None:
        model.set_comet_stacks(star_stack, nucleus_stack)
        model.comet_tracking_mode = "stars"
        preview = model.comet_preview_image
        assert preview is not None
        np.testing.assert_array_equal(preview, star_stack)

    def test_comet_mode_returns_nucleus_stack(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray
    ) -> None:
        model.set_comet_stacks(star_stack, nucleus_stack)
        model.comet_tracking_mode = "comet"
        preview = model.comet_preview_image
        assert preview is not None
        np.testing.assert_array_equal(preview, nucleus_stack)

    def test_blend_mode_weighted(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray
    ) -> None:
        model.set_comet_stacks(star_stack, nucleus_stack)
        model.comet_tracking_mode = "blend"
        model.comet_blend_factor = 0.0
        preview = model.comet_preview_image
        assert preview is not None
        np.testing.assert_allclose(preview, star_stack)

    def test_blend_factor_one_returns_nucleus(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray
    ) -> None:
        model.set_comet_stacks(star_stack, nucleus_stack)
        model.comet_tracking_mode = "blend"
        model.comet_blend_factor = 1.0
        preview = model.comet_preview_image
        assert preview is not None
        np.testing.assert_allclose(preview, nucleus_stack)

    def test_blend_factor_half(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray
    ) -> None:
        model.set_comet_stacks(star_stack, nucleus_stack)
        model.comet_tracking_mode = "blend"
        model.comet_blend_factor = 0.5
        expected = 0.5 * star_stack + 0.5 * nucleus_stack
        preview = model.comet_preview_image
        assert preview is not None
        np.testing.assert_allclose(preview, expected)

    def test_blend_preserves_dtype(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray
    ) -> None:
        model.set_comet_stacks(star_stack, nucleus_stack)
        model.comet_tracking_mode = "blend"
        preview = model.comet_preview_image
        assert preview is not None
        assert preview.dtype == star_stack.dtype


class TestCometPreviewSignals:
    def test_set_comet_stacks_emits_preview_changed(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray, qtbot
    ) -> None:
        with qtbot.waitSignal(model.comet_preview_changed, timeout=500):
            model.set_comet_stacks(star_stack, nucleus_stack)

    def test_tracking_mode_change_emits_preview_when_stacks_present(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray, qtbot
    ) -> None:
        model.set_comet_stacks(star_stack, nucleus_stack)
        with qtbot.waitSignal(model.comet_preview_changed, timeout=500):
            model.comet_tracking_mode = "stars"

    def test_tracking_mode_change_no_signal_without_stacks(
        self, model: PipelineModel, qtbot
    ) -> None:
        signals: list[bool] = []
        model.comet_preview_changed.connect(lambda: signals.append(True))
        model.comet_tracking_mode = "stars"
        assert len(signals) == 0

    def test_blend_factor_change_emits_preview_in_blend_mode(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray, qtbot
    ) -> None:
        model.set_comet_stacks(star_stack, nucleus_stack)
        model.comet_tracking_mode = "blend"
        with qtbot.waitSignal(model.comet_preview_changed, timeout=500):
            model.comet_blend_factor = 0.3

    def test_blend_factor_change_no_signal_in_stars_mode(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray, qtbot
    ) -> None:
        model.set_comet_stacks(star_stack, nucleus_stack)
        model.comet_tracking_mode = "stars"
        signals: list[bool] = []
        model.comet_preview_changed.connect(lambda: signals.append(True))
        model.comet_blend_factor = 0.3
        assert len(signals) == 0


class TestClearCometStacks:
    def test_clear_resets_preview(
        self, model: PipelineModel, star_stack: np.ndarray, nucleus_stack: np.ndarray
    ) -> None:
        model.set_comet_stacks(star_stack, nucleus_stack)
        model.clear_comet_stacks()
        assert model.comet_preview_image is None
