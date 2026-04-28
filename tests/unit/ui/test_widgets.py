"""Tests for custom AstroAI widgets."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.ui.models import PipelineModel, StepState
from astroai.ui.widgets.comet_stack_panel import CometStackPanel
from astroai.ui.widgets.histogram_widget import HistogramWidget
from astroai.ui.widgets.image_viewer import ImageViewer
from astroai.ui.widgets.progress_widget import ProgressWidget
from astroai.ui.widgets.synthetic_flat_panel import SyntheticFlatPanel
from astroai.ui.widgets.workflow_graph import WorkflowGraph


class TestImageViewer:
    @pytest.fixture()
    def viewer(self, qtbot) -> ImageViewer:  # type: ignore[no-untyped-def]
        w = ImageViewer()
        qtbot.addWidget(w)
        return w

    def test_initial_state(self, viewer: ImageViewer) -> None:
        assert viewer._raw_data is None
        assert viewer.zoom_level == 1.0

    def test_set_image_data(self, viewer: ImageViewer) -> None:
        data = np.random.rand(100, 200).astype(np.float32)
        viewer.set_image_data(data)
        assert viewer._raw_data is not None
        assert viewer._width == 200
        assert viewer._height == 100

    def test_set_image_3d_reduces(self, viewer: ImageViewer) -> None:
        data = np.random.rand(3, 100, 200).astype(np.float32)
        viewer.set_image_data(data)
        assert viewer._raw_data is not None
        assert viewer._raw_data.ndim == 2

    def test_clear(self, viewer: ImageViewer) -> None:
        viewer.set_image_data(np.zeros((10, 10), dtype=np.float32))
        viewer.clear()
        assert viewer._raw_data is None

    def test_set_zoom(self, viewer: ImageViewer) -> None:
        viewer.set_zoom(2.5)
        assert viewer.zoom_level == 2.5

    def test_zoom_clamps(self, viewer: ImageViewer) -> None:
        viewer.set_zoom(100.0)
        assert viewer.zoom_level == 50.0
        viewer.set_zoom(0.001)
        assert viewer.zoom_level == 0.05

    def test_zoom_signal(self, viewer: ImageViewer, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(viewer.zoom_changed, timeout=500):
            viewer.set_zoom(3.0)

    def test_render_tile_uses_cache_on_second_call(self, viewer: ImageViewer) -> None:
        data = np.random.rand(256, 256).astype(np.float32)
        viewer.set_image_data(data)
        tile1 = viewer._render_tile(0, 0)
        tile2 = viewer._render_tile(0, 0)
        assert tile1 is tile2

    def test_paint_event_with_data(self, viewer: ImageViewer, qtbot) -> None:  # type: ignore[no-untyped-def]
        data = np.random.rand(256, 256).astype(np.float32)
        viewer.set_image_data(data)
        viewer.resize(400, 300)
        viewer.show()
        qtbot.waitExposed(viewer, timeout=1000)
        viewer.repaint()

    def test_paint_event_no_data(self, viewer: ImageViewer, qtbot) -> None:  # type: ignore[no-untyped-def]
        viewer.resize(400, 300)
        viewer.show()
        qtbot.waitExposed(viewer, timeout=1000)
        viewer.repaint()  # _raw_data is None — early return


class TestHistogramWidget:
    @pytest.fixture()
    def histogram(self, qtbot) -> HistogramWidget:  # type: ignore[no-untyped-def]
        w = HistogramWidget()
        qtbot.addWidget(w)
        return w

    def test_initial_state(self, histogram: HistogramWidget) -> None:
        assert histogram._bins is None

    def test_set_image_data(self, histogram: HistogramWidget) -> None:
        data = np.random.rand(50, 50).astype(np.float32)
        histogram.set_image_data(data)
        assert histogram._bins is not None
        assert len(histogram._bins) == 256

    def test_constant_image_clears(self, histogram: HistogramWidget) -> None:
        data = np.ones((10, 10), dtype=np.float32)
        histogram.set_image_data(data)
        assert histogram._bins is None

    def test_clear(self, histogram: HistogramWidget) -> None:
        histogram.set_image_data(np.random.rand(10, 10).astype(np.float32))
        histogram.clear()
        assert histogram._bins is None

    def test_paint_event_with_data(self, histogram: HistogramWidget, qtbot) -> None:  # type: ignore[no-untyped-def]
        histogram.set_image_data(np.random.rand(50, 50).astype(np.float32))
        histogram.resize(200, 100)
        histogram.show()
        qtbot.waitExposed(histogram, timeout=1000)
        histogram.repaint()

    def test_paint_event_when_bins_none(self, histogram: HistogramWidget, qtbot) -> None:  # type: ignore[no-untyped-def]
        histogram.resize(200, 100)
        histogram.show()
        qtbot.waitExposed(histogram, timeout=1000)
        histogram.repaint()  # _bins is None — early return

    def test_paint_event_zero_size(self, histogram: HistogramWidget, qtbot) -> None:  # type: ignore[no-untyped-def]
        from unittest.mock import patch as _patch
        histogram.set_image_data(np.random.rand(50, 50).astype(np.float32))
        histogram.show()
        qtbot.waitExposed(histogram, timeout=1000)
        # Call paintEvent directly with patched width/height to cover the w<=0 guard
        with _patch.object(histogram, "width", return_value=1), \
             _patch.object(histogram, "height", return_value=1):
            histogram.paintEvent(None)


class TestWorkflowGraph:
    @pytest.fixture()
    def graph(self, qtbot) -> WorkflowGraph:  # type: ignore[no-untyped-def]
        model = PipelineModel()
        w = WorkflowGraph(model)
        qtbot.addWidget(w)
        return w

    def test_has_model(self, graph: WorkflowGraph) -> None:
        assert graph._model is not None

    def test_updates_on_step_change(self, graph: WorkflowGraph) -> None:
        graph._model.set_step_state("calibrate", StepState.ACTIVE)
        step = graph._model.step_by_key("calibrate")
        assert step is not None
        assert step.state is StepState.ACTIVE

    def test_size_hint(self, graph: WorkflowGraph) -> None:
        hint = graph.sizeHint()
        assert hint.width() >= 200
        assert hint.height() > 0

    def test_paint_event_runs_without_error(self, graph: WorkflowGraph, qtbot) -> None:  # type: ignore[no-untyped-def]
        graph.resize(600, 80)
        graph.show()
        qtbot.waitExposed(graph, timeout=1000)
        graph.repaint()

    def test_paint_event_with_active_step_and_progress(self, graph: WorkflowGraph, qtbot) -> None:  # type: ignore[no-untyped-def]
        graph._model.set_step_state("calibrate", StepState.ACTIVE)
        graph._model.step_by_key("calibrate").progress = 0.5  # type: ignore[union-attr]
        graph.resize(600, 80)
        graph.show()
        qtbot.waitExposed(graph, timeout=1000)
        graph.repaint()

    def test_paint_event_empty_steps(self, graph: WorkflowGraph) -> None:
        from unittest.mock import patch as _patch
        with _patch.object(graph, "_visible_steps", return_value=[]):
            graph.paintEvent(None)  # directly invoke — early return before QPainter


class TestProgressWidget:
    @pytest.fixture()
    def progress(self, qtbot) -> ProgressWidget:  # type: ignore[no-untyped-def]
        w = ProgressWidget()
        qtbot.addWidget(w)
        return w

    def test_initial_state(self, progress: ProgressWidget) -> None:
        assert progress._label.text() == "Bereit"
        assert progress._bar.value() == 0
        assert not progress._cancel_btn.isEnabled()

    def test_set_progress(self, progress: ProgressWidget) -> None:
        progress.set_progress(0.5)
        assert progress._bar.value() == 500

    def test_set_status(self, progress: ProgressWidget) -> None:
        progress.set_status("Kalibrierung...")
        assert progress._label.text() == "Kalibrierung..."

    def test_set_cancellable(self, progress: ProgressWidget) -> None:
        progress.set_cancellable(True)
        assert progress._cancel_btn.isEnabled()

    def test_reset(self, progress: ProgressWidget) -> None:
        progress.set_progress(0.7)
        progress.set_status("Test")
        progress.set_cancellable(True)
        progress.reset()
        assert progress._label.text() == "Bereit"
        assert progress._bar.value() == 0
        assert not progress._cancel_btn.isEnabled()

    def test_cancel_signal(self, progress: ProgressWidget, qtbot) -> None:  # type: ignore[no-untyped-def]
        progress.set_cancellable(True)
        with qtbot.waitSignal(progress.cancel_requested, timeout=500):
            progress._cancel_btn.click()

    def test_indeterminate(self, progress: ProgressWidget) -> None:
        progress.set_indeterminate()
        assert progress._bar.maximum() == 0

    def test_determinate(self, progress: ProgressWidget) -> None:
        progress.set_indeterminate()
        progress.set_determinate()
        assert progress._bar.maximum() == 1000


class TestCometStackPanel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, qtbot, model: PipelineModel) -> CometStackPanel:  # type: ignore[no-untyped-def]
        w = CometStackPanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_state_disabled(self, panel: CometStackPanel) -> None:
        assert not panel._enabled_cb.isChecked()
        assert not panel._settings_group.isEnabled()

    def test_initial_tracking_mode_blend(self, panel: CometStackPanel) -> None:
        assert panel._mode_buttons["blend"].isChecked()

    def test_blend_row_visible_in_blend_mode(self, panel: CometStackPanel) -> None:
        assert not panel._blend_row.isHidden()

    def test_enable_toggles_settings_group(
        self, panel: CometStackPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        assert model.comet_stack_enabled
        assert panel._settings_group.isEnabled()

    def test_mode_change_updates_model(
        self, panel: CometStackPanel, model: PipelineModel
    ) -> None:
        panel._mode_buttons["stars"].setChecked(True)
        panel._on_mode_changed()
        assert model.comet_tracking_mode == "stars"

    def test_blend_row_hidden_for_non_blend_mode(
        self, panel: CometStackPanel, model: PipelineModel
    ) -> None:
        panel._mode_buttons["comet"].setChecked(True)
        panel._on_mode_changed()
        assert panel._blend_row.isHidden()

    def test_blend_slider_updates_model(
        self, panel: CometStackPanel, model: PipelineModel
    ) -> None:
        panel._blend_slider.setValue(75)
        assert model.comet_blend_factor == pytest.approx(0.75)

    def test_blend_value_label_updates(self, panel: CometStackPanel) -> None:
        panel._blend_slider.setValue(30)
        assert panel._blend_value.text() == "0.30"

    def test_sync_from_model(
        self, panel: CometStackPanel, model: PipelineModel
    ) -> None:
        model._comet_stack_enabled = True
        model._comet_tracking_mode = "comet"
        model._comet_blend_factor = 0.2
        model.comet_stack_config_changed.emit()
        assert panel._enabled_cb.isChecked()
        assert panel._mode_buttons["comet"].isChecked()
        assert panel._blend_slider.value() == 20
        assert panel._blend_row.isHidden()

    def test_config_signal_emitted(
        self, panel: CometStackPanel, model: PipelineModel, qtbot  # type: ignore[no-untyped-def]
    ) -> None:
        with qtbot.waitSignal(model.comet_stack_config_changed, timeout=500):
            panel._enabled_cb.setChecked(True)

    def test_step_state_disabled_by_default(self, model: PipelineModel) -> None:
        step = model.step_by_key("comet_stacking")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_step_state_pending_when_enabled(self, model: PipelineModel) -> None:
        model.comet_stack_enabled = True
        step = model.step_by_key("comet_stacking")
        assert step is not None
        assert step.state is StepState.PENDING

    def test_pipeline_reset_syncs_panel(
        self, panel: CometStackPanel, model: PipelineModel
    ) -> None:
        model._comet_stack_enabled = True
        model._comet_tracking_mode = "stars"
        model.pipeline_reset.emit()
        assert panel._enabled_cb.isChecked()
        assert panel._mode_buttons["stars"].isChecked()

    def test_on_mode_changed_no_checked_button_is_noop(
        self, panel: CometStackPanel, model: PipelineModel
    ) -> None:
        """line 128: return when checkedButton() is None."""
        from unittest.mock import patch

        original_mode = model.comet_tracking_mode
        with patch.object(panel._mode_group, "checkedButton", return_value=None):
            panel._on_mode_changed()
        assert model.comet_tracking_mode == original_mode


class TestSyntheticFlatPanel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, qtbot, model: PipelineModel) -> SyntheticFlatPanel:  # type: ignore[no-untyped-def]
        w = SyntheticFlatPanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_state_disabled(self, panel: SyntheticFlatPanel) -> None:
        assert not panel._enabled_cb.isChecked()
        assert not panel._settings_group.isEnabled()

    def test_enable_toggles_settings_group(
        self, panel: SyntheticFlatPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        assert model.synthetic_flat_enabled
        assert panel._settings_group.isEnabled()

    def test_tile_size_updates_model(
        self, panel: SyntheticFlatPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        panel._tile_spin.setValue(128)
        assert model.synthetic_flat_tile_size == 128

    def test_sigma_updates_model(
        self, panel: SyntheticFlatPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        panel._sigma_spin.setValue(12.5)
        assert model.synthetic_flat_smoothing_sigma == pytest.approx(12.5)

    def test_sync_from_model(
        self, panel: SyntheticFlatPanel, model: PipelineModel
    ) -> None:
        model._synthetic_flat_enabled = True
        model._synthetic_flat_tile_size = 32
        model._synthetic_flat_smoothing_sigma = 5.0
        model.synthetic_flat_config_changed.emit()
        assert panel._enabled_cb.isChecked()
        assert panel._tile_spin.value() == 32
        assert panel._sigma_spin.value() == pytest.approx(5.0)

    def test_pipeline_reset_preserves_enabled_state(
        self, panel: SyntheticFlatPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        assert model.synthetic_flat_enabled
        model.reset()
        assert panel._enabled_cb.isChecked()  # reset keeps user config, only resets step state
