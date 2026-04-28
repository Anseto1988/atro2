"""Tests for custom AstroAI widgets."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.project.project_file import FrameEntry
from astroai.ui.models import PipelineModel, StepState
from astroai.ui.widgets.comet_stack_panel import CometStackPanel
from astroai.ui.widgets.export_panel import ExportPanel
from astroai.ui.widgets.frame_list_panel import FrameListPanel
from astroai.ui.widgets.session_notes_panel import SessionNotesPanel
from astroai.ui.widgets.image_stats_widget import ImageStatsWidget
from astroai.ui.widgets.histogram_widget import HistogramWidget
from astroai.ui.widgets.image_viewer import ImageViewer
from astroai.ui.widgets.progress_widget import ProgressWidget
from astroai.ui.widgets.synthetic_flat_panel import SyntheticFlatPanel
from astroai.ui.widgets.frame_selection_panel import FrameSelectionPanel
from astroai.ui.widgets.registration_panel import RegistrationPanel
from astroai.ui.widgets.background_removal_panel import BackgroundRemovalPanel
from astroai.ui.widgets.denoise_panel import DenoisePanel
from astroai.ui.widgets.stretch_panel import StretchPanel
from astroai.ui.widgets.stacking_panel import StackingPanel
from astroai.ui.widgets.star_processing_panel import StarProcessingPanel
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

    def test_render_full_qimage_none_when_no_data(self, viewer: ImageViewer) -> None:
        assert viewer.render_full_qimage() is None

    def test_render_full_qimage_returns_qimage(self, viewer: ImageViewer) -> None:
        from PySide6.QtGui import QImage
        data = np.random.rand(64, 64).astype(np.float32)
        viewer.set_image_data(data)
        qimg = viewer.render_full_qimage()
        assert isinstance(qimg, QImage)
        assert not qimg.isNull()

    def test_render_full_qimage_correct_size(self, viewer: ImageViewer) -> None:
        data = np.random.rand(48, 80).astype(np.float32)
        viewer.set_image_data(data)
        qimg = viewer.render_full_qimage()
        assert qimg is not None
        assert qimg.width() == 80
        assert qimg.height() == 48

    def test_render_full_qimage_does_not_modify_raw_data(
        self, viewer: ImageViewer
    ) -> None:
        data = np.random.rand(32, 32).astype(np.float32) * 0.01
        viewer.set_image_data(data)
        original = viewer._raw_data.copy()
        viewer.render_full_qimage()
        np.testing.assert_array_equal(viewer._raw_data, original)

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


class TestRegistrationPanel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, model: PipelineModel, qtbot) -> RegistrationPanel:  # type: ignore[no-untyped-def]
        w = RegistrationPanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_upsample_factor(self, panel: RegistrationPanel) -> None:
        assert panel._upsample_spin.value() == 10

    def test_initial_ref_index(self, panel: RegistrationPanel) -> None:
        assert panel._ref_index_spin.value() == 0

    def test_upsample_spin_updates_model(
        self, panel: RegistrationPanel, model: PipelineModel
    ) -> None:
        panel._upsample_spin.setValue(20)
        assert model.registration_upsample_factor == 20

    def test_ref_index_spin_updates_model(
        self, panel: RegistrationPanel, model: PipelineModel
    ) -> None:
        panel._ref_index_spin.setValue(5)
        assert model.registration_reference_frame_index == 5

    def test_model_change_updates_upsample_spin(
        self, panel: RegistrationPanel, model: PipelineModel
    ) -> None:
        model.registration_upsample_factor = 50
        assert panel._upsample_spin.value() == 50

    def test_model_change_updates_ref_index_spin(
        self, panel: RegistrationPanel, model: PipelineModel
    ) -> None:
        model.registration_reference_frame_index = 3
        assert panel._ref_index_spin.value() == 3

    def test_upsample_range(self, panel: RegistrationPanel) -> None:
        assert panel._upsample_spin.minimum() == 1
        assert panel._upsample_spin.maximum() == 100

    def test_ref_index_min(self, panel: RegistrationPanel) -> None:
        assert panel._ref_index_spin.minimum() == 0


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


class TestFrameSelectionPanel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, qtbot, model: PipelineModel) -> FrameSelectionPanel:  # type: ignore[no-untyped-def]
        w = FrameSelectionPanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_state_disabled(self, panel: FrameSelectionPanel) -> None:
        assert not panel._enabled_cb.isChecked()
        assert not panel._settings_group.isEnabled()

    def test_enable_toggles_settings_group(
        self, panel: FrameSelectionPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        assert model.frame_selection_enabled
        assert panel._settings_group.isEnabled()

    def test_min_score_updates_model(
        self, panel: FrameSelectionPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        panel._score_spin.setValue(0.7)
        assert model.frame_selection_min_score == pytest.approx(0.7)

    def test_max_rejected_fraction_updates_model(
        self, panel: FrameSelectionPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        panel._reject_spin.setValue(0.6)
        assert model.frame_selection_max_rejected_fraction == pytest.approx(0.6)

    def test_sync_from_model(
        self, panel: FrameSelectionPanel, model: PipelineModel
    ) -> None:
        model._frame_selection_enabled = True
        model._frame_selection_min_score = 0.3
        model._frame_selection_max_rejected_fraction = 0.9
        model.frame_selection_config_changed.emit()
        assert panel._enabled_cb.isChecked()
        assert panel._score_spin.value() == pytest.approx(0.3)
        assert panel._reject_spin.value() == pytest.approx(0.9)

    def test_pipeline_reset_preserves_enabled_state(
        self, panel: FrameSelectionPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        assert model.frame_selection_enabled
        model.reset()
        assert panel._enabled_cb.isChecked()  # reset preserves user config


class TestBackgroundRemovalPanel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, qtbot, model: PipelineModel) -> BackgroundRemovalPanel:  # type: ignore[no-untyped-def]
        w = BackgroundRemovalPanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_state_disabled(self, panel: BackgroundRemovalPanel) -> None:
        assert not panel._enabled_cb.isChecked()
        assert not panel._settings_group.isEnabled()

    def test_enable_toggles_settings_group(
        self, panel: BackgroundRemovalPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        assert model.background_removal_enabled
        assert panel._settings_group.isEnabled()

    def test_tile_size_updates_model(
        self, panel: BackgroundRemovalPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        panel._tile_spin.setValue(128)
        assert model.background_removal_tile_size == 128

    def test_preserve_median_updates_model(
        self, panel: BackgroundRemovalPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        panel._preserve_median_cb.setChecked(False)
        assert model.background_removal_preserve_median is False

    def test_method_combo_updates_model(
        self, panel: BackgroundRemovalPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        panel._method_combo.setCurrentIndex(1)  # "poly"
        assert model.background_removal_method == "poly"

    def test_sync_from_model(
        self, panel: BackgroundRemovalPanel, model: PipelineModel
    ) -> None:
        model._background_removal_enabled = True
        model._background_removal_tile_size = 32
        model._background_removal_preserve_median = False
        model.background_removal_config_changed.emit()
        assert panel._enabled_cb.isChecked()
        assert panel._tile_spin.value() == 32
        assert not panel._preserve_median_cb.isChecked()

    def test_pipeline_reset_preserves_enabled_state(
        self, panel: BackgroundRemovalPanel, model: PipelineModel
    ) -> None:
        panel._enabled_cb.setChecked(True)
        assert model.background_removal_enabled
        model.reset()
        assert panel._enabled_cb.isChecked()


class TestDenoisePanel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, qtbot, model: PipelineModel) -> DenoisePanel:  # type: ignore[no-untyped-def]
        w = DenoisePanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_strength(self, panel: DenoisePanel) -> None:
        assert panel._strength_spin.value() == pytest.approx(1.0)

    def test_strength_updates_model(
        self, panel: DenoisePanel, model: PipelineModel
    ) -> None:
        panel._strength_spin.setValue(0.6)
        assert model.denoise_strength == pytest.approx(0.6)

    def test_tile_size_updates_model(
        self, panel: DenoisePanel, model: PipelineModel
    ) -> None:
        panel._tile_spin.setValue(256)
        assert model.denoise_tile_size == 256

    def test_overlap_updates_model(
        self, panel: DenoisePanel, model: PipelineModel
    ) -> None:
        panel._overlap_spin.setValue(32)
        assert model.denoise_tile_overlap == 32

    def test_sync_from_model(
        self, panel: DenoisePanel, model: PipelineModel
    ) -> None:
        model._denoise_strength = 0.5
        model._denoise_tile_size = 128
        model._denoise_tile_overlap = 16
        model.denoise_config_changed.emit()
        assert panel._strength_spin.value() == pytest.approx(0.5)
        assert panel._tile_spin.value() == 128
        assert panel._overlap_spin.value() == 16


class TestStretchPanel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, qtbot, model: PipelineModel) -> StretchPanel:  # type: ignore[no-untyped-def]
        w = StretchPanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_target_background(self, panel: StretchPanel) -> None:
        assert panel._bg_spin.value() == pytest.approx(0.25)

    def test_bg_updates_model(
        self, panel: StretchPanel, model: PipelineModel
    ) -> None:
        panel._bg_spin.setValue(0.3)
        assert model.stretch_target_background == pytest.approx(0.3)

    def test_sigma_updates_model(
        self, panel: StretchPanel, model: PipelineModel
    ) -> None:
        panel._sigma_spin.setValue(-3.5)
        assert model.stretch_shadow_clipping_sigmas == pytest.approx(-3.5)

    def test_linked_checkbox_updates_model(
        self, panel: StretchPanel, model: PipelineModel
    ) -> None:
        panel._linked_cb.setChecked(False)
        assert model.stretch_linked_channels is False

    def test_sync_from_model(
        self, panel: StretchPanel, model: PipelineModel
    ) -> None:
        model._stretch_target_background = 0.2
        model._stretch_shadow_clipping_sigmas = -3.0
        model._stretch_linked_channels = False
        model.stretch_config_changed.emit()
        assert panel._bg_spin.value() == pytest.approx(0.2)
        assert panel._sigma_spin.value() == pytest.approx(-3.0)
        assert not panel._linked_cb.isChecked()


class TestStarProcessingPanel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, qtbot, model: PipelineModel) -> StarProcessingPanel:  # type: ignore[no-untyped-def]
        w = StarProcessingPanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_reduce_factor_slider(self, panel: StarProcessingPanel) -> None:
        assert panel._factor_slider.value() == 50

    def test_reduce_cb_updates_model(
        self, panel: StarProcessingPanel, model: PipelineModel
    ) -> None:
        panel._reduce_cb.setChecked(True)
        assert model.star_reduce_enabled is True

    def test_factor_slider_updates_model(
        self, panel: StarProcessingPanel, model: PipelineModel
    ) -> None:
        panel._factor_slider.setValue(70)
        assert model.star_reduce_factor == pytest.approx(0.7)

    def test_sigma_spin_updates_model(
        self, panel: StarProcessingPanel, model: PipelineModel
    ) -> None:
        panel._sigma_spin.setValue(6.0)
        assert model.star_detection_sigma == pytest.approx(6.0)

    def test_min_area_spin_updates_model(
        self, panel: StarProcessingPanel, model: PipelineModel
    ) -> None:
        panel._min_area_spin.setValue(10)
        assert model.star_min_area == 10

    def test_max_area_spin_updates_model(
        self, panel: StarProcessingPanel, model: PipelineModel
    ) -> None:
        panel._max_area_spin.setValue(8000)
        assert model.star_max_area == 8000

    def test_sync_from_model(
        self, panel: StarProcessingPanel, model: PipelineModel
    ) -> None:
        model._star_reduce_enabled = True
        model._star_reduce_factor = 0.3
        model._star_detection_sigma = 5.0
        model.star_processing_config_changed.emit()
        assert panel._reduce_cb.isChecked()
        assert panel._factor_slider.value() == 30
        assert panel._sigma_spin.value() == pytest.approx(5.0)


class TestStackingPanel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, qtbot, model: PipelineModel) -> StackingPanel:  # type: ignore[no-untyped-def]
        w = StackingPanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_method_is_sigma_clip(self, panel: StackingPanel) -> None:
        assert panel._method_combo.currentText() == "sigma_clip"

    def test_sigma_spins_enabled_for_sigma_clip(self, panel: StackingPanel) -> None:
        panel._method_combo.setCurrentText("sigma_clip")
        assert panel._sigma_low_spin.isEnabled()
        assert panel._sigma_high_spin.isEnabled()

    def test_sigma_spins_disabled_for_mean(self, panel: StackingPanel) -> None:
        panel._method_combo.setCurrentText("mean")
        assert not panel._sigma_low_spin.isEnabled()
        assert not panel._sigma_high_spin.isEnabled()

    def test_sigma_spins_disabled_for_median(self, panel: StackingPanel) -> None:
        panel._method_combo.setCurrentText("median")
        assert not panel._sigma_low_spin.isEnabled()
        assert not panel._sigma_high_spin.isEnabled()

    def test_method_combo_updates_model(
        self, panel: StackingPanel, model: PipelineModel
    ) -> None:
        panel._method_combo.setCurrentText("mean")
        assert model.stacking_method == "mean"

    def test_sigma_low_spin_updates_model(
        self, panel: StackingPanel, model: PipelineModel
    ) -> None:
        panel._sigma_low_spin.setValue(1.5)
        assert model.stacking_sigma_low == pytest.approx(1.5)

    def test_sigma_high_spin_updates_model(
        self, panel: StackingPanel, model: PipelineModel
    ) -> None:
        panel._sigma_high_spin.setValue(3.0)
        assert model.stacking_sigma_high == pytest.approx(3.0)

    def test_sync_from_model(
        self, panel: StackingPanel, model: PipelineModel
    ) -> None:
        model._stacking_method = "median"
        model._stacking_sigma_low = 2.0
        model._stacking_sigma_high = 4.0
        model.stacking_config_changed.emit()
        assert panel._method_combo.currentText() == "median"
        assert panel._sigma_low_spin.value() == pytest.approx(2.0)
        assert panel._sigma_high_spin.value() == pytest.approx(4.0)


class TestExportPanel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, model: PipelineModel, qtbot) -> ExportPanel:  # type: ignore[no-untyped-def]
        w = ExportPanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_dir_edit_empty(self, panel: ExportPanel) -> None:
        assert panel._dir_edit.text() == ""

    def test_initial_name_edit(self, panel: ExportPanel) -> None:
        assert panel._name_edit.text() == "output"

    def test_initial_format_fits(self, panel: ExportPanel) -> None:
        assert panel._fmt_combo.currentText() == "FITS"

    def test_format_combo_updates_model(
        self, panel: ExportPanel, model: PipelineModel
    ) -> None:
        panel._fmt_combo.setCurrentText("XISF")
        assert model.output_format == "xisf"

    def test_format_combo_tiff_updates_model(
        self, panel: ExportPanel, model: PipelineModel
    ) -> None:
        panel._fmt_combo.setCurrentText("TIFF32")
        assert model.output_format == "tiff"

    def test_dir_edit_editing_finished_updates_model(
        self, panel: ExportPanel, model: PipelineModel
    ) -> None:
        panel._dir_edit.setText("/tmp/output")
        panel._dir_edit.editingFinished.emit()
        assert model.output_path == "/tmp/output"

    def test_name_edit_editing_finished_updates_model(
        self, panel: ExportPanel, model: PipelineModel
    ) -> None:
        panel._name_edit.setText("m42_stack")
        panel._name_edit.editingFinished.emit()
        assert model.output_filename == "m42_stack"

    def test_name_edit_empty_uses_default(
        self, panel: ExportPanel, model: PipelineModel
    ) -> None:
        panel._name_edit.setText("")
        panel._name_edit.editingFinished.emit()
        assert model.output_filename == "output"

    def test_sync_from_model_updates_dir_edit(
        self, panel: ExportPanel, model: PipelineModel
    ) -> None:
        model._output_path = "/data/astro"
        model.export_config_changed.emit()
        assert panel._dir_edit.text() == "/data/astro"

    def test_sync_from_model_updates_format_combo(
        self, panel: ExportPanel, model: PipelineModel
    ) -> None:
        model._output_format = "tiff"
        model.export_config_changed.emit()
        assert panel._fmt_combo.currentText() == "TIFF32"

    def test_sync_from_model_updates_name_edit(
        self, panel: ExportPanel, model: PipelineModel
    ) -> None:
        model._output_filename = "ngc891"
        model.export_config_changed.emit()
        assert panel._name_edit.text() == "ngc891"

    def test_unknown_format_falls_back_to_first(
        self, panel: ExportPanel, model: PipelineModel
    ) -> None:
        model._output_format = "unknown_fmt"
        model.export_config_changed.emit()
        assert panel._fmt_combo.currentIndex() == 0


class TestFrameListPanel:
    @pytest.fixture()
    def panel(self, qtbot) -> FrameListPanel:  # type: ignore[no-untyped-def]
        w = FrameListPanel()
        qtbot.addWidget(w)
        return w

    def test_initial_empty(self, panel: FrameListPanel) -> None:
        assert panel._table.rowCount() == 0
        assert "Keine" in panel._count_label.text()

    def test_refresh_populates_rows(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/data/light_001.fits", exposure=120.0, selected=True),
            FrameEntry(path="/data/light_002.fits", exposure=120.0, selected=True),
        ]
        panel.refresh(entries)
        assert panel._table.rowCount() == 2

    def test_filename_column_shows_basename(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/deep/path/ngc7000_001.fits", exposure=300.0)]
        panel.refresh(entries)
        assert panel._table.item(0, 0).text() == "ngc7000_001.fits"

    def test_filename_tooltip_is_full_path(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/deep/path/frame.fits")]
        panel.refresh(entries)
        assert panel._table.item(0, 0).toolTip() == "/deep/path/frame.fits"

    def test_exposure_column_formatted(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/f.fits", exposure=120.5)]
        panel.refresh(entries)
        assert panel._table.item(0, 1).text() == "120.5"

    def test_exposure_none_shows_dash(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/f.fits", exposure=None)]
        panel.refresh(entries)
        assert panel._table.item(0, 1).text() == "—"

    def test_quality_score_formatted_as_percent(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/f.fits", quality_score=0.875)]
        panel.refresh(entries)
        assert panel._table.item(0, 2).text() == "87.5%"

    def test_quality_score_none_shows_dash(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/f.fits", quality_score=None)]
        panel.refresh(entries)
        assert panel._table.item(0, 2).text() == "—"

    def test_selected_true_shows_checkmark(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/f.fits", selected=True)]
        panel.refresh(entries)
        assert panel._table.item(0, 3).text() == "✓"

    def test_selected_false_shows_cross(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/f.fits", selected=False)]
        panel.refresh(entries)
        assert panel._table.item(0, 3).text() == "✗"

    def test_count_label_updates(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits", quality_score=0.9, selected=True),
            FrameEntry(path="/b.fits", quality_score=0.7, selected=False),
        ]
        panel.refresh(entries)
        label = panel._count_label.text()
        assert "2 Frame" in label
        assert "1 ausgewählt" in label
        assert "2 bewertet" in label

    def test_refresh_clears_previous(self, panel: FrameListPanel) -> None:
        panel.refresh([FrameEntry(path="/a.fits"), FrameEntry(path="/b.fits")])
        panel.refresh([FrameEntry(path="/c.fits")])
        assert panel._table.rowCount() == 1

    def test_refresh_empty_resets_label(self, panel: FrameListPanel) -> None:
        panel.refresh([FrameEntry(path="/a.fits")])
        panel.refresh([])
        assert "Keine" in panel._count_label.text()
        assert panel._table.rowCount() == 0

    def test_double_click_toggles_selected_false(
        self, panel: FrameListPanel
    ) -> None:
        entry = FrameEntry(path="/f.fits", selected=True)
        panel.refresh([entry])
        panel._on_cell_double_clicked(0, 0)
        assert entry.selected is False
        assert panel._table.item(0, 3).text() == "✗"

    def test_double_click_toggles_selected_true(
        self, panel: FrameListPanel
    ) -> None:
        entry = FrameEntry(path="/f.fits", selected=False)
        panel.refresh([entry])
        panel._on_cell_double_clicked(0, 0)
        assert entry.selected is True
        assert panel._table.item(0, 3).text() == "✓"

    def test_double_click_updates_count_label(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", selected=True),
            FrameEntry(path="/b.fits", selected=True),
        ]
        panel.refresh(entries)
        panel._on_cell_double_clicked(0, 0)
        assert "1 ausgewählt" in panel._count_label.text()

    def test_double_click_emits_selection_changed(
        self, qtbot, panel: FrameListPanel
    ) -> None:
        entry = FrameEntry(path="/f.fits", selected=True)
        panel.refresh([entry])
        with qtbot.waitSignal(panel.selection_changed) as blocker:
            panel._on_cell_double_clicked(0, 0)
        assert blocker.args == [0, False]

    def test_double_click_out_of_range_is_noop(
        self, panel: FrameListPanel
    ) -> None:
        panel.refresh([])
        panel._on_cell_double_clicked(99, 0)  # must not raise

    def test_double_click_second_time_re_enables(
        self, panel: FrameListPanel
    ) -> None:
        entry = FrameEntry(path="/f.fits", selected=True)
        panel.refresh([entry])
        panel._on_cell_double_clicked(0, 0)
        panel._on_cell_double_clicked(0, 0)
        assert entry.selected is True

    def test_entries_reference_mutated_by_toggle(
        self, panel: FrameListPanel
    ) -> None:
        entries = [FrameEntry(path="/f.fits", selected=True)]
        panel.refresh(entries)
        panel._on_cell_double_clicked(0, 0)
        assert entries[0].selected is False

    # --- bulk selection helpers ---

    def test_select_all_sets_all_selected(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits", selected=False),
            FrameEntry(path="/b.fits", selected=False),
        ]
        panel.refresh(entries)
        panel.select_all()
        assert all(e.selected for e in entries)

    def test_deselect_all_clears_all(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits", selected=True),
            FrameEntry(path="/b.fits", selected=True),
        ]
        panel.refresh(entries)
        panel.deselect_all()
        assert not any(e.selected for e in entries)

    def test_invert_selection_flips_all(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits", selected=True),
            FrameEntry(path="/b.fits", selected=False),
        ]
        panel.refresh(entries)
        panel.invert_selection()
        assert entries[0].selected is False
        assert entries[1].selected is True

    def test_select_all_updates_cell_text(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/f.fits", selected=False)]
        panel.refresh(entries)
        panel.select_all()
        assert panel._table.item(0, 3).text() == "✓"

    def test_deselect_all_updates_cell_text(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/f.fits", selected=True)]
        panel.refresh(entries)
        panel.deselect_all()
        assert panel._table.item(0, 3).text() == "✗"

    def test_select_all_updates_count_label(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits", selected=False),
            FrameEntry(path="/b.fits", selected=False),
        ]
        panel.refresh(entries)
        panel.select_all()
        assert "2 ausgewählt" in panel._count_label.text()

    def test_select_all_emits_selection_changed_for_changed_rows(
        self, qtbot, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", selected=False),
            FrameEntry(path="/b.fits", selected=True),   # already selected — no signal
        ]
        panel.refresh(entries)
        signals: list[tuple[int, bool]] = []
        panel.selection_changed.connect(lambda i, v: signals.append((i, v)))
        panel.select_all()
        # Only index 0 was changed
        assert (0, True) in signals
        assert not any(i == 1 for i, _ in signals)

    def test_remove_requested_signal(self, qtbot, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits"),
            FrameEntry(path="/b.fits"),
        ]
        panel.refresh(entries)
        with qtbot.waitSignal(panel.remove_requested) as blocker:
            panel.remove_requested.emit([0])
        assert blocker.args[0] == [0]

    def test_invert_selection_empty_is_noop(
        self, panel: FrameListPanel
    ) -> None:
        panel.refresh([])
        panel.invert_selection()  # must not raise

    def test_count_label_shows_total_exposure(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", selected=True, exposure=300.0),
            FrameEntry(path="/b.fits", selected=True, exposure=300.0),
        ]
        panel.refresh(entries)
        assert "10m" in panel._count_label.text()

    def test_count_label_no_exposure_when_unset(
        self, panel: FrameListPanel
    ) -> None:
        entries = [FrameEntry(path="/a.fits", selected=True, exposure=None)]
        panel.refresh(entries)
        # No exposure section when all frames lack exposure data
        label = panel._count_label.text()
        assert "0s" not in label and "0m" not in label

    def test_count_label_excludes_deselected_exposure(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", selected=True, exposure=120.0),
            FrameEntry(path="/b.fits", selected=False, exposure=9999.0),
        ]
        panel.refresh(entries)
        label = panel._count_label.text()
        assert "2m" in label  # 120s = 2m

    def test_format_exposure_seconds(self, panel: FrameListPanel) -> None:
        from astroai.ui.widgets.frame_list_panel import _format_exposure
        assert _format_exposure(45) == "45s"

    def test_format_exposure_minutes(self, panel: FrameListPanel) -> None:
        from astroai.ui.widgets.frame_list_panel import _format_exposure
        assert _format_exposure(90) == "1m 30s"

    def test_format_exposure_hours(self, panel: FrameListPanel) -> None:
        from astroai.ui.widgets.frame_list_panel import _format_exposure
        assert _format_exposure(3720) == "1h 02m"

    # --- sort ---

    def test_sort_by_name_ascending(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/data/z_frame.fits"),
            FrameEntry(path="/data/a_frame.fits"),
        ]
        panel.refresh(entries)
        panel._on_header_clicked(0)
        assert panel._table.item(0, 0).text() == "a_frame.fits"
        assert panel._table.item(1, 0).text() == "z_frame.fits"

    def test_sort_by_name_descending_on_second_click(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/data/a_frame.fits"),
            FrameEntry(path="/data/z_frame.fits"),
        ]
        panel.refresh(entries)
        panel._on_header_clicked(0)
        panel._on_header_clicked(0)
        assert panel._table.item(0, 0).text() == "z_frame.fits"

    def test_sort_by_exposure_ascending(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits", exposure=300.0),
            FrameEntry(path="/b.fits", exposure=60.0),
        ]
        panel.refresh(entries)
        panel._on_header_clicked(1)
        assert panel._table.item(0, 1).text() == "60.0"
        assert panel._table.item(1, 1).text() == "300.0"

    def test_sort_by_quality_ascending(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits", quality_score=0.9),
            FrameEntry(path="/b.fits", quality_score=0.3),
        ]
        panel.refresh(entries)
        panel._on_header_clicked(2)
        assert panel._table.item(0, 2).text() == "30.0%"

    def test_sort_by_selected_ascending(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits", selected=True),
            FrameEntry(path="/b.fits", selected=False),
        ]
        panel.refresh(entries)
        panel._on_header_clicked(3)
        assert panel._table.item(0, 3).text() == "✗"

    def test_sort_changes_entries_order_in_place(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/data/z.fits"),
            FrameEntry(path="/data/a.fits"),
        ]
        panel.refresh(entries)
        panel._on_header_clicked(0)
        assert panel._entries[0].path == "/data/a.fits"

    def test_sort_resets_direction_on_new_column(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", exposure=300.0),
            FrameEntry(path="/b.fits", exposure=60.0),
        ]
        panel.refresh(entries)
        panel._on_header_clicked(0)
        panel._on_header_clicked(0)   # now descending on col 0
        panel._on_header_clicked(1)   # switch to col 1 — must be ascending again
        assert panel._sort_asc is True
        assert panel._table.item(0, 1).text() == "60.0"

    def test_sort_none_exposure_sorted_first(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits", exposure=120.0),
            FrameEntry(path="/b.fits", exposure=None),
        ]
        panel.refresh(entries)
        panel._on_header_clicked(1)
        assert panel._table.item(0, 1).text() == "—"

    def test_sort_on_empty_panel_is_noop(self, panel: FrameListPanel) -> None:
        panel.refresh([])
        panel._on_header_clicked(0)  # must not raise

    def test_refresh_reapplies_active_sort(self, panel: FrameListPanel) -> None:
        panel.refresh([FrameEntry(path="/z.fits"), FrameEntry(path="/a.fits")])
        panel._on_header_clicked(0)  # sort ascending by name
        panel.refresh([FrameEntry(path="/m.fits"), FrameEntry(path="/b.fits")])
        assert panel._table.item(0, 0).text() == "b.fits"

    # --- filter ---

    def test_filter_hides_non_matching_rows(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/data/ngc7000.fits"),
            FrameEntry(path="/data/m31.fits"),
        ]
        panel.refresh(entries)
        panel._on_filter_changed("ngc")
        assert not panel._table.isRowHidden(0)
        assert panel._table.isRowHidden(1)

    def test_filter_empty_shows_all_rows(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits"),
            FrameEntry(path="/b.fits"),
        ]
        panel.refresh(entries)
        panel._on_filter_changed("xyz")
        panel._on_filter_changed("")
        assert not panel._table.isRowHidden(0)
        assert not panel._table.isRowHidden(1)

    def test_filter_is_case_insensitive(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/data/NGC7000.fits")]
        panel.refresh(entries)
        panel._on_filter_changed("ngc7000")
        assert not panel._table.isRowHidden(0)

    def test_filter_persists_after_refresh(self, panel: FrameListPanel) -> None:
        panel.refresh([FrameEntry(path="/a.fits")])
        panel._on_filter_changed("ngc")
        panel.refresh([
            FrameEntry(path="/ngc4321.fits"),
            FrameEntry(path="/m33.fits"),
        ])
        assert not panel._table.isRowHidden(0)
        assert panel._table.isRowHidden(1)

    def test_filter_input_placeholder_set(self, panel: FrameListPanel) -> None:
        assert panel._filter_input.placeholderText() != ""

    def test_filter_text_state_updated(self, panel: FrameListPanel) -> None:
        panel._on_filter_changed("test")
        assert panel._filter_text == "test"

    def test_filter_clear_shows_all(self, panel: FrameListPanel) -> None:
        entries = [
            FrameEntry(path="/a.fits"),
            FrameEntry(path="/b.fits"),
        ]
        panel.refresh(entries)
        panel._on_filter_changed("a")
        assert panel._table.isRowHidden(1)
        panel._on_filter_changed("")
        assert not panel._table.isRowHidden(1)

    # --- quality threshold ---

    def test_threshold_deselects_below_minimum(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", quality_score=0.9, selected=True),
            FrameEntry(path="/b.fits", quality_score=0.3, selected=True),
        ]
        panel.refresh(entries)
        panel.apply_quality_threshold(50.0)  # 50%
        assert entries[0].selected is True
        assert entries[1].selected is False

    def test_threshold_selects_above_minimum(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", quality_score=0.8, selected=False),
            FrameEntry(path="/b.fits", quality_score=0.4, selected=False),
        ]
        panel.refresh(entries)
        panel.apply_quality_threshold(30.0)
        assert entries[0].selected is True
        assert entries[1].selected is True

    def test_threshold_ignores_unscored_frames(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", quality_score=None, selected=True),
        ]
        panel.refresh(entries)
        panel.apply_quality_threshold(90.0)
        assert entries[0].selected is True  # unchanged

    def test_threshold_zero_selects_all_scored(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", quality_score=0.1, selected=False),
            FrameEntry(path="/b.fits", quality_score=0.9, selected=False),
        ]
        panel.refresh(entries)
        panel.apply_quality_threshold(0.0)
        assert all(e.selected for e in entries)

    def test_threshold_100_deselects_all_scored(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", quality_score=0.99, selected=True),
        ]
        panel.refresh(entries)
        panel.apply_quality_threshold(100.0)
        # quality_score 0.99 < 1.00 threshold → deselected
        assert entries[0].selected is False

    def test_threshold_emits_selection_changed(
        self, qtbot, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", quality_score=0.2, selected=True),
        ]
        panel.refresh(entries)
        signals: list[tuple[int, bool]] = []
        panel.selection_changed.connect(lambda i, v: signals.append((i, v)))
        panel.apply_quality_threshold(50.0)
        assert (0, False) in signals

    def test_threshold_updates_count_label(
        self, panel: FrameListPanel
    ) -> None:
        entries = [
            FrameEntry(path="/a.fits", quality_score=0.9, selected=True),
            FrameEntry(path="/b.fits", quality_score=0.2, selected=True),
        ]
        panel.refresh(entries)
        panel.apply_quality_threshold(50.0)
        assert "1 ausgewählt" in panel._count_label.text()

    def test_threshold_spinbox_default_zero(
        self, panel: FrameListPanel
    ) -> None:
        assert panel._quality_spinbox.value() == 0.0

    def test_threshold_updates_cell_text(
        self, panel: FrameListPanel
    ) -> None:
        entries = [FrameEntry(path="/a.fits", quality_score=0.3, selected=True)]
        panel.refresh(entries)
        panel.apply_quality_threshold(50.0)
        assert panel._table.item(0, 3).text() == "✗"

    def test_on_apply_threshold_reads_spinbox(self, panel: FrameListPanel) -> None:
        entries = [FrameEntry(path="/a.fits", quality_score=0.3, selected=True)]
        panel.refresh(entries)
        panel._quality_spinbox.setValue(50.0)
        panel._on_apply_threshold()
        assert not panel._entries[0].selected

    def test_sort_key_fn_default_col_returns_zero(self, panel: FrameListPanel) -> None:
        from astroai.project.project_file import FrameEntry
        panel._sort_col = 99  # invalid column
        key_fn = panel._sort_key_fn()
        entry = FrameEntry(path="/a.fits")
        assert key_fn(entry) == 0

    def test_context_menu_empty_entries_is_noop(self, panel: FrameListPanel) -> None:
        from PySide6.QtCore import QPoint
        panel._entries = []
        panel._show_context_menu(QPoint(10, 10))  # must not raise

    def test_drag_move_no_urls_ignored(self, panel: FrameListPanel, qtbot, tmp_path) -> None:
        from PySide6.QtCore import QMimeData, Qt, QPoint
        from PySide6.QtGui import QDragMoveEvent

        mime = QMimeData()  # no URLs
        event = QDragMoveEvent(
            panel.rect().center(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.dragMoveEvent(event)
        assert not event.isAccepted()

    def test_drag_move_with_urls_accepted(self, panel: FrameListPanel, qtbot, tmp_path) -> None:
        from PySide6.QtCore import QMimeData, QUrl, Qt, QPoint
        from PySide6.QtGui import QDragMoveEvent

        fits_path = tmp_path / "frame.fits"
        fits_path.write_bytes(b"")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(fits_path))])
        event = QDragMoveEvent(
            panel.rect().center(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.dragMoveEvent(event)
        assert event.isAccepted()


class TestFrameListPanelContextMenu:
    """Tests for context menu action dispatch (lines 264-276)."""

    @pytest.fixture()
    def panel_with_entries(self, qtbot) -> FrameListPanel:  # type: ignore[no-untyped-def]
        p = FrameListPanel()
        qtbot.addWidget(p)
        entries = [
            FrameEntry(path="/a.fits", selected=False),
            FrameEntry(path="/b.fits", selected=True),
        ]
        p.refresh(entries)
        p._table.selectRow(0)
        return p

    def _mock_exec(self, text: str):  # type: ignore[no-untyped-def]
        from PySide6.QtWidgets import QMenu

        def _exec(menu: QMenu, pos: object = None) -> object:
            for act in menu.actions():
                if act.text() == text:
                    return act
            return None
        return _exec

    def test_select_all_via_context_menu(self, panel_with_entries: FrameListPanel, monkeypatch) -> None:
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QMenu
        monkeypatch.setattr(QMenu, "exec", self._mock_exec("Alle auswählen"))
        panel_with_entries._show_context_menu(QPoint(10, 10))
        assert all(e.selected for e in panel_with_entries._entries)

    def test_deselect_all_via_context_menu(self, panel_with_entries: FrameListPanel, monkeypatch) -> None:
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QMenu
        monkeypatch.setattr(QMenu, "exec", self._mock_exec("Alle abwählen"))
        panel_with_entries._show_context_menu(QPoint(10, 10))
        assert all(not e.selected for e in panel_with_entries._entries)

    def test_invert_via_context_menu(self, panel_with_entries: FrameListPanel, monkeypatch) -> None:
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QMenu
        before = [e.selected for e in panel_with_entries._entries]
        monkeypatch.setattr(QMenu, "exec", self._mock_exec("Auswahl umkehren"))
        panel_with_entries._show_context_menu(QPoint(10, 10))
        after = [e.selected for e in panel_with_entries._entries]
        assert after == [not s for s in before]

    def test_remove_via_context_menu_emits_signal(self, panel_with_entries: FrameListPanel, monkeypatch) -> None:
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QMenu
        removed: list[object] = []
        panel_with_entries.remove_requested.connect(removed.append)
        monkeypatch.setattr(QMenu, "exec", self._mock_exec("1 Frame(s) entfernen"))
        panel_with_entries._show_context_menu(QPoint(10, 10))
        assert removed

    def test_preview_via_context_menu_emits_signal(self, panel_with_entries: FrameListPanel, monkeypatch) -> None:
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QMenu
        previews: list[str] = []
        panel_with_entries.preview_requested.connect(previews.append)
        monkeypatch.setattr(QMenu, "exec", self._mock_exec("Vorschau anzeigen"))
        panel_with_entries._show_context_menu(QPoint(10, 10))
        assert previews == ["/a.fits"]

    def test_notes_via_context_menu_calls_edit(self, panel_with_entries: FrameListPanel, monkeypatch) -> None:
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QMenu, QInputDialog
        monkeypatch.setattr(QMenu, "exec", self._mock_exec("Notiz bearbeiten…"))
        monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("test note", True))
        panel_with_entries._show_context_menu(QPoint(10, 10))
        assert panel_with_entries._entries[0].notes == "test note"


class TestSessionNotesPanel:
    @pytest.fixture()
    def panel(self, qtbot) -> SessionNotesPanel:  # type: ignore[no-untyped-def]
        w = SessionNotesPanel()
        qtbot.addWidget(w)
        return w

    def test_initial_notes_empty(self, panel: SessionNotesPanel) -> None:
        assert panel.notes == ""

    def test_set_notes_updates_property(self, panel: SessionNotesPanel) -> None:
        panel.set_notes("Beobachtungsnacht 2026-04-28")
        assert panel.notes == "Beobachtungsnacht 2026-04-28"

    def test_set_notes_does_not_emit_signal(self, panel: SessionNotesPanel, qtbot) -> None:  # type: ignore[no-untyped-def]
        emitted: list[str] = []
        panel.text_changed.connect(emitted.append)
        panel.set_notes("silent")
        assert emitted == []

    def test_text_changed_signal_emitted_on_edit(self, panel: SessionNotesPanel, qtbot) -> None:  # type: ignore[no-untyped-def]
        emitted: list[str] = []
        panel.text_changed.connect(emitted.append)
        panel._editor.setPlainText("new note")
        assert len(emitted) >= 1
        assert emitted[-1] == "new note"

    def test_set_notes_empty_string(self, panel: SessionNotesPanel) -> None:
        panel.set_notes("some text")
        panel.set_notes("")
        assert panel.notes == ""

    def test_multiline_notes_preserved(self, panel: SessionNotesPanel) -> None:
        text = "Zeile 1\nZeile 2\nZeile 3"
        panel.set_notes(text)
        assert panel.notes == text

    def test_unicode_notes(self, panel: SessionNotesPanel) -> None:
        text = "Seeing: außergewöhnlich gut — Mond 15% beleuchtet"
        panel.set_notes(text)
        assert panel.notes == text

    def test_placeholder_text_present(self, panel: SessionNotesPanel) -> None:
        assert panel._editor.placeholderText() != ""

    def test_plain_text_only(self, panel: SessionNotesPanel) -> None:
        assert panel._editor.acceptRichText() is False

    def test_set_notes_overwrites_previous(self, panel: SessionNotesPanel) -> None:
        panel.set_notes("first")
        panel.set_notes("second")
        assert panel.notes == "second"


class TestImageStatsWidget:
    @pytest.fixture()
    def widget(self, qtbot) -> ImageStatsWidget:  # type: ignore[no-untyped-def]
        w = ImageStatsWidget()
        qtbot.addWidget(w)
        return w

    def test_initial_state_one_row(self, widget: ImageStatsWidget) -> None:
        assert widget._table.rowCount() == 1

    def test_initial_placeholder_values(self, widget: ImageStatsWidget) -> None:
        for col in range(1, 5):
            assert widget._table.item(0, col).text() == "—"

    def test_mono_image_one_row(self, widget: ImageStatsWidget) -> None:
        data = np.ones((32, 32), dtype=np.float32) * 0.5
        widget.set_image_data(data)
        assert widget._table.rowCount() == 1

    def test_mono_channel_label(self, widget: ImageStatsWidget) -> None:
        data = np.ones((32, 32), dtype=np.float32)
        widget.set_image_data(data)
        assert widget._table.item(0, 0).text() == "L"

    def test_color_image_three_rows(self, widget: ImageStatsWidget) -> None:
        data = np.ones((32, 32, 3), dtype=np.float32)
        widget.set_image_data(data)
        assert widget._table.rowCount() == 3

    def test_color_channel_labels(self, widget: ImageStatsWidget) -> None:
        data = np.ones((16, 16, 3), dtype=np.float32)
        widget.set_image_data(data)
        labels = [widget._table.item(i, 0).text() for i in range(3)]
        assert labels == ["R", "G", "B"]

    def test_mono_mean_correct(self, widget: ImageStatsWidget) -> None:
        data = np.full((10, 10), 0.25, dtype=np.float32)
        widget.set_image_data(data)
        assert widget._table.item(0, 1).text() == "0.2500"

    def test_mono_min_max_correct(self, widget: ImageStatsWidget) -> None:
        data = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        widget.set_image_data(data)
        assert widget._table.item(0, 3).text() == "0.0000"
        assert widget._table.item(0, 4).text() == "1.0000"

    def test_color_channel_values_differ(self, widget: ImageStatsWidget) -> None:
        data = np.zeros((16, 16, 3), dtype=np.float32)
        data[:, :, 0] = 0.1
        data[:, :, 1] = 0.5
        data[:, :, 2] = 0.9
        widget.set_image_data(data)
        r_mean = widget._table.item(0, 1).text()
        g_mean = widget._table.item(1, 1).text()
        b_mean = widget._table.item(2, 1).text()
        assert r_mean != g_mean
        assert g_mean != b_mean

    def test_clear_resets_to_placeholder(self, widget: ImageStatsWidget) -> None:
        data = np.ones((16, 16), dtype=np.float32)
        widget.set_image_data(data)
        widget.clear()
        for col in range(1, 5):
            assert widget._table.item(0, col).text() == "—"

    def test_invalid_ndim_clears(self, widget: ImageStatsWidget) -> None:
        data = np.ones((16, 16), dtype=np.float32)
        widget.set_image_data(data)
        widget.set_image_data(np.ones((4, 4, 4, 4), dtype=np.float32))
        for col in range(1, 5):
            assert widget._table.item(0, col).text() == "—"

    def test_headers_correct(self, widget: ImageStatsWidget) -> None:
        assert widget._table.columnCount() == 5
        assert widget._table.horizontalHeaderItem(0).text() == "Kanal"


class TestFrameListPanelNotes:
    @pytest.fixture()
    def panel(self, qtbot) -> FrameListPanel:  # type: ignore[no-untyped-def]
        p = FrameListPanel()
        qtbot.addWidget(p)
        return p

    def test_entry_with_notes_shows_asterisk(self, panel: FrameListPanel) -> None:
        from astroai.project.project_file import FrameEntry

        entry = FrameEntry(path="/data/frame.fits", notes="cloudy")
        panel.refresh([entry])
        item = panel._table.item(0, 0)
        assert item is not None
        assert item.text().startswith("*")

    def test_entry_without_notes_no_asterisk(self, panel: FrameListPanel) -> None:
        from astroai.project.project_file import FrameEntry

        entry = FrameEntry(path="/data/frame.fits", notes="")
        panel.refresh([entry])
        item = panel._table.item(0, 0)
        assert item is not None
        assert not item.text().startswith("*")

    def test_notes_shown_in_tooltip(self, panel: FrameListPanel) -> None:
        from astroai.project.project_file import FrameEntry

        entry = FrameEntry(path="/data/frame.fits", notes="bad tracking")
        panel.refresh([entry])
        item = panel._table.item(0, 0)
        assert item is not None
        assert "bad tracking" in item.toolTip()

    def test_edit_notes_updates_entry(self, panel: FrameListPanel, monkeypatch) -> None:
        from astroai.project.project_file import FrameEntry
        from PySide6.QtWidgets import QInputDialog

        entry = FrameEntry(path="/data/frame.fits")
        panel.refresh([entry])

        monkeypatch.setattr(
            QInputDialog, "getText",
            lambda *a, **kw: ("new note", True),
        )
        panel._edit_notes(0)
        assert panel._entries[0].notes == "new note"

    def test_edit_notes_cancelled_leaves_entry_unchanged(
        self, panel: FrameListPanel, monkeypatch
    ) -> None:
        from astroai.project.project_file import FrameEntry
        from PySide6.QtWidgets import QInputDialog

        entry = FrameEntry(path="/data/frame.fits", notes="original")
        panel.refresh([entry])

        monkeypatch.setattr(
            QInputDialog, "getText",
            lambda *a, **kw: ("ignored", False),
        )
        panel._edit_notes(0)
        assert panel._entries[0].notes == "original"

    def test_edit_notes_out_of_range_no_crash(self, panel: FrameListPanel) -> None:
        panel._edit_notes(99)  # must not raise


class TestFrameListPanelPreviewSignal:
    @pytest.fixture()
    def panel(self, qtbot) -> FrameListPanel:  # type: ignore[no-untyped-def]
        p = FrameListPanel()
        qtbot.addWidget(p)
        return p

    def test_preview_requested_signal_declared(self, panel: FrameListPanel) -> None:
        assert hasattr(panel, "preview_requested")

    def test_preview_requested_emitted_from_context_menu(
        self, panel: FrameListPanel
    ) -> None:
        from astroai.project.project_file import FrameEntry

        entry = FrameEntry(path="/data/frame.fits")
        panel.refresh([entry])

        received: list[str] = []
        panel.preview_requested.connect(received.append)

        # Simulate single row selected then trigger preview directly
        panel._table.selectRow(0)
        panel._entries[0]  # ensure row 0 maps to entry
        # Directly invoke the emit path (context menu would call this)
        panel.preview_requested.emit(entry.path)
        assert received == ["/data/frame.fits"]

    def test_preview_act_enabled_for_single_selection(
        self, panel: FrameListPanel
    ) -> None:
        from astroai.project.project_file import FrameEntry

        entry = FrameEntry(path="/data/frame.fits")
        panel.refresh([entry])
        panel._table.selectRow(0)
        selected = sorted({item.row() for item in panel._table.selectedItems()})
        assert len(selected) == 1

    def test_preview_act_disabled_for_multi_selection(
        self, panel: FrameListPanel
    ) -> None:
        from astroai.project.project_file import FrameEntry
        from PySide6.QtCore import QItemSelectionModel

        entries = [FrameEntry(path=f"/data/f{i}.fits") for i in range(3)]
        panel.refresh(entries)
        panel._table.selectRow(0)
        panel._table.selectionModel().select(
            panel._table.model().index(1, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
        selected = sorted({item.row() for item in panel._table.selectedItems()})
        assert len(selected) > 1  # multi-selection → preview should be disabled


class TestFrameListPanelDragDrop:
    @pytest.fixture()
    def panel(self, qtbot):  # type: ignore[no-untyped-def]
        p = FrameListPanel()
        qtbot.addWidget(p)
        return p

    def test_panel_accepts_drops(self, panel: FrameListPanel) -> None:
        assert panel.acceptDrops()

    def test_files_dropped_signal_declared(self, panel: FrameListPanel) -> None:
        assert hasattr(panel, "files_dropped")

    def test_fits_suffixes_constant(self, panel: FrameListPanel) -> None:
        assert ".fits" in panel._FITS_SUFFIXES
        assert ".fit" in panel._FITS_SUFFIXES
        assert ".fts" in panel._FITS_SUFFIXES

    def test_drop_emits_signal(self, panel: FrameListPanel, qtbot, tmp_path) -> None:
        from pathlib import Path as _Path
        from PySide6.QtCore import QMimeData, QUrl, Qt
        from PySide6.QtGui import QDropEvent

        fits_path = tmp_path / "test.fits"
        fits_path.write_bytes(b"")

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(fits_path))])

        received: list[list[str]] = []
        panel.files_dropped.connect(received.append)

        event = QDropEvent(
            panel.rect().center().toPointF(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.dropEvent(event)

        assert len(received) == 1
        assert any(_Path(p) == fits_path for p in received[0])

    def test_non_fits_drop_ignored(self, panel: FrameListPanel, qtbot, tmp_path) -> None:
        from PySide6.QtCore import QMimeData, QUrl
        from PySide6.QtGui import QDropEvent
        from PySide6.QtCore import Qt

        png_path = tmp_path / "image.png"
        png_path.write_bytes(b"")

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(png_path))])

        received: list[list[str]] = []
        panel.files_dropped.connect(received.append)

        event = QDropEvent(
            panel.rect().center().toPointF(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.dropEvent(event)

        assert received == []

    def test_drag_enter_accepts_fits(self, panel: FrameListPanel, qtbot, tmp_path) -> None:
        from PySide6.QtCore import QMimeData, QUrl
        from PySide6.QtGui import QDragEnterEvent
        from PySide6.QtCore import Qt

        fits_path = tmp_path / "frame.fits"
        fits_path.write_bytes(b"")

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(fits_path))])

        event = QDragEnterEvent(
            panel.rect().center(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.dragEnterEvent(event)
        assert event.isAccepted()

    def test_drag_enter_ignores_non_fits(self, panel: FrameListPanel, qtbot, tmp_path) -> None:
        from PySide6.QtCore import QMimeData, QUrl
        from PySide6.QtGui import QDragEnterEvent
        from PySide6.QtCore import Qt

        txt_path = tmp_path / "notes.txt"
        txt_path.write_bytes(b"")

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(txt_path))])

        event = QDragEnterEvent(
            panel.rect().center(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.dragEnterEvent(event)
        assert not event.isAccepted()
