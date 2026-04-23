"""Tests for PipelineModel mosaic config and MosaicPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel, StepState


class TestPipelineModelMosaic:

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    def test_mosaic_enabled_default(self, model: PipelineModel) -> None:
        assert model.mosaic_enabled is False

    def test_mosaic_enabled_setter(self, model: PipelineModel) -> None:
        model.mosaic_enabled = True
        assert model.mosaic_enabled is True

    def test_mosaic_enabled_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.mosaic_enabled = True

    def test_mosaic_enabled_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        signals: list[bool] = []
        model.mosaic_config_changed.connect(lambda: signals.append(True))
        model.mosaic_enabled = False
        assert len(signals) == 0

    def test_mosaic_enabled_toggles_step_to_pending(self, model: PipelineModel) -> None:
        step = model.step_by_key("mosaic")
        assert step is not None
        assert step.state is StepState.DISABLED
        model.mosaic_enabled = True
        assert step.state is StepState.PENDING

    def test_mosaic_enabled_toggles_step_to_disabled(self, model: PipelineModel) -> None:
        model.mosaic_enabled = True
        step = model.step_by_key("mosaic")
        assert step is not None
        assert step.state is StepState.PENDING
        model.mosaic_enabled = False
        assert step.state is StepState.DISABLED

    def test_mosaic_enabled_emits_step_changed(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.mosaic_enabled = True
        assert blocker.args == ["mosaic", StepState.PENDING.value]

    def test_mosaic_blend_mode_default(self, model: PipelineModel) -> None:
        assert model.mosaic_blend_mode == "average"

    def test_mosaic_blend_mode_setter(self, model: PipelineModel) -> None:
        model.mosaic_blend_mode = "median"
        assert model.mosaic_blend_mode == "median"

    def test_mosaic_blend_mode_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.mosaic_blend_mode = "max"

    def test_mosaic_blend_mode_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        signals: list[bool] = []
        model.mosaic_config_changed.connect(lambda: signals.append(True))
        model.mosaic_blend_mode = "average"
        assert len(signals) == 0

    def test_mosaic_gradient_correct_default(self, model: PipelineModel) -> None:
        assert model.mosaic_gradient_correct is True

    def test_mosaic_gradient_correct_setter(self, model: PipelineModel) -> None:
        model.mosaic_gradient_correct = False
        assert model.mosaic_gradient_correct is False

    def test_mosaic_gradient_correct_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.mosaic_gradient_correct = False

    def test_mosaic_gradient_correct_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        signals: list[bool] = []
        model.mosaic_config_changed.connect(lambda: signals.append(True))
        model.mosaic_gradient_correct = True
        assert len(signals) == 0

    def test_mosaic_output_scale_default(self, model: PipelineModel) -> None:
        assert model.mosaic_output_scale == 1.0

    def test_mosaic_output_scale_setter(self, model: PipelineModel) -> None:
        model.mosaic_output_scale = 2.0
        assert model.mosaic_output_scale == pytest.approx(2.0)

    def test_mosaic_output_scale_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.mosaic_output_scale = 2.0

    def test_mosaic_output_scale_clamps_high(self, model: PipelineModel) -> None:
        model.mosaic_output_scale = 10.0
        assert model.mosaic_output_scale == 4.0

    def test_mosaic_output_scale_clamps_low(self, model: PipelineModel) -> None:
        model.mosaic_output_scale = 0.1
        assert model.mosaic_output_scale == 0.25

    def test_mosaic_output_scale_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        signals: list[bool] = []
        model.mosaic_config_changed.connect(lambda: signals.append(True))
        model.mosaic_output_scale = 1.0
        assert len(signals) == 0

    def test_mosaic_panels_default(self, model: PipelineModel) -> None:
        assert model.mosaic_panels == []

    def test_mosaic_panels_setter(self, model: PipelineModel) -> None:
        model.mosaic_panels = ["/a.fits", "/b.fits"]
        assert model.mosaic_panels == ["/a.fits", "/b.fits"]

    def test_mosaic_panels_returns_copy(self, model: PipelineModel) -> None:
        model.mosaic_panels = ["/a.fits"]
        panels = model.mosaic_panels
        panels.append("/sneaky.fits")
        assert model.mosaic_panels == ["/a.fits"]

    def test_add_mosaic_panel(self, model: PipelineModel) -> None:
        model.add_mosaic_panel("/x.fits")
        assert model.mosaic_panels == ["/x.fits"]

    def test_add_mosaic_panel_no_duplicate(self, model: PipelineModel) -> None:
        model.add_mosaic_panel("/x.fits")
        model.add_mosaic_panel("/x.fits")
        assert model.mosaic_panels == ["/x.fits"]

    def test_add_mosaic_panel_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.add_mosaic_panel("/new.fits")

    def test_remove_mosaic_panel(self, model: PipelineModel) -> None:
        model.mosaic_panels = ["/a.fits", "/b.fits"]
        model.remove_mosaic_panel("/a.fits")
        assert model.mosaic_panels == ["/b.fits"]

    def test_remove_mosaic_panel_missing_no_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        signals: list[bool] = []
        model.mosaic_config_changed.connect(lambda: signals.append(True))
        model.remove_mosaic_panel("/nonexistent.fits")
        assert len(signals) == 0

    def test_reset_keeps_mosaic_disabled(self, model: PipelineModel) -> None:
        model.set_step_state("mosaic", StepState.ERROR)
        model.reset()
        step = model.step_by_key("mosaic")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_reset_sets_mosaic_pending_when_enabled(self, model: PipelineModel) -> None:
        model.mosaic_enabled = True
        model.set_step_state("mosaic", StepState.DONE)
        model.reset()
        step = model.step_by_key("mosaic")
        assert step is not None
        assert step.state is StepState.PENDING

    def test_default_steps_includes_mosaic(self, model: PipelineModel) -> None:
        keys = [s.key for s in model.steps]
        assert "mosaic" in keys
        mosaic_idx = keys.index("mosaic")
        drizzle_idx = keys.index("drizzle")
        stretch_idx = keys.index("stretch")
        assert drizzle_idx < mosaic_idx < stretch_idx


class TestMosaicPanel:

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, model: PipelineModel, qtbot) -> "MosaicPanel":  # type: ignore[no-untyped-def]
        from astroai.ui.widgets.mosaic_panel import MosaicPanel
        w = MosaicPanel(model)
        qtbot.addWidget(w)
        return w

    def test_initial_state_disabled(self, panel: "MosaicPanel") -> None:
        assert not panel._enabled_cb.isChecked()
        assert not panel._settings_group.isEnabled()

    def test_enable_updates_model(self, panel: "MosaicPanel", model: PipelineModel) -> None:
        panel._enabled_cb.setChecked(True)
        assert model.mosaic_enabled is True

    def test_blend_mode_updates_model(self, panel: "MosaicPanel", model: PipelineModel) -> None:
        panel._enabled_cb.setChecked(True)
        panel._blend_combo.setCurrentText("median")
        assert model.mosaic_blend_mode == "median"

    def test_gradient_updates_model(self, panel: "MosaicPanel", model: PipelineModel) -> None:
        panel._enabled_cb.setChecked(True)
        panel._gradient_cb.setChecked(False)
        assert model.mosaic_gradient_correct is False

    def test_scale_slider_updates_model(self, panel: "MosaicPanel", model: PipelineModel) -> None:
        panel._enabled_cb.setChecked(True)
        panel._scale_slider.setValue(200)
        assert model.mosaic_output_scale == pytest.approx(2.0)

    def test_scale_label_updates(self, panel: "MosaicPanel") -> None:
        panel._scale_slider.setValue(150)
        assert panel._scale_value.text() == "1.50x"

    def test_sync_from_model(self, panel: "MosaicPanel", model: PipelineModel) -> None:
        model.mosaic_enabled = True
        model.mosaic_blend_mode = "max"
        model.mosaic_gradient_correct = False
        model.mosaic_output_scale = 2.5
        model.mosaic_panels = ["/test.fits"]
        assert panel._enabled_cb.isChecked()
        assert panel._blend_combo.currentText() == "max"
        assert not panel._gradient_cb.isChecked()
        assert panel._scale_slider.value() == 250
        assert panel._panel_list.count() == 1
        assert panel._panel_list.item(0).text() == "/test.fits"

    def test_model_add_panel_updates_list(self, panel: "MosaicPanel", model: PipelineModel) -> None:
        model.add_mosaic_panel("/a.fits")
        assert panel._panel_list.count() == 1

    def test_model_remove_panel_updates_list(self, panel: "MosaicPanel", model: PipelineModel) -> None:
        model.mosaic_panels = ["/a.fits", "/b.fits"]
        model.remove_mosaic_panel("/a.fits")
        assert panel._panel_list.count() == 1
        assert panel._panel_list.item(0).text() == "/b.fits"

    def test_accessible_names_set(self, panel: "MosaicPanel") -> None:
        assert panel._enabled_cb.accessibleName() != ""
        assert panel._blend_combo.accessibleName() != ""
        assert panel._gradient_cb.accessibleName() != ""
        assert panel._scale_slider.accessibleName() != ""
        assert panel._panel_list.accessibleName() != ""
        assert panel._add_btn.accessibleName() != ""
        assert panel._remove_btn.accessibleName() != ""
