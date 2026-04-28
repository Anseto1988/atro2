"""Tests for CometStackPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.comet_stack_panel import CometStackPanel, _TRACKING_MODES


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = CometStackPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_enabled_cb_matches_model(self, model, panel):
        assert panel._enabled_cb.isChecked() == model.comet_stack_enabled

    def test_settings_group_disabled_when_not_enabled(self, panel):
        assert not panel._settings_group.isEnabled()

    def test_all_mode_buttons_present(self, panel):
        keys = [k for k, _ in _TRACKING_MODES]
        for key in keys:
            assert key in panel._mode_buttons

    def test_blend_slider_matches_model(self, model, panel):
        expected = int(model.comet_blend_factor * 100)
        assert panel._blend_slider.value() == expected

    def test_blend_value_label_matches_model(self, model, panel):
        expected = f"{model.comet_blend_factor:.2f}"
        assert panel._blend_value.text() == expected

    def test_blend_row_hidden_when_mode_not_blend(self, model, panel):
        if model.comet_tracking_mode != "blend":
            assert panel._blend_row.isHidden()


class TestUIToModel:
    def test_enable_checkbox_updates_model(self, model, panel):
        panel._enabled_cb.setChecked(True)
        assert model.comet_stack_enabled is True

    def test_enable_checkbox_enables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        assert panel._settings_group.isEnabled()

    def test_disable_checkbox_disables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert not panel._settings_group.isEnabled()

    def test_blend_slider_updates_model(self, model, panel):
        panel._blend_slider.setValue(75)
        assert abs(model.comet_blend_factor - 0.75) < 0.01

    def test_blend_slider_updates_label(self, panel):
        panel._blend_slider.setValue(30)
        assert panel._blend_value.text() == "0.30"

    def test_on_mode_changed_comet_updates_model(self, model, panel):
        panel._mode_buttons["comet"].setChecked(True)
        panel._on_mode_changed()
        assert model.comet_tracking_mode == "comet"

    def test_on_mode_changed_blend_shows_blend_row(self, panel):
        panel._mode_buttons["blend"].setChecked(True)
        panel._on_mode_changed()
        assert not panel._blend_row.isHidden()

    def test_on_mode_changed_stars_hides_blend_row(self, panel):
        panel._mode_buttons["blend"].setChecked(True)
        panel._on_mode_changed()
        panel._mode_buttons["stars"].setChecked(True)
        panel._on_mode_changed()
        assert panel._blend_row.isHidden()

    def test_on_mode_changed_no_button_checked_no_crash(self, panel, monkeypatch):
        monkeypatch.setattr(panel._mode_group, "checkedButton", lambda: None)
        panel._on_mode_changed()


class TestModelToUI:
    def test_model_enabled_syncs_checkbox(self, model, panel):
        model.comet_stack_enabled = True
        assert panel._enabled_cb.isChecked()

    def test_model_tracking_mode_syncs_buttons(self, model, panel):
        model.comet_tracking_mode = "comet"
        assert panel._mode_buttons["comet"].isChecked()

    def test_model_blend_factor_syncs_slider(self, model, panel):
        model.comet_blend_factor = 0.4
        assert panel._blend_slider.value() == 40

    def test_model_blend_mode_shows_blend_row(self, model, panel):
        model.comet_tracking_mode = "blend"
        assert not panel._blend_row.isHidden()

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.comet_stack_enabled = True
        model.comet_blend_factor = 0.8
        model.reset()
        assert panel._enabled_cb.isChecked() == model.comet_stack_enabled
        assert abs(panel._blend_slider.value() - int(model.comet_blend_factor * 100)) <= 1
