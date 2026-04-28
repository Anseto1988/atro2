"""Tests for DrizzlePanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.drizzle_panel import DrizzlePanel, _DROP_SIZE_OPTIONS


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = DrizzlePanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_enabled_cb_matches_model(self, model, panel):
        assert panel._enabled_cb.isChecked() == model.drizzle_enabled

    def test_settings_group_disabled_when_not_enabled(self, panel):
        assert not panel._settings_group.isEnabled()

    def test_drop_buttons_count(self, panel):
        assert len(panel._drop_buttons) == len(_DROP_SIZE_OPTIONS)

    def test_scale_spin_matches_model(self, model, panel):
        assert abs(panel._scale_spin.value() - model.drizzle_scale) < 0.01

    def test_pixfrac_slider_matches_model(self, model, panel):
        expected = int(model.drizzle_pixfrac * 100)
        assert panel._pixfrac_slider.value() == expected

    def test_pixfrac_label_matches_model(self, model, panel):
        expected = f"{model.drizzle_pixfrac:.1f}"
        assert panel._pixfrac_value.text() == expected


class TestUIToModel:
    def test_enable_checkbox_updates_model(self, model, panel):
        panel._enabled_cb.setChecked(True)
        assert model.drizzle_enabled is True

    def test_enable_checkbox_enables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        assert panel._settings_group.isEnabled()

    def test_disable_checkbox_disables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert not panel._settings_group.isEnabled()

    def test_scale_spin_updates_model(self, model, panel):
        panel._scale_spin.setValue(2.0)
        assert abs(model.drizzle_scale - 2.0) < 0.01

    def test_pixfrac_slider_updates_model(self, model, panel):
        panel._pixfrac_slider.setValue(75)
        assert abs(model.drizzle_pixfrac - 0.75) < 0.01

    def test_pixfrac_slider_updates_label(self, panel):
        panel._pixfrac_slider.setValue(50)
        assert panel._pixfrac_value.text() == "0.5"

    def test_drop_button_updates_model(self, model, panel):
        panel._drop_buttons[0].setChecked(True)
        panel._on_drop_size_changed()
        assert abs(model.drizzle_drop_size - _DROP_SIZE_OPTIONS[0]) < 0.01

    def test_drop_button_second_option(self, model, panel):
        panel._drop_buttons[1].setChecked(True)
        panel._on_drop_size_changed()
        assert abs(model.drizzle_drop_size - _DROP_SIZE_OPTIONS[1]) < 0.01


class TestModelToUI:
    def test_model_enabled_syncs_checkbox(self, model, panel):
        model.drizzle_enabled = True
        assert panel._enabled_cb.isChecked()

    def test_model_scale_syncs_spin(self, model, panel):
        model.drizzle_scale = 1.5
        assert abs(panel._scale_spin.value() - 1.5) < 0.01

    def test_model_pixfrac_syncs_slider(self, model, panel):
        model.drizzle_pixfrac = 0.6
        assert panel._pixfrac_slider.value() == 60

    def test_model_drop_size_syncs_radio(self, model, panel):
        model.drizzle_drop_size = _DROP_SIZE_OPTIONS[2]
        checked_vals = [float(b.text()) for b in panel._drop_buttons if b.isChecked()]
        assert _DROP_SIZE_OPTIONS[2] in checked_vals

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.drizzle_enabled = True
        model.drizzle_scale = 2.5
        model.reset()
        assert panel._enabled_cb.isChecked() == model.drizzle_enabled
        assert abs(panel._scale_spin.value() - model.drizzle_scale) < 0.01
