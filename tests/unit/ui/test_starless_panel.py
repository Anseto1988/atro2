"""Tests for StarlessPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.starless_panel import StarlessPanel, _FORMAT_OPTIONS


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = StarlessPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_enabled_cb_matches_model(self, model, panel):
        assert panel._enabled_cb.isChecked() == model.starless_enabled

    def test_settings_group_disabled_when_not_enabled(self, panel):
        assert not panel._settings_group.isEnabled()

    def test_format_combo_has_correct_count(self, panel):
        assert panel._format_combo.count() == len(_FORMAT_OPTIONS)

    def test_strength_slider_matches_model(self, model, panel):
        expected = int(model.starless_strength * 100)
        assert panel._strength_slider.value() == expected

    def test_strength_label_matches_model(self, model, panel):
        expected = f"{int(model.starless_strength * 100)}%"
        assert panel._strength_value.text() == expected

    def test_mask_cb_matches_model(self, model, panel):
        assert panel._mask_cb.isChecked() == model.save_star_mask


class TestUIToModel:
    def test_enable_checkbox_updates_model(self, model, panel):
        panel._enabled_cb.setChecked(True)
        assert model.starless_enabled is True

    def test_enable_checkbox_enables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        assert panel._settings_group.isEnabled()

    def test_disable_checkbox_disables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert not panel._settings_group.isEnabled()

    def test_strength_slider_updates_model(self, model, panel):
        panel._strength_slider.setValue(75)
        assert abs(model.starless_strength - 0.75) < 0.01

    def test_strength_slider_updates_label(self, panel):
        panel._strength_slider.setValue(60)
        assert panel._strength_value.text() == "60%"

    def test_format_combo_updates_model(self, model, panel):
        panel._format_combo.setCurrentIndex(1)
        expected_fmt = _FORMAT_OPTIONS[1][0]
        assert model.starless_format == expected_fmt

    def test_mask_checkbox_updates_model(self, model, panel):
        panel._on_mask_changed(True)
        assert model.save_star_mask is True
        panel._on_mask_changed(False)
        assert model.save_star_mask is False


class TestModelToUI:
    def test_model_enabled_syncs_checkbox(self, model, panel):
        model.starless_enabled = True
        assert panel._enabled_cb.isChecked()

    def test_model_strength_syncs_slider(self, model, panel):
        model.starless_strength = 0.5
        assert panel._strength_slider.value() == 50

    def test_model_format_syncs_combo(self, model, panel):
        model.starless_format = _FORMAT_OPTIONS[2][0]
        idx = panel._format_combo.findData(_FORMAT_OPTIONS[2][0])
        assert panel._format_combo.currentIndex() == idx

    def test_model_mask_syncs_checkbox(self, model, panel):
        model.save_star_mask = True
        assert panel._mask_cb.isChecked()

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.starless_enabled = True
        model.starless_strength = 0.3
        model.reset()
        assert panel._enabled_cb.isChecked() == model.starless_enabled
        assert abs(panel._strength_slider.value() - int(model.starless_strength * 100)) <= 1
