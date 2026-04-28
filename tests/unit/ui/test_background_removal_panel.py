"""Tests for BackgroundRemovalPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.background_removal_panel import BackgroundRemovalPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = BackgroundRemovalPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_enabled_cb_matches_model(self, model, panel):
        assert panel._enabled_cb.isChecked() == model.background_removal_enabled

    def test_settings_group_disabled_when_not_enabled(self, model, panel):
        assert not panel._settings_group.isEnabled()

    def test_tile_spin_matches_model(self, model, panel):
        assert panel._tile_spin.value() == model.background_removal_tile_size

    def test_preserve_median_cb_matches_model(self, model, panel):
        assert panel._preserve_median_cb.isChecked() == model.background_removal_preserve_median

    def test_method_combo_has_two_items(self, panel):
        assert panel._method_combo.count() == 2


class TestUIToModel:
    def test_enable_checkbox_updates_model(self, model, panel):
        panel._enabled_cb.setChecked(True)
        assert model.background_removal_enabled is True

    def test_enable_checkbox_enables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        assert panel._settings_group.isEnabled()

    def test_disable_checkbox_disables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert not panel._settings_group.isEnabled()

    def test_tile_spin_updates_model(self, model, panel):
        panel._on_tile_size_changed(128)
        assert model.background_removal_tile_size == 128

    def test_preserve_median_cb_updates_model(self, model, panel):
        original = model.background_removal_preserve_median
        panel._preserve_median_cb.setChecked(not original)
        assert model.background_removal_preserve_median is not original

    def test_method_combo_updates_model(self, model, panel):
        panel._method_combo.setCurrentIndex(1)
        key = panel._method_combo.itemData(1)
        assert model.background_removal_method == str(key)

    def test_on_method_changed_slot_with_none_data(self, model, panel, monkeypatch):
        monkeypatch.setattr(panel._method_combo, "itemData", lambda i: None)
        original = model.background_removal_method
        panel._on_method_changed(0)
        assert model.background_removal_method == original


class TestModelToUI:
    def test_model_enabled_updates_checkbox(self, model, panel):
        model.background_removal_enabled = True
        assert panel._enabled_cb.isChecked()

    def test_model_tile_size_updates_spin(self, model, panel):
        model.background_removal_tile_size = 64
        assert panel._tile_spin.value() == 64

    def test_model_preserve_median_updates_checkbox(self, model, panel):
        original = model.background_removal_preserve_median
        model.background_removal_preserve_median = not original
        assert panel._preserve_median_cb.isChecked() is not original

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.background_removal_enabled = True
        model.reset()
        assert panel._enabled_cb.isChecked() == model.background_removal_enabled
