"""Tests for ColorCalibrationPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.color_calibration_panel import ColorCalibrationPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = ColorCalibrationPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_enabled_cb_matches_model(self, model, panel):
        assert panel._enabled_cb.isChecked() == model.color_calibration_enabled

    def test_settings_group_disabled_when_not_enabled(self, panel):
        assert not panel._settings_group.isEnabled()

    def test_catalog_combo_has_two_items(self, panel):
        assert panel._catalog_combo.count() == 2

    def test_radius_spin_matches_model(self, model, panel):
        assert panel._radius_spin.value() == model.color_calibration_sample_radius


class TestUIToModel:
    def test_enable_checkbox_updates_model(self, model, panel):
        panel._enabled_cb.setChecked(True)
        assert model.color_calibration_enabled is True

    def test_enable_checkbox_enables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        assert panel._settings_group.isEnabled()

    def test_disable_checkbox_disables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert not panel._settings_group.isEnabled()

    def test_catalog_combo_updates_model(self, model, panel):
        panel._catalog_combo.setCurrentIndex(1)
        value = panel._catalog_combo.itemData(1)
        assert model.color_calibration_catalog == value

    def test_radius_spin_updates_model(self, model, panel):
        panel._on_radius_changed(12)
        assert model.color_calibration_sample_radius == 12

    def test_on_catalog_changed_slot_ignores_empty(self, model, panel, monkeypatch):
        original = model.color_calibration_catalog
        monkeypatch.setattr(panel._catalog_combo, "itemData", lambda i: None)
        panel._on_catalog_changed(0)
        assert model.color_calibration_catalog == original


class TestModelToUI:
    def test_model_enabled_updates_checkbox(self, model, panel):
        model.color_calibration_enabled = True
        assert panel._enabled_cb.isChecked()

    def test_model_radius_updates_spin(self, model, panel):
        model.color_calibration_sample_radius = 10
        assert panel._radius_spin.value() == 10

    def test_model_catalog_unknown_keeps_index(self, model, panel):
        model._color_calibration_catalog = "unknown"
        panel._sync_from_model()
        assert panel._catalog_combo.currentIndex() >= 0

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.color_calibration_enabled = True
        model.reset()
        assert panel._enabled_cb.isChecked() == model.color_calibration_enabled
