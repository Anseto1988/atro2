"""Tests for SyntheticFlatPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.synthetic_flat_panel import SyntheticFlatPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = SyntheticFlatPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_enabled_cb_matches_model(self, model, panel):
        assert panel._enabled_cb.isChecked() == model.synthetic_flat_enabled

    def test_settings_group_disabled_when_not_enabled(self, panel):
        assert not panel._settings_group.isEnabled()

    def test_tile_spin_matches_model(self, model, panel):
        assert panel._tile_spin.value() == model.synthetic_flat_tile_size

    def test_sigma_spin_matches_model(self, model, panel):
        assert abs(panel._sigma_spin.value() - model.synthetic_flat_smoothing_sigma) < 0.01


class TestUIToModel:
    def test_enable_checkbox_updates_model(self, model, panel):
        panel._enabled_cb.setChecked(True)
        assert model.synthetic_flat_enabled is True

    def test_enable_checkbox_enables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        assert panel._settings_group.isEnabled()

    def test_disable_checkbox_disables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert not panel._settings_group.isEnabled()

    def test_tile_spin_updates_model(self, model, panel):
        panel._tile_spin.setValue(128)
        assert model.synthetic_flat_tile_size == 128

    def test_sigma_spin_updates_model(self, model, panel):
        panel._sigma_spin.setValue(12.0)
        assert abs(model.synthetic_flat_smoothing_sigma - 12.0) < 0.01


class TestModelToUI:
    def test_model_enabled_syncs_checkbox(self, model, panel):
        model.synthetic_flat_enabled = True
        assert panel._enabled_cb.isChecked()

    def test_model_tile_size_syncs_spin(self, model, panel):
        model.synthetic_flat_tile_size = 96
        assert panel._tile_spin.value() == 96

    def test_model_sigma_syncs_spin(self, model, panel):
        model.synthetic_flat_smoothing_sigma = 5.0
        assert abs(panel._sigma_spin.value() - 5.0) < 0.01

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.synthetic_flat_enabled = True
        model.synthetic_flat_tile_size = 128
        model.reset()
        assert panel._enabled_cb.isChecked() == model.synthetic_flat_enabled
        assert panel._tile_spin.value() == model.synthetic_flat_tile_size
