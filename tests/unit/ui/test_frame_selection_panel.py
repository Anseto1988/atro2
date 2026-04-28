"""Tests for FrameSelectionPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.frame_selection_panel import FrameSelectionPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = FrameSelectionPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_enabled_cb_matches_model(self, model, panel):
        assert panel._enabled_cb.isChecked() == model.frame_selection_enabled

    def test_settings_group_disabled_when_not_enabled(self, panel):
        assert not panel._settings_group.isEnabled()

    def test_score_spin_matches_model(self, model, panel):
        assert abs(panel._score_spin.value() - model.frame_selection_min_score) < 0.01

    def test_reject_spin_matches_model(self, model, panel):
        assert abs(panel._reject_spin.value() - model.frame_selection_max_rejected_fraction) < 0.01


class TestUIToModel:
    def test_enable_checkbox_updates_model(self, model, panel):
        panel._enabled_cb.setChecked(True)
        assert model.frame_selection_enabled is True

    def test_enable_checkbox_enables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        assert panel._settings_group.isEnabled()

    def test_disable_checkbox_disables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert not panel._settings_group.isEnabled()

    def test_score_spin_updates_model(self, model, panel):
        panel._score_spin.setValue(0.75)
        assert abs(model.frame_selection_min_score - 0.75) < 0.01

    def test_reject_spin_updates_model(self, model, panel):
        panel._reject_spin.setValue(0.6)
        assert abs(model.frame_selection_max_rejected_fraction - 0.6) < 0.01


class TestModelToUI:
    def test_model_enabled_syncs_checkbox(self, model, panel):
        model.frame_selection_enabled = True
        assert panel._enabled_cb.isChecked()

    def test_model_min_score_syncs_spin(self, model, panel):
        model.frame_selection_min_score = 0.3
        assert abs(panel._score_spin.value() - 0.3) < 0.01

    def test_model_max_rejected_syncs_spin(self, model, panel):
        model.frame_selection_max_rejected_fraction = 0.5
        assert abs(panel._reject_spin.value() - 0.5) < 0.01

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.frame_selection_enabled = True
        model.frame_selection_min_score = 0.9
        model.reset()
        assert panel._enabled_cb.isChecked() == model.frame_selection_enabled
        assert abs(panel._score_spin.value() - model.frame_selection_min_score) < 0.01
