"""Tests for StackingPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.stacking_panel import StackingPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = StackingPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_method_combo_has_three_items(self, panel):
        assert panel._method_combo.count() == 3

    def test_method_combo_default_matches_model(self, model, panel):
        assert panel._method_combo.currentText() == model.stacking_method

    def test_sigma_low_default_matches_model(self, model, panel):
        assert panel._sigma_low_spin.value() == pytest.approx(model.stacking_sigma_low)

    def test_sigma_high_default_matches_model(self, model, panel):
        assert panel._sigma_high_spin.value() == pytest.approx(model.stacking_sigma_high)

    def test_sigma_spins_enabled_for_sigma_clip(self, model, panel):
        model.stacking_method = "sigma_clip"
        panel._sync_from_model()
        assert panel._sigma_low_spin.isEnabled()
        assert panel._sigma_high_spin.isEnabled()

    def test_sigma_spins_disabled_for_mean(self, model, panel):
        model.stacking_method = "mean"
        panel._sync_from_model()
        assert not panel._sigma_low_spin.isEnabled()
        assert not panel._sigma_high_spin.isEnabled()


class TestUIToModelSync:
    def test_changing_method_combo_updates_model(self, model, panel):
        panel._method_combo.setCurrentText("mean")
        assert model.stacking_method == "mean"

    def test_changing_sigma_low_updates_model(self, model, panel):
        panel._sigma_low_spin.setValue(1.5)
        assert model.stacking_sigma_low == pytest.approx(1.5)

    def test_changing_sigma_high_updates_model(self, model, panel):
        panel._sigma_high_spin.setValue(4.0)
        assert model.stacking_sigma_high == pytest.approx(4.0)

    def test_on_method_changed_slot_disables_sigma(self, model, panel):
        panel._on_method_changed("median")
        assert model.stacking_method == "median"
        assert not panel._sigma_low_spin.isEnabled()

    def test_on_method_changed_sigma_clip_enables_spins(self, model, panel):
        panel._on_method_changed("mean")
        panel._on_method_changed("sigma_clip")
        assert panel._sigma_low_spin.isEnabled()
        assert panel._sigma_high_spin.isEnabled()

    def test_on_sigma_low_changed_slot(self, model, panel):
        panel._on_sigma_low_changed(2.0)
        assert model.stacking_sigma_low == pytest.approx(2.0)

    def test_on_sigma_high_changed_slot(self, model, panel):
        panel._on_sigma_high_changed(3.5)
        assert model.stacking_sigma_high == pytest.approx(3.5)


class TestModelToUISync:
    def test_model_method_change_updates_combo(self, model, panel):
        model.stacking_method = "median"
        assert panel._method_combo.currentText() == "median"

    def test_model_sigma_low_change_updates_spin(self, model, panel):
        model.stacking_sigma_low = 1.0
        assert panel._sigma_low_spin.value() == pytest.approx(1.0)

    def test_model_sigma_high_change_updates_spin(self, model, panel):
        model.stacking_sigma_high = 4.5
        assert panel._sigma_high_spin.value() == pytest.approx(4.5)

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.stacking_method = "mean"
        model.reset()
        assert panel._method_combo.currentText() == model.stacking_method

    def test_unknown_method_falls_back_to_index_zero(self, model, panel):
        model._stacking_method = "unknown"
        panel._sync_from_model()
        assert panel._method_combo.currentIndex() == 0
