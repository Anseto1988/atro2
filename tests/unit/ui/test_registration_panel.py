"""Tests for RegistrationPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.registration_panel import RegistrationPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = RegistrationPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_method_combo_has_two_items(self, panel):
        assert panel._method_combo.count() == 2

    def test_upsample_spin_default(self, model, panel):
        assert panel._upsample_spin.value() == model.registration_upsample_factor

    def test_ref_index_spin_default(self, model, panel):
        assert panel._ref_index_spin.value() == model.registration_reference_frame_index

    def test_method_combo_default_matches_model(self, model, panel):
        selected_data = panel._method_combo.currentData()
        assert selected_data == model.registration_method


class TestUIToModelSync:
    def test_changing_upsample_spin_updates_model(self, model, panel):
        panel._upsample_spin.setValue(25)
        assert model.registration_upsample_factor == 25

    def test_changing_ref_index_updates_model(self, model, panel):
        panel._ref_index_spin.setValue(3)
        assert model.registration_reference_frame_index == 3

    def test_changing_method_combo_updates_model(self, model, panel):
        idx = 1  # "phase_correlation"
        panel._method_combo.setCurrentIndex(idx)
        assert model.registration_method == panel._method_combo.itemData(idx)

    def test_on_method_changed_slot(self, model, panel):
        panel._on_method_changed(1)
        assert model.registration_method == "phase_correlation"

    def test_on_upsample_changed_slot(self, model, panel):
        panel._on_upsample_changed(50)
        assert model.registration_upsample_factor == 50

    def test_on_ref_index_changed_slot(self, model, panel):
        panel._on_ref_index_changed(7)
        assert model.registration_reference_frame_index == 7


class TestModelToUISync:
    def test_model_upsample_change_updates_spinbox(self, model, panel):
        model.registration_upsample_factor = 20
        assert panel._upsample_spin.value() == 20

    def test_model_ref_index_change_updates_spinbox(self, model, panel):
        model.registration_reference_frame_index = 5
        assert panel._ref_index_spin.value() == 5

    def test_model_method_change_updates_combo(self, model, panel):
        model.registration_method = "phase_correlation"
        assert panel._method_combo.currentData() == "phase_correlation"

    def test_model_method_star_updates_combo(self, model, panel):
        model.registration_method = "phase_correlation"
        model.registration_method = "star"
        assert panel._method_combo.currentData() == "star"

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.registration_upsample_factor = 30
        model.reset()
        assert panel._upsample_spin.value() == model.registration_upsample_factor

    def test_sync_no_signal_loop(self, model, panel):
        original = model.registration_upsample_factor
        model.registration_upsample_factor = 15
        model.registration_upsample_factor = original
        assert panel._upsample_spin.value() == original

    def test_unknown_method_defaults_to_index_zero(self, model, panel):
        model._registration_method = "unknown_method"
        panel._sync_from_model()
        assert panel._method_combo.currentIndex() == 0
