"""Tests for StretchPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.stretch_panel import StretchPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = StretchPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_bg_spin_matches_model(self, model, panel):
        assert panel._bg_spin.value() == pytest.approx(model.stretch_target_background)

    def test_sigma_spin_matches_model(self, model, panel):
        assert panel._sigma_spin.value() == pytest.approx(model.stretch_shadow_clipping_sigmas)

    def test_linked_cb_matches_model(self, model, panel):
        assert panel._linked_cb.isChecked() == model.stretch_linked_channels


class TestUIToModel:
    def test_bg_spin_updates_model(self, model, panel):
        panel._bg_spin.setValue(0.3)
        assert model.stretch_target_background == pytest.approx(0.3)

    def test_sigma_spin_updates_model(self, model, panel):
        panel._sigma_spin.setValue(-3.0)
        assert model.stretch_shadow_clipping_sigmas == pytest.approx(-3.0)

    def test_linked_cb_updates_model(self, model, panel):
        original = model.stretch_linked_channels
        panel._linked_cb.setChecked(not original)
        assert model.stretch_linked_channels is not original

    def test_on_bg_changed_slot(self, model, panel):
        panel._on_bg_changed(0.2)
        assert model.stretch_target_background == pytest.approx(0.2)

    def test_on_sigma_changed_slot(self, model, panel):
        panel._on_sigma_changed(-5.0)
        assert model.stretch_shadow_clipping_sigmas == pytest.approx(-5.0)

    def test_on_linked_changed_slot(self, model, panel):
        panel._on_linked_changed(False)
        assert model.stretch_linked_channels is False


class TestModelToUI:
    def test_model_bg_updates_spin(self, model, panel):
        model.stretch_target_background = 0.15
        assert panel._bg_spin.value() == pytest.approx(0.15)

    def test_model_sigma_updates_spin(self, model, panel):
        model.stretch_shadow_clipping_sigmas = -4.0
        assert panel._sigma_spin.value() == pytest.approx(-4.0)

    def test_model_linked_updates_checkbox(self, model, panel):
        original = model.stretch_linked_channels
        model.stretch_linked_channels = not original
        assert panel._linked_cb.isChecked() is not original

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.stretch_target_background = 0.5
        model.reset()
        assert panel._bg_spin.value() == pytest.approx(model.stretch_target_background)
