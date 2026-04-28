"""Tests for DeconvolutionPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.deconvolution_panel import DeconvolutionPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = DeconvolutionPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_enabled_cb_matches_model(self, model, panel):
        assert panel._enabled_cb.isChecked() == model.deconvolution_enabled

    def test_settings_group_disabled_when_not_enabled(self, panel):
        assert not panel._settings_group.isEnabled()

    def test_iter_spin_matches_model(self, model, panel):
        assert panel._iter_spin.value() == model.deconvolution_iterations

    def test_sigma_spin_matches_model(self, model, panel):
        assert panel._sigma_spin.value() == pytest.approx(model.deconvolution_psf_sigma)


class TestUIToModel:
    def test_enable_checkbox_updates_model(self, model, panel):
        panel._enabled_cb.setChecked(True)
        assert model.deconvolution_enabled is True

    def test_enable_checkbox_enables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        assert panel._settings_group.isEnabled()

    def test_disable_checkbox_disables_settings_group(self, panel):
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert not panel._settings_group.isEnabled()

    def test_iter_spin_updates_model(self, model, panel):
        panel._on_iterations_changed(20)
        assert model.deconvolution_iterations == 20

    def test_sigma_spin_updates_model(self, model, panel):
        panel._on_sigma_changed(2.5)
        assert model.deconvolution_psf_sigma == pytest.approx(2.5)

    def test_on_enabled_changed_slot(self, model, panel):
        panel._on_enabled_changed(True)
        assert model.deconvolution_enabled is True
        assert panel._settings_group.isEnabled()


class TestModelToUI:
    def test_model_enabled_updates_checkbox(self, model, panel):
        model.deconvolution_enabled = True
        assert panel._enabled_cb.isChecked()

    def test_model_iterations_updates_spin(self, model, panel):
        model.deconvolution_iterations = 15
        assert panel._iter_spin.value() == 15

    def test_model_sigma_updates_spin(self, model, panel):
        model.deconvolution_psf_sigma = 3.0
        assert panel._sigma_spin.value() == pytest.approx(3.0)

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.deconvolution_enabled = True
        model.reset()
        assert panel._enabled_cb.isChecked() == model.deconvolution_enabled
