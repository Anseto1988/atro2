"""Tests for StarProcessingPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.star_processing_panel import StarProcessingPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = StarProcessingPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_reduce_cb_matches_model(self, model, panel):
        assert panel._reduce_cb.isChecked() == model.star_reduce_enabled

    def test_factor_slider_matches_model(self, model, panel):
        expected = int(model.star_reduce_factor * 100)
        assert panel._factor_slider.value() == expected

    def test_factor_label_matches_model(self, model, panel):
        expected = f"{int(model.star_reduce_factor * 100)}%"
        assert panel._factor_value.text() == expected

    def test_sigma_spin_matches_model(self, model, panel):
        assert abs(panel._sigma_spin.value() - model.star_detection_sigma) < 0.01

    def test_min_area_spin_matches_model(self, model, panel):
        assert panel._min_area_spin.value() == model.star_min_area

    def test_max_area_spin_matches_model(self, model, panel):
        assert panel._max_area_spin.value() == model.star_max_area

    def test_dilation_spin_matches_model(self, model, panel):
        assert panel._dilation_spin.value() == model.star_mask_dilation


class TestUIToModel:
    def test_reduce_cb_updates_model(self, model, panel):
        panel._on_reduce_enabled_changed(True)
        assert model.star_reduce_enabled is True
        panel._on_reduce_enabled_changed(False)
        assert model.star_reduce_enabled is False

    def test_factor_slider_updates_model(self, model, panel):
        panel._factor_slider.setValue(70)
        assert abs(model.star_reduce_factor - 0.70) < 0.01

    def test_factor_slider_updates_label(self, panel):
        panel._factor_slider.setValue(30)
        assert panel._factor_value.text() == "30%"

    def test_sigma_spin_updates_model(self, model, panel):
        panel._sigma_spin.setValue(3.0)
        assert abs(model.star_detection_sigma - 3.0) < 0.01

    def test_min_area_spin_updates_model(self, model, panel):
        panel._min_area_spin.setValue(10)
        assert model.star_min_area == 10

    def test_max_area_spin_updates_model(self, model, panel):
        panel._max_area_spin.setValue(2000)
        assert model.star_max_area == 2000

    def test_dilation_spin_updates_model(self, model, panel):
        panel._dilation_spin.setValue(5)
        assert model.star_mask_dilation == 5


class TestModelToUI:
    def test_model_reduce_enabled_syncs_cb(self, model, panel):
        model.star_reduce_enabled = True
        assert panel._reduce_cb.isChecked()

    def test_model_factor_syncs_slider(self, model, panel):
        model.star_reduce_factor = 0.25
        assert panel._factor_slider.value() == 25

    def test_model_sigma_syncs_spin(self, model, panel):
        model.star_detection_sigma = 5.0
        assert abs(panel._sigma_spin.value() - 5.0) < 0.01

    def test_model_min_area_syncs_spin(self, model, panel):
        model.star_min_area = 8
        assert panel._min_area_spin.value() == 8

    def test_model_max_area_syncs_spin(self, model, panel):
        model.star_max_area = 3000
        assert panel._max_area_spin.value() == 3000

    def test_model_dilation_syncs_spin(self, model, panel):
        model.star_mask_dilation = 7
        assert panel._dilation_spin.value() == 7

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.star_reduce_enabled = True
        model.star_detection_sigma = 6.0
        model.reset()
        assert panel._reduce_cb.isChecked() == model.star_reduce_enabled
        assert abs(panel._sigma_spin.value() - model.star_detection_sigma) < 0.01
