"""Tests for DenoisePanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.denoise_panel import DenoisePanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = DenoisePanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_strength_spin_matches_model(self, model, panel):
        assert panel._strength_spin.value() == pytest.approx(model.denoise_strength)

    def test_tile_spin_matches_model(self, model, panel):
        assert panel._tile_spin.value() == model.denoise_tile_size

    def test_overlap_spin_matches_model(self, model, panel):
        assert panel._overlap_spin.value() == model.denoise_tile_overlap


class TestUIToModel:
    def test_strength_spin_updates_model(self, model, panel):
        panel._strength_spin.setValue(0.3)
        assert model.denoise_strength == pytest.approx(0.3)

    def test_tile_spin_updates_model(self, model, panel):
        panel._tile_spin.setValue(256)
        assert model.denoise_tile_size == 256

    def test_overlap_spin_updates_model(self, model, panel):
        panel._overlap_spin.setValue(32)
        assert model.denoise_tile_overlap == 32

    def test_on_strength_changed_slot(self, model, panel):
        panel._on_strength_changed(0.7)
        assert model.denoise_strength == pytest.approx(0.7)

    def test_on_tile_size_changed_slot(self, model, panel):
        panel._on_tile_size_changed(128)
        assert model.denoise_tile_size == 128

    def test_on_overlap_changed_slot(self, model, panel):
        panel._on_overlap_changed(16)
        assert model.denoise_tile_overlap == 16


class TestModelToUI:
    def test_model_strength_updates_spin(self, model, panel):
        model.denoise_strength = 0.5
        assert panel._strength_spin.value() == pytest.approx(0.5)

    def test_model_tile_size_updates_spin(self, model, panel):
        model.denoise_tile_size = 128
        assert panel._tile_spin.value() == 128

    def test_model_overlap_updates_spin(self, model, panel):
        model.denoise_tile_overlap = 32
        assert panel._overlap_spin.value() == 32

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.denoise_strength = 0.3
        model.reset()
        assert panel._strength_spin.value() == pytest.approx(model.denoise_strength)
