"""Tests for DenoisePanel widget."""
from __future__ import annotations

import pytest

from astroai.core.noise_estimator import NoiseEstimate
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


class TestAutoDetect:
    def test_auto_btn_exists(self, panel):
        assert panel._auto_btn is not None

    def test_auto_btn_emits_signal(self, panel, qtbot):
        with qtbot.waitSignal(panel.noise_detect_requested, timeout=500):
            panel._auto_btn.click()

    def test_apply_estimate_sets_strength(self, model, panel):
        est = NoiseEstimate(sky_sigma=0.025, snr_db=15.0, noise_level_pct=25.0, suggested_strength=0.6)
        panel.apply_estimate(est)
        assert model.denoise_strength == pytest.approx(0.6)

    def test_apply_estimate_updates_noise_label(self, panel):
        est = NoiseEstimate(sky_sigma=0.025, snr_db=15.0, noise_level_pct=25.0, suggested_strength=0.6)
        panel.apply_estimate(est)
        text = panel._noise_label.text()
        assert "σ=" in text
        assert "SNR=" in text

    def test_apply_estimate_low_strength(self, model, panel):
        est = NoiseEstimate(sky_sigma=0.003, snr_db=30.0, noise_level_pct=3.0, suggested_strength=0.2)
        panel.apply_estimate(est)
        assert model.denoise_strength == pytest.approx(0.2)

    def test_noise_label_initially_empty(self, panel):
        assert panel._noise_label.text() == ""

    def test_apply_estimate_shows_sigma_value(self, panel):
        est = NoiseEstimate(sky_sigma=0.0123, snr_db=22.5, noise_level_pct=12.3, suggested_strength=0.35)
        panel.apply_estimate(est)
        assert "0.0123" in panel._noise_label.text()
