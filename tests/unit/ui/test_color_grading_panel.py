"""Tests for ColorGradingPanel widget."""
from __future__ import annotations

import pytest

from astroai.processing.color.color_grading import ColorGradingConfig
from astroai.ui.models import PipelineModel
from astroai.ui.widgets.color_grading_panel import ColorGradingPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = ColorGradingPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_all_spins_default_zero(self, panel):
        for attr, spin in panel._spins.items():
            assert spin.value() == pytest.approx(0.0), f"{attr} should default to 0.0"

    def test_spin_count(self, panel):
        """Panel should have exactly 9 spinboxes (3 zones × 3 channels)."""
        assert len(panel._spins) == 9

    def test_spin_range(self, panel):
        for attr, spin in panel._spins.items():
            assert spin.minimum() == pytest.approx(-0.5), f"{attr} min wrong"
            assert spin.maximum() == pytest.approx(0.5), f"{attr} max wrong"


class TestUIToModel:
    def test_shadow_r_spin_updates_model(self, model, panel):
        panel._spins["cg_shadow_r"].setValue(0.1)
        assert model.cg_shadow_r == pytest.approx(0.1)

    def test_midtone_g_spin_updates_model(self, model, panel):
        panel._spins["cg_midtone_g"].setValue(-0.2)
        assert model.cg_midtone_g == pytest.approx(-0.2)

    def test_highlight_b_spin_updates_model(self, model, panel):
        panel._spins["cg_highlight_b"].setValue(0.3)
        assert model.cg_highlight_b == pytest.approx(0.3)

    def test_signal_emitted_on_change(self, model, panel, qtbot):
        received = []
        panel.color_grading_changed.connect(received.append)
        panel._spins["cg_shadow_r"].setValue(0.05)
        assert len(received) == 1
        assert isinstance(received[0], ColorGradingConfig)

    def test_signal_config_values_correct(self, model, panel, qtbot):
        received = []
        panel.color_grading_changed.connect(received.append)
        panel._spins["cg_highlight_r"].setValue(0.4)
        cfg = received[-1]
        assert cfg.highlight_r == pytest.approx(0.4)


class TestModelToUI:
    def test_model_shadow_r_updates_spin(self, model, panel):
        model.cg_shadow_r = 0.15
        assert panel._spins["cg_shadow_r"].value() == pytest.approx(0.15)

    def test_model_midtone_b_updates_spin(self, model, panel):
        model.cg_midtone_b = -0.1
        assert panel._spins["cg_midtone_b"].value() == pytest.approx(-0.1)

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.cg_shadow_r = 0.3
        model.reset()
        # After reset the model value is still 0.3 (reset only changes step states)
        assert panel._spins["cg_shadow_r"].value() == pytest.approx(model.cg_shadow_r)


class TestResetButton:
    def test_reset_sets_all_spins_to_zero(self, model, panel):
        model.cg_shadow_r = 0.3
        model.cg_highlight_b = -0.2
        panel._on_reset()
        for attr, spin in panel._spins.items():
            assert spin.value() == pytest.approx(0.0), f"{attr} not reset"

    def test_reset_updates_model(self, model, panel):
        model.cg_shadow_g = 0.25
        panel._on_reset()
        assert model.cg_shadow_g == pytest.approx(0.0)
