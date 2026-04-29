"""Tests for WhiteBalancePanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.white_balance_panel import WhiteBalancePanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = WhiteBalancePanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_red_spin_matches_model(self, model, panel):
        assert panel._spins["wb_red"].value() == pytest.approx(model.wb_red)

    def test_green_spin_matches_model(self, model, panel):
        assert panel._spins["wb_green"].value() == pytest.approx(model.wb_green)

    def test_blue_spin_matches_model(self, model, panel):
        assert panel._spins["wb_blue"].value() == pytest.approx(model.wb_blue)

    def test_all_spins_default_to_1(self, panel):
        for spin in panel._spins.values():
            assert spin.value() == pytest.approx(1.0)

    def test_reset_button_exists(self, panel):
        assert panel._reset_btn is not None


class TestUIToModel:
    def test_red_spin_updates_model(self, model, panel):
        panel._spins["wb_red"].setValue(1.5)
        assert model.wb_red == pytest.approx(1.5)

    def test_green_spin_updates_model(self, model, panel):
        panel._spins["wb_green"].setValue(0.8)
        assert model.wb_green == pytest.approx(0.8)

    def test_blue_spin_updates_model(self, model, panel):
        panel._spins["wb_blue"].setValue(2.0)
        assert model.wb_blue == pytest.approx(2.0)

    def test_on_spin_changed_slot(self, model, panel):
        panel._on_spin_changed("wb_red", 1.3)
        assert model.wb_red == pytest.approx(1.3)

    def test_reset_button_resets_model(self, model, panel):
        model.wb_red = 2.0
        model.wb_green = 0.5
        model.wb_blue = 3.0
        panel._reset_btn.click()
        assert model.wb_red == pytest.approx(1.0)
        assert model.wb_green == pytest.approx(1.0)
        assert model.wb_blue == pytest.approx(1.0)


class TestModelToUI:
    def test_model_red_updates_spin(self, model, panel):
        model.wb_red = 1.8
        assert panel._spins["wb_red"].value() == pytest.approx(1.8)

    def test_model_green_updates_spin(self, model, panel):
        model.wb_green = 0.6
        assert panel._spins["wb_green"].value() == pytest.approx(0.6)

    def test_model_blue_updates_spin(self, model, panel):
        model.wb_blue = 2.5
        assert panel._spins["wb_blue"].value() == pytest.approx(2.5)

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.wb_red = 1.9
        model.reset()
        assert panel._spins["wb_red"].value() == pytest.approx(model.wb_red)


class TestSignals:
    def test_spin_change_emits_white_balance_changed(self, panel, qtbot):
        with qtbot.waitSignal(panel.white_balance_changed, timeout=500):
            panel._spins["wb_red"].setValue(1.4)

    def test_emitted_config_has_correct_values(self, model, panel, qtbot):
        received = []
        panel.white_balance_changed.connect(received.append)
        panel._spins["wb_blue"].setValue(2.2)
        assert len(received) == 1
        assert received[0].blue_factor == pytest.approx(2.2, abs=0.01)
