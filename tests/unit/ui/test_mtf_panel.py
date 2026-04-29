"""Tests for MTFStretchPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.mtf_stretch_panel import MTFStretchPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = MTFStretchPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_midpoint_spin_matches_model(self, model, panel):
        assert panel._midpoint_spin.value() == pytest.approx(model.mtf_midpoint)

    def test_shadows_spin_matches_model(self, model, panel):
        assert panel._shadows_spin.value() == pytest.approx(model.mtf_shadows_clipping)

    def test_highlights_spin_matches_model(self, model, panel):
        assert panel._highlights_spin.value() == pytest.approx(model.mtf_highlights)

    def test_auto_btf_button_exists(self, panel):
        assert panel._auto_btf_btn is not None

    def test_reset_button_exists(self, panel):
        assert panel._reset_btn is not None


class TestUIToModel:
    def test_midpoint_spin_updates_model(self, model, panel):
        panel._midpoint_spin.setValue(0.15)
        assert model.mtf_midpoint == pytest.approx(0.15, abs=0.005)

    def test_shadows_spin_updates_model(self, model, panel):
        panel._shadows_spin.setValue(0.05)
        assert model.mtf_shadows_clipping == pytest.approx(0.05, abs=0.001)

    def test_highlights_spin_updates_model(self, model, panel):
        panel._highlights_spin.setValue(0.99)
        assert model.mtf_highlights == pytest.approx(0.99, abs=0.001)

    def test_reset_restores_model_defaults(self, model, panel):
        model.mtf_midpoint = 0.1
        model.mtf_shadows_clipping = 0.05
        model.mtf_highlights = 0.99
        panel._reset_btn.click()
        assert model.mtf_midpoint == pytest.approx(0.25)
        assert model.mtf_shadows_clipping == pytest.approx(0.0)
        assert model.mtf_highlights == pytest.approx(1.0)

    def test_set_auto_btf_midpoint_updates_model(self, model, panel):
        panel.set_auto_btf_midpoint(0.18)
        assert model.mtf_midpoint == pytest.approx(0.18)

    def test_set_auto_btf_midpoint_updates_spin(self, model, panel):
        panel.set_auto_btf_midpoint(0.3)
        assert panel._midpoint_spin.value() == pytest.approx(0.3, abs=0.005)


class TestModelToUI:
    def test_model_midpoint_updates_spin(self, model, panel):
        model.mtf_midpoint = 0.12
        assert panel._midpoint_spin.value() == pytest.approx(0.12, abs=0.005)

    def test_model_shadows_updates_spin(self, model, panel):
        model.mtf_shadows_clipping = 0.03
        assert panel._shadows_spin.value() == pytest.approx(0.03, abs=0.001)

    def test_model_highlights_updates_spin(self, model, panel):
        model.mtf_highlights = 0.985
        assert panel._highlights_spin.value() == pytest.approx(0.985, abs=0.001)

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.mtf_midpoint = 0.1
        model.reset()
        assert panel._midpoint_spin.value() == pytest.approx(model.mtf_midpoint)


class TestSignals:
    def test_midpoint_change_emits_mtf_changed(self, panel, qtbot):
        with qtbot.waitSignal(panel.mtf_changed, timeout=500):
            panel._midpoint_spin.setValue(0.2)

    def test_auto_btf_button_emits_auto_btf_requested(self, panel, qtbot):
        with qtbot.waitSignal(panel.auto_btf_requested, timeout=500):
            panel._auto_btf_btn.click()

    def test_emitted_config_has_correct_midpoint(self, model, panel):
        received = []
        panel.mtf_changed.connect(received.append)
        panel._midpoint_spin.setValue(0.35)
        assert len(received) == 1
        assert received[0].midpoint == pytest.approx(0.35, abs=0.005)

    def test_set_auto_btf_midpoint_emits_mtf_changed(self, panel, qtbot):
        with qtbot.waitSignal(panel.mtf_changed, timeout=500):
            panel.set_auto_btf_midpoint(0.22)
