"""Unit tests for AsinHStretchPanel widget."""
from __future__ import annotations

import pytest

from astroai.processing.stretch.asinh_stretcher import AsinHConfig
from astroai.ui.models import PipelineModel
from astroai.ui.widgets.asinh_stretch_panel import AsinHStretchPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = AsinHStretchPanel(model)
    qtbot.addWidget(w)
    return w


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_stretch_factor_spin_matches_model(self, model, panel):
        assert panel._stretch_factor_spin.value() == pytest.approx(model.asinh_stretch_factor)

    def test_black_point_spin_matches_model(self, model, panel):
        assert panel._black_point_spin.value() == pytest.approx(model.asinh_black_point)

    def test_linked_cb_matches_model(self, model, panel):
        assert panel._linked_cb.isChecked() == model.asinh_linked


# ---------------------------------------------------------------------------
# UI → Model
# ---------------------------------------------------------------------------

class TestUIToModel:
    def test_stretch_factor_spin_updates_model(self, model, panel):
        panel._stretch_factor_spin.setValue(10.0)
        assert model.asinh_stretch_factor == pytest.approx(10.0)

    def test_black_point_spin_updates_model(self, model, panel):
        panel._black_point_spin.setValue(0.1)
        assert model.asinh_black_point == pytest.approx(0.1)

    def test_linked_cb_updates_model_false(self, model, panel):
        panel._linked_cb.setChecked(False)
        assert model.asinh_linked is False

    def test_linked_cb_updates_model_true(self, model, panel):
        model.asinh_linked = False
        panel._linked_cb.setChecked(True)
        assert model.asinh_linked is True

    def test_on_stretch_factor_changed_slot(self, model, panel):
        panel._on_stretch_factor_changed(50.0)
        assert model.asinh_stretch_factor == pytest.approx(50.0)

    def test_on_black_point_changed_slot(self, model, panel):
        panel._on_black_point_changed(0.25)
        assert model.asinh_black_point == pytest.approx(0.25)

    def test_on_linked_changed_slot(self, model, panel):
        panel._on_linked_changed(False)
        assert model.asinh_linked is False


# ---------------------------------------------------------------------------
# Model → UI
# ---------------------------------------------------------------------------

class TestModelToUI:
    def test_model_stretch_factor_updates_spin(self, model, panel):
        model.asinh_stretch_factor = 20.0
        assert panel._stretch_factor_spin.value() == pytest.approx(20.0)

    def test_model_black_point_updates_spin(self, model, panel):
        model.asinh_black_point = 0.15
        assert panel._black_point_spin.value() == pytest.approx(0.15)

    def test_model_linked_updates_checkbox(self, model, panel):
        model.asinh_linked = False
        assert panel._linked_cb.isChecked() is False

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.asinh_stretch_factor = 50.0
        model.reset()
        assert panel._stretch_factor_spin.value() == pytest.approx(model.asinh_stretch_factor)


# ---------------------------------------------------------------------------
# Reset button
# ---------------------------------------------------------------------------

class TestResetButton:
    def test_reset_restores_defaults(self, model, panel):
        model.asinh_stretch_factor = 100.0
        model.asinh_black_point = 0.3
        model.asinh_linked = False
        panel._on_reset()
        assert model.asinh_stretch_factor == pytest.approx(1.0)
        assert model.asinh_black_point == pytest.approx(0.0)
        assert model.asinh_linked is True


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class TestSignal:
    def test_signal_emitted_on_stretch_factor_change(self, model, panel, qtbot):
        with qtbot.waitSignal(panel.asinh_stretch_changed, timeout=1000):
            panel._stretch_factor_spin.setValue(5.0)

    def test_signal_carries_correct_config(self, model, panel, qtbot):
        received: list[AsinHConfig] = []
        panel.asinh_stretch_changed.connect(received.append)
        panel._on_stretch_factor_changed(7.0)
        assert len(received) == 1
        assert received[0].stretch_factor == pytest.approx(7.0)
