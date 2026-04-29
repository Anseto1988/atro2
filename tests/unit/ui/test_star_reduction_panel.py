"""Unit tests for astroai.ui.widgets.star_reduction_panel."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

from astroai.processing.stars.star_reducer import StarReductionConfig
from astroai.ui.models import PipelineModel
from astroai.ui.widgets.star_reduction_panel import StarReductionPanel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def model(qapp):
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = StarReductionPanel(model)
    qtbot.addWidget(w)
    return w


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStarReductionPanelConstruction:
    def test_panel_is_widget(self, panel):
        from PySide6.QtWidgets import QWidget
        assert isinstance(panel, QWidget)

    def test_panel_has_amount_spinbox(self, panel):
        assert isinstance(panel._spin_amount, QDoubleSpinBox)

    def test_panel_has_radius_spinbox(self, panel):
        assert isinstance(panel._spin_radius, QSpinBox)

    def test_panel_has_threshold_spinbox(self, panel):
        assert isinstance(panel._spin_threshold, QDoubleSpinBox)

    def test_initial_amount_matches_model(self, panel, model):
        assert panel._spin_amount.value() == pytest.approx(model.star_reduction_amount)

    def test_initial_radius_matches_model(self, panel, model):
        assert panel._spin_radius.value() == model.star_reduction_radius

    def test_initial_threshold_matches_model(self, panel, model):
        assert panel._spin_threshold.value() == pytest.approx(model.star_reduction_threshold)


class TestStarReductionPanelInteraction:
    def test_change_amount_updates_model(self, panel, model, qtbot):
        panel._spin_amount.setValue(0.8)
        assert model.star_reduction_amount == pytest.approx(0.8)

    def test_change_radius_updates_model(self, panel, model, qtbot):
        panel._spin_radius.setValue(5)
        assert model.star_reduction_radius == 5

    def test_change_threshold_updates_model(self, panel, model, qtbot):
        panel._spin_threshold.setValue(0.7)
        assert model.star_reduction_threshold == pytest.approx(0.7)

    def test_signal_emitted_on_amount_change(self, panel, qtbot):
        received = []
        panel.star_reduction_changed.connect(received.append)
        panel._spin_amount.setValue(0.3)
        assert len(received) == 1
        assert isinstance(received[0], StarReductionConfig)

    def test_reset_restores_defaults(self, panel, model, qtbot):
        panel._spin_amount.setValue(0.9)
        panel._spin_radius.setValue(7)
        panel._spin_threshold.setValue(0.8)
        panel._reset_btn.click()
        defaults = StarReductionConfig()
        assert model.star_reduction_amount == pytest.approx(defaults.amount)
        assert model.star_reduction_radius == defaults.radius
        assert model.star_reduction_threshold == pytest.approx(defaults.threshold)

    def test_model_change_syncs_to_panel(self, panel, model, qtbot):
        model.star_reduction_amount = 0.25
        assert panel._spin_amount.value() == pytest.approx(0.25)
