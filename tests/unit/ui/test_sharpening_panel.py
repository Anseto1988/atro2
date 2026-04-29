"""Tests for SharpeningPanel widget."""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.sharpening_panel import SharpeningPanel


@pytest.fixture()
def model() -> PipelineModel:
    return PipelineModel()


@pytest.fixture()
def panel(model: PipelineModel, qtbot) -> SharpeningPanel:
    w = SharpeningPanel(model)
    qtbot.addWidget(w)
    return w


class TestSharpeningPanelBasic:
    def test_has_preview_step(self) -> None:
        assert SharpeningPanel.PREVIEW_STEP == "sharpening"

    def test_initial_values_match_model_defaults(self, panel: SharpeningPanel, model: PipelineModel) -> None:
        assert panel._radius_spin.value() == pytest.approx(model.sharpening_radius)
        assert panel._amount_spin.value() == pytest.approx(model.sharpening_amount)
        assert panel._threshold_spin.value() == pytest.approx(model.sharpening_threshold)

    def test_radius_spin_range(self, panel: SharpeningPanel) -> None:
        assert panel._radius_spin.minimum() == pytest.approx(0.1)
        assert panel._radius_spin.maximum() == pytest.approx(10.0)

    def test_amount_spin_range(self, panel: SharpeningPanel) -> None:
        assert panel._amount_spin.minimum() == pytest.approx(0.0)
        assert panel._amount_spin.maximum() == pytest.approx(1.0)

    def test_threshold_spin_range(self, panel: SharpeningPanel) -> None:
        assert panel._threshold_spin.minimum() == pytest.approx(0.0)
        assert panel._threshold_spin.maximum() == pytest.approx(0.5)


class TestSharpeningPanelSignals:
    def test_emits_preview_on_radius_change(self, panel: SharpeningPanel, qtbot) -> None:
        signals: list[dict] = []
        panel.preview_requested.connect(signals.append)
        panel._radius_spin.setValue(2.5)
        assert len(signals) >= 1
        assert signals[-1]["radius"] == pytest.approx(2.5)

    def test_emits_preview_on_amount_change(self, panel: SharpeningPanel, qtbot) -> None:
        signals: list[dict] = []
        panel.preview_requested.connect(signals.append)
        panel._amount_spin.setValue(0.8)
        assert len(signals) >= 1
        assert signals[-1]["amount"] == pytest.approx(0.8)

    def test_emits_preview_on_threshold_change(self, panel: SharpeningPanel, qtbot) -> None:
        signals: list[dict] = []
        panel.preview_requested.connect(signals.append)
        panel._threshold_spin.setValue(0.05)
        assert len(signals) >= 1
        assert signals[-1]["threshold"] == pytest.approx(0.05)

    def test_radius_change_updates_model(self, panel: SharpeningPanel, model: PipelineModel) -> None:
        panel._radius_spin.setValue(3.0)
        assert model.sharpening_radius == pytest.approx(3.0)

    def test_amount_change_updates_model(self, panel: SharpeningPanel, model: PipelineModel) -> None:
        panel._amount_spin.setValue(0.3)
        assert model.sharpening_amount == pytest.approx(0.3)

    def test_threshold_change_updates_model(self, panel: SharpeningPanel, model: PipelineModel) -> None:
        panel._threshold_spin.setValue(0.1)
        assert model.sharpening_threshold == pytest.approx(0.1)


class TestSharpeningPanelSyncFromModel:
    def test_sync_from_model_on_config_changed(self, panel: SharpeningPanel, model: PipelineModel) -> None:
        model._sharpening_radius = 4.0
        model._sharpening_amount = 0.9
        model._sharpening_threshold = 0.15
        model.sharpening_config_changed.emit()
        assert panel._radius_spin.value() == pytest.approx(4.0)
        assert panel._amount_spin.value() == pytest.approx(0.9)
        assert panel._threshold_spin.value() == pytest.approx(0.15)

    def test_sync_on_pipeline_reset(self, panel: SharpeningPanel, model: PipelineModel) -> None:
        panel._radius_spin.setValue(5.0)
        model.pipeline_reset.emit()
        assert panel._radius_spin.value() == pytest.approx(model.sharpening_radius)


class TestSharpeningPanelPreviewContent:
    def test_preview_dict_has_all_keys(self, panel: SharpeningPanel, qtbot) -> None:
        signals: list[dict] = []
        panel.preview_requested.connect(signals.append)
        panel._amount_spin.setValue(0.4)
        assert "radius" in signals[-1]
        assert "amount" in signals[-1]
        assert "threshold" in signals[-1]
