"""Tests for CLAHEPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.clahe_panel import CLAHEPanel


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = CLAHEPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_clip_spin_matches_model_default(self, model, panel):
        assert panel._spin_clip.value() == pytest.approx(model.clahe_clip_limit)

    def test_tile_spin_matches_model_default(self, model, panel):
        assert panel._spin_tile.value() == model.clahe_tile_size

    def test_combo_mode_matches_model_default(self, model, panel):
        # "luminance" is the default, index 0
        assert panel._combo_mode.currentIndex() == 0

    def test_reset_button_exists(self, panel):
        assert panel._reset_btn is not None

    def test_group_box_title(self, panel):
        # Find the QGroupBox child
        from PySide6.QtWidgets import QGroupBox
        groups = panel.findChildren(QGroupBox)
        assert any("CLAHE" in g.title() for g in groups)


class TestUIToModel:
    def test_clip_spin_updates_model(self, model, panel):
        panel._spin_clip.setValue(4.5)
        assert model.clahe_clip_limit == pytest.approx(4.5, abs=0.05)

    def test_tile_spin_updates_model(self, model, panel):
        panel._spin_tile.setValue(32)
        assert model.clahe_tile_size == 32

    def test_combo_mode_each_updates_model(self, model, panel):
        # Index 1 = "each"
        panel._combo_mode.setCurrentIndex(1)
        assert model.clahe_channel_mode == "each"

    def test_reset_button_resets_model(self, model, panel):
        model.clahe_clip_limit = 7.0
        model.clahe_tile_size = 16
        model.clahe_channel_mode = "each"
        panel._reset_btn.click()
        assert model.clahe_clip_limit == pytest.approx(2.0)
        assert model.clahe_tile_size == 64
        assert model.clahe_channel_mode == "luminance"


class TestModelToUI:
    def test_model_clip_updates_spin(self, model, panel):
        model.clahe_clip_limit = 6.0
        assert panel._spin_clip.value() == pytest.approx(6.0, abs=0.05)

    def test_model_tile_updates_spin(self, model, panel):
        model.clahe_tile_size = 128
        assert panel._spin_tile.value() == 128

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.clahe_clip_limit = 5.0
        model.reset()
        assert panel._spin_clip.value() == pytest.approx(model.clahe_clip_limit, abs=0.05)


class TestSignals:
    def test_clip_change_emits_clahe_changed(self, panel, qtbot):
        with qtbot.waitSignal(panel.clahe_changed, timeout=500):
            panel._spin_clip.setValue(3.5)

    def test_tile_change_emits_clahe_changed(self, panel, qtbot):
        with qtbot.waitSignal(panel.clahe_changed, timeout=500):
            panel._spin_tile.setValue(32)

    def test_emitted_config_has_correct_clip(self, model, panel, qtbot):
        received = []
        panel.clahe_changed.connect(received.append)
        panel._spin_clip.setValue(4.0)
        assert len(received) >= 1
        assert received[-1].clip_limit == pytest.approx(4.0, abs=0.05)

    def test_emitted_config_has_correct_tile(self, model, panel, qtbot):
        received = []
        panel.clahe_changed.connect(received.append)
        panel._spin_tile.setValue(32)
        assert len(received) >= 1
        assert received[-1].tile_size == 32
