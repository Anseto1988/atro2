"""Tests for MosaicPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.mosaic_panel import MosaicPanel, _BLEND_MODES


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = MosaicPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_enabled_cb_matches_model(self, model, panel):
        assert panel._enabled_cb.isChecked() == model.mosaic_enabled

    def test_settings_group_disabled_when_not_enabled(self, panel):
        assert not panel._settings_group.isEnabled()

    def test_blend_combo_has_all_modes(self, panel):
        assert panel._blend_combo.count() == len(_BLEND_MODES)

    def test_scale_slider_matches_model(self, model, panel):
        expected = int(model.mosaic_output_scale * 100)
        assert panel._scale_slider.value() == expected

    def test_scale_label_matches_model(self, model, panel):
        expected = f"{model.mosaic_output_scale:.2f}x"
        assert panel._scale_value.text() == expected

    def test_panel_list_empty_initially(self, panel):
        assert panel._panel_list.count() == 0


class TestUIToModel:
    def test_enable_checkbox_updates_model(self, model, panel):
        panel._enabled_cb.setChecked(True)
        assert model.mosaic_enabled is True

    def test_enable_enables_controls(self, panel):
        panel._enabled_cb.setChecked(True)
        assert panel._settings_group.isEnabled()
        assert panel._add_btn.isEnabled()
        assert panel._remove_btn.isEnabled()

    def test_disable_disables_controls(self, panel):
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert not panel._settings_group.isEnabled()
        assert not panel._add_btn.isEnabled()

    def test_blend_combo_updates_model(self, model, panel):
        panel._blend_combo.setCurrentText("median")
        assert model.mosaic_blend_mode == "median"

    def test_gradient_cb_updates_model(self, model, panel):
        panel._on_gradient_changed(True)
        assert model.mosaic_gradient_correct is True

    def test_scale_slider_updates_model(self, model, panel):
        panel._scale_slider.setValue(200)
        assert abs(model.mosaic_output_scale - 2.0) < 0.01

    def test_scale_slider_updates_label(self, panel):
        panel._scale_slider.setValue(150)
        assert panel._scale_value.text() == "1.50x"


class TestAddRemovePanels:
    def test_add_panels_via_dialog(self, model, panel, monkeypatch):
        monkeypatch.setattr(
            "astroai.ui.widgets.mosaic_panel.QFileDialog.getOpenFileNames",
            lambda *a, **kw: (["/tmp/panel1.fits", "/tmp/panel2.fits"], ""),
        )
        panel._on_add_panels()
        assert "/tmp/panel1.fits" in model.mosaic_panels
        assert "/tmp/panel2.fits" in model.mosaic_panels

    def test_add_panels_cancelled_no_change(self, model, panel, monkeypatch):
        monkeypatch.setattr(
            "astroai.ui.widgets.mosaic_panel.QFileDialog.getOpenFileNames",
            lambda *a, **kw: ([], ""),
        )
        panel._on_add_panels()
        assert len(model.mosaic_panels) == 0

    def test_remove_panel_removes_from_model(self, model, panel, monkeypatch):
        model.add_mosaic_panel("/tmp/panel.fits")
        panel._panel_list.setCurrentRow(0)
        panel._on_remove_panel()
        assert "/tmp/panel.fits" not in model.mosaic_panels

    def test_remove_panel_no_selection_no_crash(self, panel):
        panel._on_remove_panel()


class TestModelToUI:
    def test_model_enabled_syncs_checkbox(self, model, panel):
        model.mosaic_enabled = True
        assert panel._enabled_cb.isChecked()

    def test_model_blend_mode_syncs_combo(self, model, panel):
        model.mosaic_blend_mode = "max"
        assert panel._blend_combo.currentText() == "max"

    def test_model_gradient_syncs_cb(self, model, panel):
        model.mosaic_gradient_correct = True
        assert panel._gradient_cb.isChecked()

    def test_model_scale_syncs_slider(self, model, panel):
        model.mosaic_output_scale = 2.0
        assert panel._scale_slider.value() == 200

    def test_model_panels_populate_list(self, model, panel):
        model.add_mosaic_panel("/tmp/p1.fits")
        model.add_mosaic_panel("/tmp/p2.fits")
        assert panel._panel_list.count() == 2
        assert panel._panel_list.item(0).text() == "/tmp/p1.fits"

    def test_pipeline_reset_syncs_list_to_model(self, model, panel):
        model.add_mosaic_panel("/tmp/p.fits")
        model.reset()
        assert panel._panel_list.count() == len(model.mosaic_panels)
