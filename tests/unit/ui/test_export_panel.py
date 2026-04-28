"""Tests for ExportPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.export_panel import ExportPanel, _FORMAT_VALUES


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = ExportPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_dir_edit_matches_model(self, model, panel):
        assert panel._dir_edit.text() == model.output_path

    def test_name_edit_matches_model(self, model, panel):
        assert panel._name_edit.text() == model.output_filename

    def test_fmt_combo_has_three_items(self, panel):
        assert panel._fmt_combo.count() == 3

    def test_fmt_combo_matches_model(self, model, panel):
        idx = panel._fmt_combo.currentIndex()
        assert _FORMAT_VALUES[idx] == model.output_format


class TestUIToModel:
    def test_on_dir_edited_updates_model(self, model, panel):
        panel._dir_edit.setText("/output/dir")
        panel._on_dir_edited()
        assert model.output_path == "/output/dir"

    def test_on_name_edited_updates_model(self, model, panel):
        panel._name_edit.setText("stacked")
        panel._on_name_edited()
        assert model.output_filename == "stacked"

    def test_on_name_edited_empty_falls_back_to_output(self, model, panel):
        panel._name_edit.setText("  ")
        panel._on_name_edited()
        assert model.output_filename == "output"

    def test_on_format_changed_updates_model(self, model, panel):
        panel._on_format_changed(1)
        assert model.output_format == _FORMAT_VALUES[1]

    def test_on_format_changed_out_of_range_no_crash(self, model, panel):
        original = model.output_format
        panel._on_format_changed(99)
        assert model.output_format == original


class TestBrowse:
    def test_browse_updates_model_when_path_given(self, model, panel, monkeypatch):
        monkeypatch.setattr(
            "astroai.ui.widgets.export_panel.QFileDialog.getExistingDirectory",
            lambda *a, **kw: "/chosen/path",
        )
        panel._on_browse()
        assert model.output_path == "/chosen/path"

    def test_browse_cancelled_leaves_model_unchanged(self, model, panel, monkeypatch):
        original = model.output_path
        monkeypatch.setattr(
            "astroai.ui.widgets.export_panel.QFileDialog.getExistingDirectory",
            lambda *a, **kw: "",
        )
        panel._on_browse()
        assert model.output_path == original


class TestModelToUI:
    def test_model_path_updates_dir_edit(self, model, panel):
        model.output_path = "/new/path"
        assert panel._dir_edit.text() == "/new/path"

    def test_model_filename_updates_name_edit(self, model, panel):
        model.output_filename = "result"
        assert panel._name_edit.text() == "result"

    def test_model_format_updates_combo(self, model, panel):
        model.output_format = "xisf"
        idx = _FORMAT_VALUES.index("xisf")
        assert panel._fmt_combo.currentIndex() == idx

    def test_model_unknown_format_falls_back_to_index_zero(self, model, panel):
        model._output_format = "unknown"
        panel._sync_from_model()
        assert panel._fmt_combo.currentIndex() == 0

    def test_pipeline_reset_syncs_panel(self, model, panel):
        model.output_path = "/some/path"
        model.reset()
        assert panel._dir_edit.text() == model.output_path
