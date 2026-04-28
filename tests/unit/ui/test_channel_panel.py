"""Tests for ChannelCombinerPanel widget."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.ui.models import PipelineModel
from astroai.ui.widgets.channel_panel import (
    ChannelCombinerPanel,
    _LRGB_CHANNELS,
    _NB_CHANNELS,
    _PALETTE_LABELS,
)


@pytest.fixture()
def model():
    return PipelineModel()


@pytest.fixture()
def panel(model, qtbot):
    w = ChannelCombinerPanel(model)
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_lrgb_radio_checked_initially(self, panel):
        assert panel._rb_lrgb.isChecked()

    def test_lrgb_group_visible_initially(self, panel):
        assert not panel._lrgb_group.isHidden()

    def test_nb_group_hidden_initially(self, panel):
        assert panel._nb_group.isHidden()

    def test_palette_widget_hidden_initially(self, panel):
        assert panel._palette_widget.isHidden()

    def test_palette_combo_has_correct_items(self, panel):
        assert panel._palette_combo.count() == len(_PALETTE_LABELS)

    def test_all_lrgb_edits_present(self, panel):
        for ch in _LRGB_CHANNELS:
            assert ch in panel._channel_edits

    def test_all_nb_edits_present(self, panel):
        for ch in _NB_CHANNELS:
            assert ch in panel._channel_edits

    def test_status_label_empty_initially(self, panel):
        assert panel._status_label.text() == ""


class TestModeToggle:
    def test_switch_to_narrowband_shows_nb_group(self, panel):
        panel._rb_nb.setChecked(True)
        assert not panel._nb_group.isHidden()

    def test_switch_to_narrowband_hides_lrgb_group(self, panel):
        panel._rb_nb.setChecked(True)
        assert panel._lrgb_group.isHidden()

    def test_switch_to_narrowband_shows_palette_widget(self, panel):
        panel._rb_nb.setChecked(True)
        assert not panel._palette_widget.isHidden()

    def test_switch_back_to_lrgb_hides_nb_group(self, panel):
        panel._rb_nb.setChecked(True)
        panel._rb_lrgb.setChecked(True)
        assert panel._nb_group.isHidden()

    def test_switch_back_to_lrgb_shows_lrgb_group(self, panel):
        panel._rb_nb.setChecked(True)
        panel._rb_lrgb.setChecked(True)
        assert not panel._lrgb_group.isHidden()

    def test_switch_back_to_lrgb_hides_palette_widget(self, panel):
        panel._rb_nb.setChecked(True)
        panel._rb_lrgb.setChecked(True)
        assert panel._palette_widget.isHidden()


class TestResetFields:
    def test_reset_clears_all_edits(self, panel):
        for edit in panel._channel_edits.values():
            edit.setText("/some/path.fits")
        panel._reset_fields()
        for edit in panel._channel_edits.values():
            assert edit.text() == ""

    def test_reset_clears_status_label(self, panel):
        panel._status_label.setText("OK — Shape: (100, 100)")
        panel._reset_fields()
        assert panel._status_label.text() == ""

    def test_pipeline_reset_triggers_reset_fields(self, model, panel):
        for edit in panel._channel_edits.values():
            edit.setText("/tmp/frame.fits")
        model.reset()
        for edit in panel._channel_edits.values():
            assert edit.text() == ""


class TestCombineNoChannels:
    def test_combine_with_no_files_sets_status(self, panel):
        panel._on_combine()
        assert panel._status_label.text() == "Keine Kanäle geladen."

    def test_combine_narrowband_with_no_files_sets_status(self, panel):
        panel._rb_nb.setChecked(True)
        panel._on_combine()
        assert panel._status_label.text() == "Keine Kanäle geladen."


class TestCombineReadError:
    def test_combine_read_error_sets_lesefehler_status(self, panel, monkeypatch):
        import astroai.core.io as io_mod
        panel._channel_edits["L"].setText("/fake/L.fits")

        def bad_read(p):
            raise OSError("file missing")

        monkeypatch.setattr(io_mod, "read_fits", bad_read)
        panel._on_combine()
        assert "Lesefehler" in panel._status_label.text()


class TestCombineLRGB:
    def test_combine_lrgb_ok_sets_status(self, panel, monkeypatch):
        import astroai.core.io as io_mod
        import astroai.processing.channels as ch_mod

        fake_arr = np.zeros((4, 4, 3), dtype=np.float32)

        def fake_read_fits(p):
            return np.zeros((4, 4), dtype=np.float32), {}

        class FakeCombiner:
            def combine_lrgb(self, L=None, R=None, G=None, B=None):
                return fake_arr

        monkeypatch.setattr(io_mod, "read_fits", fake_read_fits)
        monkeypatch.setattr(ch_mod, "ChannelCombiner", FakeCombiner)

        panel._channel_edits["L"].setText("/tmp/L.fits")
        panel._on_combine()
        assert "OK" in panel._status_label.text()

    def test_combine_lrgb_error_sets_fehler_status(self, panel, monkeypatch):
        import astroai.core.io as io_mod
        import astroai.processing.channels as ch_mod

        def fake_read_fits(p):
            return np.zeros((4, 4), dtype=np.float32), {}

        class BadCombiner:
            def combine_lrgb(self, **kw):
                raise ValueError("bad combine")

        monkeypatch.setattr(io_mod, "read_fits", fake_read_fits)
        monkeypatch.setattr(ch_mod, "ChannelCombiner", BadCombiner)

        panel._channel_edits["L"].setText("/tmp/L.fits")
        panel._on_combine()
        assert "Fehler" in panel._status_label.text()

    def test_combine_tiff_read_path(self, panel, monkeypatch):
        import astroai.core.io as io_mod
        import astroai.processing.channels as ch_mod

        def fake_read_tiff(p):
            return np.zeros((4, 4), dtype=np.float32), {}

        class FakeCombiner:
            def combine_lrgb(self, **kw):
                return np.zeros((4, 4, 3), dtype=np.float32)

        monkeypatch.setattr(io_mod, "read_tiff", fake_read_tiff)
        monkeypatch.setattr(ch_mod, "ChannelCombiner", FakeCombiner)

        panel._channel_edits["L"].setText("/tmp/L.tif")
        panel._on_combine()
        assert "OK" in panel._status_label.text()


class TestCombineNarrowband:
    def test_combine_nb_ok_sets_status(self, panel, monkeypatch):
        import astroai.core.io as io_mod
        import astroai.processing.channels as ch_mod

        fake_result = np.zeros((4, 4, 3), dtype=np.float32)

        def fake_read_fits(p):
            return np.zeros((4, 4), dtype=np.float32), {}

        class FakeMapper:
            def map(self, Ha=None, OIII=None, SII=None, palette=None):
                return fake_result

        monkeypatch.setattr(io_mod, "read_fits", fake_read_fits)
        monkeypatch.setattr(ch_mod, "NarrowbandMapper", FakeMapper)

        panel._rb_nb.setChecked(True)
        panel._channel_edits["Ha"].setText("/tmp/Ha.fits")
        panel._on_combine()
        assert "OK" in panel._status_label.text()


class TestPickFile:
    def test_pick_file_sets_edit_text(self, panel, monkeypatch):
        monkeypatch.setattr(
            "astroai.ui.widgets.channel_panel.QFileDialog.getOpenFileName",
            lambda *a, **kw: ("/chosen/L.fits", "FITS (*.fits)"),
        )
        edit = panel._channel_edits["L"]
        panel._pick_file(edit, "L")
        assert edit.text() == "/chosen/L.fits"

    def test_pick_file_cancelled_leaves_edit_unchanged(self, panel, monkeypatch):
        edit = panel._channel_edits["R"]
        edit.setText("/original/R.fits")
        monkeypatch.setattr(
            "astroai.ui.widgets.channel_panel.QFileDialog.getOpenFileName",
            lambda *a, **kw: ("", ""),
        )
        panel._pick_file(edit, "R")
        assert edit.text() == "/original/R.fits"


class TestModelResultReady:
    def test_model_result_ready_default_no_crash(self, panel):
        panel.model_result_ready(np.zeros((4, 4, 3), dtype=np.float32))
