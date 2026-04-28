"""Tests for FITSMetadataPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.widgets.fits_metadata_panel import FITSMetadataPanel, _LABEL_MAP


@pytest.fixture()
def panel(qtbot):
    w = FITSMetadataPanel()
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_placeholder_not_hidden_initially(self, panel):
        assert not panel._placeholder.isHidden()

    def test_table_hidden_initially(self, panel):
        assert panel._table.isHidden()

    def test_table_has_two_columns(self, panel):
        assert panel._table.columnCount() == 2

    def test_table_headers(self, panel):
        assert panel._table.horizontalHeaderItem(0).text() == "Eigenschaft"
        assert panel._table.horizontalHeaderItem(1).text() == "Wert"


class TestSetHeaderClear:
    def test_set_header_none_shows_placeholder(self, panel):
        panel.set_header(None)
        assert not panel._placeholder.isHidden()
        assert panel._table.isHidden()

    def test_set_header_empty_dict_shows_placeholder(self, panel):
        panel.set_header({})
        assert not panel._placeholder.isHidden()
        assert panel._table.isHidden()

    def test_set_header_non_dict_shows_placeholder(self, panel):
        panel.set_header("invalid")
        assert not panel._placeholder.isHidden()
        assert panel._table.isHidden()

    def test_set_header_unknown_keys_shows_placeholder(self, panel):
        panel.set_header({"UNKNOWN_KEY": "value", "ALSO_UNKNOWN": "123"})
        assert not panel._placeholder.isHidden()
        assert panel._table.isHidden()


class TestSetHeaderWithData:
    def test_known_key_shows_table(self, panel):
        panel.set_header({"OBJECT": "M42"})
        assert not panel._table.isHidden()
        assert panel._placeholder.isHidden()

    def test_one_known_key_one_row(self, panel):
        panel.set_header({"OBJECT": "NGC 1499"})
        assert panel._table.rowCount() == 1

    def test_two_known_keys_two_rows(self, panel):
        panel.set_header({"OBJECT": "M31", "NAXIS1": 4096})
        assert panel._table.rowCount() == 2

    def test_row_content_label_matches_label_map(self, panel):
        panel.set_header({"OBJECT": "M42"})
        assert panel._table.item(0, 0).text() == _LABEL_MAP["OBJECT"]

    def test_row_content_value_matches_header(self, panel):
        panel.set_header({"OBJECT": "M42"})
        assert panel._table.item(0, 1).text() == "M42"

    def test_numeric_value_converted_to_str(self, panel):
        panel.set_header({"NAXIS1": 1024})
        assert panel._table.item(0, 1).text() == "1024"

    def test_mixed_known_unknown_uses_only_known(self, panel):
        panel.set_header({"OBJECT": "Orion", "UNKNOWN": "skip"})
        assert panel._table.rowCount() == 1

    def test_order_follows_label_map_order(self, panel):
        header = {"NAXIS1": 100, "OBJECT": "M42"}
        panel.set_header(header)
        labels = [panel._table.item(r, 0).text() for r in range(panel._table.rowCount())]
        expected_order = [_LABEL_MAP[k] for k in _LABEL_MAP if k in header]
        assert labels == expected_order

    def test_second_call_replaces_rows(self, panel):
        panel.set_header({"OBJECT": "M42", "NAXIS1": 100})
        panel.set_header({"EXPTIME": 300.0})
        assert panel._table.rowCount() == 1
        assert panel._table.item(0, 0).text() == _LABEL_MAP["EXPTIME"]


class TestClear:
    def test_clear_hides_table(self, panel):
        panel.set_header({"OBJECT": "M42"})
        panel.clear()
        assert panel._table.isHidden()

    def test_clear_shows_placeholder(self, panel):
        panel.set_header({"OBJECT": "M42"})
        panel.clear()
        assert not panel._placeholder.isHidden()

    def test_clear_resets_row_count(self, panel):
        panel.set_header({"OBJECT": "M42", "NAXIS1": 100})
        panel.clear()
        assert panel._table.rowCount() == 0

    def test_clear_on_empty_panel_no_crash(self, panel):
        panel.clear()
        assert not panel._placeholder.isHidden()
