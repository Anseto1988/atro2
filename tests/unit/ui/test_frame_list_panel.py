"""Tests for FrameListPanel widget."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QMimeData, QPoint, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent

from astroai.project.project_file import FrameEntry
from astroai.ui.widgets.frame_list_panel import FrameListPanel, _format_exposure, _score_text, _exposure_text


def _entry(name: str = "frame.fits", *, score: float | None = None, exp: float | None = 120.0, selected: bool = True, notes: str = "") -> FrameEntry:
    return FrameEntry(path=f"/tmp/{name}", exposure=exp, quality_score=score, selected=selected, notes=notes)


@pytest.fixture()
def panel(qtbot):
    w = FrameListPanel()
    qtbot.addWidget(w)
    return w


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_format_exposure_seconds(self):
        assert _format_exposure(45) == "45s"

    def test_format_exposure_minutes(self):
        assert _format_exposure(90) == "1m 30s"

    def test_format_exposure_hours(self):
        assert _format_exposure(3720) == "1h 02m"

    def test_score_text_none(self):
        assert _score_text(None) == "—"

    def test_score_text_value(self):
        assert _score_text(0.85) == "85.0%"

    def test_exposure_text_none(self):
        assert _exposure_text(None) == "—"

    def test_exposure_text_value(self):
        assert _exposure_text(120.0) == "120.0"


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_table_starts_empty(self, panel):
        assert panel._table.rowCount() == 0

    def test_accepts_drops(self, panel):
        assert panel.acceptDrops()

    def test_count_label_no_frames(self, panel):
        assert panel._count_label.text() == "Keine Frames geladen"


# ---------------------------------------------------------------------------
# Refresh / repopulate
# ---------------------------------------------------------------------------

class TestRefresh:
    def test_refresh_populates_rows(self, panel):
        panel.refresh([_entry("a.fits"), _entry("b.fits")])
        assert panel._table.rowCount() == 2

    def test_refresh_shows_filename(self, panel):
        panel.refresh([_entry("myfile.fits")])
        item = panel._table.item(0, 0)
        assert item is not None
        assert "myfile.fits" in item.text()

    def test_refresh_shows_exposure(self, panel):
        panel.refresh([_entry(exp=60.0)])
        item = panel._table.item(0, 1)
        assert item is not None
        assert "60.0" in item.text()

    def test_refresh_shows_score(self, panel):
        panel.refresh([_entry(score=0.9)])
        item = panel._table.item(0, 2)
        assert item is not None
        assert "90.0%" in item.text()

    def test_refresh_selected_shows_checkmark(self, panel):
        panel.refresh([_entry(selected=True)])
        item = panel._table.item(0, 3)
        assert item is not None
        assert item.text() == "✓"

    def test_refresh_deselected_shows_cross(self, panel):
        panel.refresh([_entry(selected=False)])
        item = panel._table.item(0, 3)
        assert item is not None
        assert item.text() == "✗"

    def test_refresh_no_score_shows_dash(self, panel):
        panel.refresh([_entry(score=None)])
        item = panel._table.item(0, 2)
        assert item is not None
        assert item.text() == "—"

    def test_refresh_updates_count_label(self, panel):
        panel.refresh([_entry("a.fits"), _entry("b.fits")])
        text = panel._count_label.text()
        assert "2" in text

    def test_refresh_clears_previous(self, panel):
        panel.refresh([_entry("a.fits"), _entry("b.fits"), _entry("c.fits")])
        panel.refresh([_entry("x.fits")])
        assert panel._table.rowCount() == 1

    def test_entry_with_notes_has_asterisk_prefix(self, panel):
        panel.refresh([_entry(notes="test note")])
        item = panel._table.item(0, 0)
        assert item is not None
        assert item.text().startswith("*")

    def test_entry_without_notes_no_asterisk(self, panel):
        panel.refresh([_entry(notes="")])
        item = panel._table.item(0, 0)
        assert item is not None
        assert not item.text().startswith("*")


# ---------------------------------------------------------------------------
# Count label
# ---------------------------------------------------------------------------

class TestCountLabel:
    def test_shows_selected_count(self, panel):
        panel.refresh([_entry(selected=True), _entry(selected=False)])
        assert "1" in panel._count_label.text()

    def test_shows_scored_count(self, panel):
        panel.refresh([_entry(score=0.8), _entry(score=None)])
        text = panel._count_label.text()
        assert "1 bewertet" in text

    def test_shows_total_exposure_for_selected(self, panel):
        panel.refresh([_entry(exp=120.0, selected=True), _entry(exp=60.0, selected=False)])
        text = panel._count_label.text()
        assert "2m" in text or "120" in text or "2m 00s" in text

    def test_no_exposure_when_none_selected(self, panel):
        panel.refresh([_entry(selected=False)])
        text = panel._count_label.text()
        assert "0 ausgewählt" in text


# ---------------------------------------------------------------------------
# Selection toggle
# ---------------------------------------------------------------------------

class TestSelectionToggle:
    def test_double_click_toggles_selected_to_false(self, panel, qtbot):
        entries = [_entry(selected=True)]
        panel.refresh(entries)
        signals = []
        panel.selection_changed.connect(lambda i, v: signals.append((i, v)))
        panel._on_cell_double_clicked(0, 0)
        assert entries[0].selected is False
        assert signals == [(0, False)]

    def test_double_click_toggles_selected_to_true(self, panel, qtbot):
        entries = [_entry(selected=False)]
        panel.refresh(entries)
        signals = []
        panel.selection_changed.connect(lambda i, v: signals.append((i, v)))
        panel._on_cell_double_clicked(0, 0)
        assert entries[0].selected is True
        assert signals == [(0, True)]

    def test_double_click_updates_cell_text(self, panel):
        panel.refresh([_entry(selected=True)])
        panel._on_cell_double_clicked(0, 0)
        item = panel._table.item(0, 3)
        assert item is not None
        assert item.text() == "✗"

    def test_double_click_out_of_bounds_no_crash(self, panel):
        panel.refresh([_entry()])
        panel._on_cell_double_clicked(99, 0)  # should not raise


# ---------------------------------------------------------------------------
# Select all / deselect all / invert
# ---------------------------------------------------------------------------

class TestBulkSelection:
    def test_select_all(self, panel, qtbot):
        entries = [_entry(selected=False), _entry(selected=False)]
        panel.refresh(entries)
        panel.select_all()
        assert all(e.selected for e in entries)

    def test_deselect_all(self, panel, qtbot):
        entries = [_entry(selected=True), _entry(selected=True)]
        panel.refresh(entries)
        panel.deselect_all()
        assert not any(e.selected for e in entries)

    def test_invert_selection(self, panel, qtbot):
        entries = [_entry(selected=True), _entry(selected=False)]
        panel.refresh(entries)
        panel.invert_selection()
        assert entries[0].selected is False
        assert entries[1].selected is True

    def test_select_all_emits_signals(self, panel, qtbot):
        signals = []
        panel.refresh([_entry(selected=False), _entry(selected=False)])
        panel.selection_changed.connect(lambda i, v: signals.append((i, v)))
        panel.select_all()
        assert len(signals) == 2

    def test_select_all_same_value_no_signal(self, panel, qtbot):
        panel.refresh([_entry(selected=True)])
        signals = []
        panel.selection_changed.connect(lambda i, v: signals.append((i, v)))
        panel.select_all()
        assert signals == []

    def test_deselect_all_updates_cell_text(self, panel):
        panel.refresh([_entry(selected=True)])
        panel.deselect_all()
        item = panel._table.item(0, 3)
        assert item is not None
        assert item.text() == "✗"


# ---------------------------------------------------------------------------
# Quality threshold
# ---------------------------------------------------------------------------

class TestQualityThreshold:
    def test_threshold_deselects_below_score(self, panel):
        entries = [_entry(score=0.3, selected=True), _entry(score=0.8, selected=True)]
        panel.refresh(entries)
        panel.apply_quality_threshold(50.0)
        assert entries[0].selected is False
        assert entries[1].selected is True

    def test_threshold_ignores_unscored(self, panel):
        entries = [_entry(score=None, selected=True)]
        panel.refresh(entries)
        panel.apply_quality_threshold(80.0)
        assert entries[0].selected is True

    def test_threshold_emits_signal_on_change(self, panel):
        signals = []
        entries = [_entry(score=0.2, selected=True)]
        panel.refresh(entries)
        panel.selection_changed.connect(lambda i, v: signals.append((i, v)))
        panel.apply_quality_threshold(50.0)
        assert len(signals) == 1
        assert signals[0] == (0, False)

    def test_threshold_no_signal_if_unchanged(self, panel):
        signals = []
        entries = [_entry(score=0.9, selected=True)]
        panel.refresh(entries)
        panel.selection_changed.connect(lambda i, v: signals.append((i, v)))
        panel.apply_quality_threshold(50.0)
        assert signals == []

    def test_threshold_applies_via_button(self, panel, qtbot):
        entries = [_entry(score=0.1, selected=True)]
        panel.refresh(entries)
        panel._quality_spinbox.setValue(80.0)
        panel._on_apply_threshold()
        assert entries[0].selected is False


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class TestFilter:
    def test_filter_hides_non_matching_rows(self, panel):
        panel.refresh([_entry("alpha.fits"), _entry("beta.fits")])
        panel._on_filter_changed("alpha")
        assert panel._table.isRowHidden(0) is False
        assert panel._table.isRowHidden(1) is True

    def test_filter_case_insensitive(self, panel):
        panel.refresh([_entry("MyFrame.fits")])
        panel._on_filter_changed("myframe")
        assert panel._table.isRowHidden(0) is False

    def test_empty_filter_shows_all_rows(self, panel):
        panel.refresh([_entry("a.fits"), _entry("b.fits")])
        panel._on_filter_changed("x")
        panel._on_filter_changed("")
        assert panel._table.isRowHidden(0) is False
        assert panel._table.isRowHidden(1) is False


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

class TestSort:
    def test_sort_by_filename_asc(self, panel):
        panel.refresh([_entry("z.fits"), _entry("a.fits")])
        panel._on_header_clicked(0)
        assert panel._entries[0].path.endswith("a.fits")

    def test_sort_by_filename_desc(self, panel):
        panel.refresh([_entry("a.fits"), _entry("z.fits")])
        panel._on_header_clicked(0)
        panel._on_header_clicked(0)
        assert panel._entries[0].path.endswith("z.fits")

    def test_sort_by_score(self, panel):
        panel.refresh([_entry(score=0.9), _entry(score=0.1)])
        panel._on_header_clicked(2)
        assert panel._entries[0].quality_score == pytest.approx(0.1)

    def test_sort_by_selection(self, panel):
        panel.refresh([_entry(selected=True), _entry(selected=False)])
        panel._on_header_clicked(3)
        assert panel._entries[0].selected is False

    def test_sort_empty_entries_no_crash(self, panel):
        panel._on_header_clicked(0)

    def test_sort_persists_after_refresh(self, panel):
        panel.refresh([_entry("b.fits"), _entry("a.fits")])
        panel._on_header_clicked(0)  # sort asc — a.fits first
        panel.refresh([_entry("z.fits"), _entry("a.fits")])
        assert panel._entries[0].path.endswith("a.fits")

    def test_sort_by_exposure(self, panel):
        panel.refresh([_entry(exp=300.0), _entry(exp=60.0)])
        panel._on_header_clicked(1)
        assert panel._entries[0].exposure == pytest.approx(60.0)

    def test_sort_exposure_none_treated_as_negative(self, panel):
        panel.refresh([_entry(exp=None), _entry(exp=120.0)])
        panel._on_header_clicked(1)
        assert panel._entries[0].exposure is None


# ---------------------------------------------------------------------------
# Notes editing
# ---------------------------------------------------------------------------

class TestNotes:
    def test_edit_notes_out_of_bounds_no_crash(self, panel):
        panel.refresh([_entry()])
        panel._edit_notes(99)

    def test_edit_notes_empty_entries_no_crash(self, panel):
        panel._edit_notes(0)


# ---------------------------------------------------------------------------
# Signals: files_dropped, preview_requested, remove_requested
# ---------------------------------------------------------------------------

class TestContextMenuSignals:
    def test_remove_requested_emits(self, panel, qtbot):
        panel.refresh([_entry("a.fits"), _entry("b.fits")])
        signals = []
        panel.remove_requested.connect(signals.append)
        panel.remove_requested.emit([0])
        assert signals == [[0]]

    def test_preview_requested_emits(self, panel, qtbot):
        panel.refresh([_entry("a.fits")])
        signals = []
        panel.preview_requested.connect(signals.append)
        panel.preview_requested.emit("/tmp/a.fits")
        assert signals == ["/tmp/a.fits"]


class TestContextMenu:
    """Test _show_context_menu by monkeypatching _exec_menu."""

    def test_no_entries_returns_early(self, panel):
        panel._show_context_menu(QPoint(0, 0))

    def test_select_all_action(self, panel, monkeypatch):
        entries = [_entry(selected=False)]
        panel.refresh(entries)
        monkeypatch.setattr(panel, "_exec_menu", lambda menu, pos: next(
            a for a in menu.actions() if a.text() == "Alle auswählen"
        ))
        panel._show_context_menu(QPoint(0, 0))
        assert entries[0].selected is True

    def test_deselect_all_action(self, panel, monkeypatch):
        entries = [_entry(selected=True)]
        panel.refresh(entries)
        monkeypatch.setattr(panel, "_exec_menu", lambda menu, pos: next(
            a for a in menu.actions() if a.text() == "Alle abwählen"
        ))
        panel._show_context_menu(QPoint(0, 0))
        assert entries[0].selected is False

    def test_invert_action(self, panel, monkeypatch):
        entries = [_entry(selected=True)]
        panel.refresh(entries)
        monkeypatch.setattr(panel, "_exec_menu", lambda menu, pos: next(
            a for a in menu.actions() if a.text() == "Auswahl umkehren"
        ))
        panel._show_context_menu(QPoint(0, 0))
        assert entries[0].selected is False

    def test_none_action_no_crash(self, panel, monkeypatch):
        panel.refresh([_entry()])
        monkeypatch.setattr(panel, "_exec_menu", lambda menu, pos: None)
        panel._show_context_menu(QPoint(0, 0))

    def test_remove_action_emits_signal(self, panel, monkeypatch, qtbot):
        panel.refresh([_entry("a.fits"), _entry("b.fits")])
        panel._table.selectRow(0)
        signals = []
        panel.remove_requested.connect(signals.append)

        def fake_exec(menu, pos):
            for a in menu.actions():
                if "entfernen" in a.text().lower():
                    return a
            return None

        monkeypatch.setattr(panel, "_exec_menu", fake_exec)
        panel._show_context_menu(QPoint(0, 0))
        assert len(signals) == 1

    def test_notes_action_opens_edit(self, panel, monkeypatch, qtbot):
        entries = [_entry("frame.fits", notes="")]
        panel.refresh(entries)
        panel._table.selectRow(0)
        monkeypatch.setattr(
            "astroai.ui.widgets.frame_list_panel.QInputDialog.getText",
            lambda *a, **kw: ("updated", True),
        )
        monkeypatch.setattr(panel, "_exec_menu", lambda menu, pos: next(
            a for a in menu.actions() if a.text() == "Notiz bearbeiten…"
        ))
        panel._show_context_menu(QPoint(0, 0))
        assert entries[0].notes == "updated"

    def test_preview_action_with_one_row_selected(self, panel, monkeypatch, qtbot):
        panel.refresh([_entry("preview.fits")])
        panel._table.selectRow(0)
        previews = []
        panel.preview_requested.connect(previews.append)

        monkeypatch.setattr(panel, "_exec_menu", lambda menu, pos: next(
            a for a in menu.actions() if a.text() == "Vorschau anzeigen"
        ))
        panel._show_context_menu(QPoint(0, 0))
        assert len(previews) == 1
        assert previews[0].endswith("preview.fits")


class TestEditNotes:
    def test_edit_notes_ok_updates_entry(self, panel, monkeypatch):
        entries = [_entry(notes="old")]
        panel.refresh(entries)
        monkeypatch.setattr(
            "astroai.ui.widgets.frame_list_panel.QInputDialog.getText",
            lambda *a, **kw: ("new note", True),
        )
        panel._edit_notes(0)
        assert entries[0].notes == "new note"

    def test_edit_notes_cancel_keeps_old(self, panel, monkeypatch):
        entries = [_entry(notes="keep")]
        panel.refresh(entries)
        monkeypatch.setattr(
            "astroai.ui.widgets.frame_list_panel.QInputDialog.getText",
            lambda *a, **kw: ("ignored", False),
        )
        panel._edit_notes(0)
        assert entries[0].notes == "keep"


class TestSortKeyFnFallback:
    def test_unknown_col_returns_zero(self, panel):
        panel.refresh([_entry()])
        panel._sort_col = 99
        key_fn = panel._sort_key_fn()
        assert key_fn(panel._entries[0]) == 0


class TestCountLabelReset:
    def test_count_label_resets_to_no_frames_when_empty(self, panel):
        panel.refresh([_entry()])
        panel.refresh([])
        assert panel._count_label.text() == "Keine Frames geladen"


class TestDragAndDrop:
    def _make_mime(self, paths: list[str]) -> QMimeData:
        mime = QMimeData()
        urls = [QUrl.fromLocalFile(p) for p in paths]
        mime.setUrls(urls)
        return mime

    def test_drag_enter_accepts_fits(self, panel):
        mime = self._make_mime(["/tmp/frame.fits"])
        event = MagicMock()
        event.mimeData.return_value = mime
        panel.dragEnterEvent(event)
        event.acceptProposedAction.assert_called_once()

    def test_drag_enter_ignores_non_fits(self, panel):
        mime = self._make_mime(["/tmp/frame.jpg"])
        event = MagicMock()
        event.mimeData.return_value = mime
        panel.dragEnterEvent(event)
        event.ignore.assert_called_once()

    def test_drag_enter_ignores_no_urls(self, panel):
        mime = QMimeData()
        event = MagicMock()
        event.mimeData.return_value = mime
        panel.dragEnterEvent(event)
        event.ignore.assert_called_once()

    def test_drag_move_accepts_with_urls(self, panel):
        mime = self._make_mime(["/tmp/frame.fits"])
        event = MagicMock()
        event.mimeData.return_value = mime
        panel.dragMoveEvent(event)
        event.acceptProposedAction.assert_called_once()

    def test_drag_move_ignores_without_urls(self, panel):
        mime = QMimeData()
        event = MagicMock()
        event.mimeData.return_value = mime
        panel.dragMoveEvent(event)
        event.ignore.assert_called_once()

    def test_drop_event_emits_fits_paths(self, panel, qtbot):
        mime = self._make_mime(["/tmp/frame.fits", "/tmp/other.fit"])
        event = MagicMock()
        event.mimeData.return_value = mime
        dropped = []
        panel.files_dropped.connect(dropped.append)
        panel.dropEvent(event)
        assert len(dropped) == 1
        assert len(dropped[0]) == 2
        event.acceptProposedAction.assert_called_once()

    def test_drop_event_ignores_non_fits(self, panel):
        mime = self._make_mime(["/tmp/image.png"])
        event = MagicMock()
        event.mimeData.return_value = mime
        panel.dropEvent(event)
        event.ignore.assert_called_once()
