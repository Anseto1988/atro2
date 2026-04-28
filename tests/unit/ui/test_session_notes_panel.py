"""Tests for SessionNotesPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.widgets.session_notes_panel import SessionNotesPanel


@pytest.fixture()
def panel(qtbot):
    w = SessionNotesPanel()
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_notes_empty_by_default(self, panel):
        assert panel.notes == ""

    def test_editor_not_rich_text(self, panel):
        assert not panel._editor.acceptRichText()


class TestSetNotes:
    def test_set_notes_updates_content(self, panel):
        panel.set_notes("Hello World")
        assert panel.notes == "Hello World"

    def test_set_notes_does_not_emit_signal(self, panel, qtbot):
        with qtbot.assertNotEmitted(panel.text_changed):
            panel.set_notes("Silent update")

    def test_set_notes_empty_string(self, panel):
        panel.set_notes("something")
        panel.set_notes("")
        assert panel.notes == ""


class TestTextChanged:
    def test_typing_emits_text_changed(self, panel, qtbot):
        signals = []
        panel.text_changed.connect(signals.append)
        panel._editor.setPlainText("new text")
        assert len(signals) >= 1
        assert signals[-1] == "new text"

    def test_signal_carries_current_text(self, panel, qtbot):
        received = []
        panel.text_changed.connect(received.append)
        panel._editor.setPlainText("note content")
        assert "note content" in received
