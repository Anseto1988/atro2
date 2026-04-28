"""Tests for ShortcutsDialog."""
from __future__ import annotations

import pytest

from astroai.ui.widgets.shortcuts_dialog import ShortcutsDialog, _SECTIONS


class TestShortcutsDialog:
    @pytest.fixture()
    def dialog(self, qtbot):  # type: ignore[no-untyped-def]
        d = ShortcutsDialog()
        qtbot.addWidget(d)
        return d

    def test_dialog_creates(self, dialog: ShortcutsDialog) -> None:
        assert dialog is not None

    def test_window_title(self, dialog: ShortcutsDialog) -> None:
        assert "Tastaturkürzel" in dialog.windowTitle()

    def test_has_sections(self, dialog: ShortcutsDialog) -> None:
        assert len(_SECTIONS) >= 4

    def test_all_sections_have_entries(self) -> None:
        for title, rows in _SECTIONS:
            assert len(rows) > 0, f"Section '{title}' has no entries"

    def test_section_titles_are_strings(self) -> None:
        for title, _ in _SECTIONS:
            assert isinstance(title, str) and title

    def test_shortcut_strings_non_empty(self) -> None:
        for _, rows in _SECTIONS:
            for func, keys in rows:
                assert func and keys

    def test_minimum_width(self, dialog: ShortcutsDialog) -> None:
        assert dialog.minimumWidth() >= 400

    def test_delete_on_close(self, dialog: ShortcutsDialog) -> None:
        from PySide6.QtCore import Qt
        assert dialog.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    def test_ctrl_n_shortcut_present(self) -> None:
        all_shortcuts = {keys for _, rows in _SECTIONS for _, keys in rows}
        assert "Ctrl+N" in all_shortcuts

    def test_ctrl_r_shortcut_present(self) -> None:
        all_shortcuts = {keys for _, rows in _SECTIONS for _, keys in rows}
        assert "Ctrl+R" in all_shortcuts

    def test_esc_shortcut_present(self) -> None:
        all_shortcuts = {keys for _, rows in _SECTIONS for _, keys in rows}
        assert "Esc" in all_shortcuts

    def test_no_duplicate_shortcuts(self) -> None:
        all_shortcuts = [keys for _, rows in _SECTIONS for _, keys in rows]
        assert len(all_shortcuts) == len(set(all_shortcuts)), "Duplicate shortcut entries found"

    def test_dialog_accepts_parent(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PySide6.QtWidgets import QWidget
        parent = QWidget()
        qtbot.addWidget(parent)
        d = ShortcutsDialog(parent)
        qtbot.addWidget(d)
        assert d.parent() is parent
