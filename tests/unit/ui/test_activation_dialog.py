"""Tests for ActivationDialog – 100% statement coverage."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from astroai.ui.widgets.activation_dialog import ActivationDialog


def _make_adapter(activated: bool = False) -> MagicMock:
    mock = MagicMock()
    mock.is_activated = activated
    for sig in ("activation_started", "activation_succeeded", "activation_failed"):
        s = MagicMock()
        s.connect = MagicMock()
        setattr(mock, sig, s)
    return mock


class TestActivationDialogInit:
    @pytest.fixture()
    def adapter(self) -> MagicMock:
        return _make_adapter()

    @pytest.fixture()
    def dialog(self, qtbot, adapter) -> ActivationDialog:
        dlg = ActivationDialog(adapter)
        qtbot.addWidget(dlg)
        return dlg

    def test_window_title(self, dialog: ActivationDialog) -> None:
        assert "Lizenz" in dialog.windowTitle()

    def test_activate_button_disabled_on_empty_input(self, dialog: ActivationDialog) -> None:
        assert not dialog._activate_btn.isEnabled()

    def test_deactivate_hidden_when_not_activated(self, dialog: ActivationDialog) -> None:
        assert dialog._deactivate_btn.isHidden()

    def test_error_label_hidden_initially(self, dialog: ActivationDialog) -> None:
        assert dialog._error_label.isHidden()

    def test_success_label_hidden_initially(self, dialog: ActivationDialog) -> None:
        assert dialog._success_label.isHidden()

    def test_progress_hidden_initially(self, dialog: ActivationDialog) -> None:
        assert dialog._progress.isHidden()


class TestActivationDialogKeyInput:
    @pytest.fixture()
    def dialog(self, qtbot) -> ActivationDialog:
        dlg = ActivationDialog(_make_adapter())
        qtbot.addWidget(dlg)
        return dlg

    def test_activate_enabled_for_long_key(self, dialog: ActivationDialog) -> None:
        dialog._key_input.setText("ASTRO-1234-5678-ABCD")
        assert dialog._activate_btn.isEnabled()

    def test_activate_disabled_for_short_key(self, dialog: ActivationDialog) -> None:
        dialog._key_input.setText("SHORT")
        assert not dialog._activate_btn.isEnabled()

    def test_error_clears_on_new_input(self, dialog: ActivationDialog) -> None:
        dialog._on_failed("some error")
        assert not dialog._error_label.isHidden()
        dialog._key_input.setText("ASTRO-NEWKEY-HERE-XX")
        assert dialog._error_label.isHidden()

    def test_success_clears_on_new_input(self, dialog: ActivationDialog) -> None:
        dialog._on_succeeded(None)
        assert not dialog._success_label.isHidden()
        dialog._key_input.setText("ASTRO-NEWKEY-HERE-XX")
        assert dialog._success_label.isHidden()


class TestActivationDialogActivate:
    @pytest.fixture()
    def adapter(self) -> MagicMock:
        return _make_adapter()

    @pytest.fixture()
    def dialog(self, qtbot, adapter) -> ActivationDialog:
        dlg = ActivationDialog(adapter)
        qtbot.addWidget(dlg)
        return dlg

    def test_activate_emits_signal(self, dialog: ActivationDialog, qtbot) -> None:
        dialog._key_input.setText("ASTRO-1234-5678-ABCD")
        with qtbot.waitSignal(dialog.activation_requested, timeout=500) as blocker:
            dialog._on_activate()
        assert blocker.args == ["ASTRO-1234-5678-ABCD"]

    def test_activate_calls_adapter(self, dialog: ActivationDialog, adapter: MagicMock) -> None:
        dialog._key_input.setText("ASTRO-1234-5678-ABCD")
        dialog._on_activate()
        adapter.activate_async.assert_called_once_with("ASTRO-1234-5678-ABCD")

    def test_activate_empty_key_no_signal(self, dialog: ActivationDialog, qtbot) -> None:
        dialog._key_input.setText("   ")
        with qtbot.assertNotEmitted(dialog.activation_requested):
            dialog._on_activate()

    def test_on_started_shows_progress(self, dialog: ActivationDialog) -> None:
        dialog._on_started()
        assert not dialog._progress.isHidden()
        assert not dialog._activate_btn.isEnabled()
        assert not dialog._key_input.isEnabled()

    def test_on_succeeded_shows_success_label(self, dialog: ActivationDialog) -> None:
        dialog._on_started()
        dialog._on_succeeded(None)
        assert dialog._progress.isHidden()
        assert not dialog._success_label.isHidden()
        assert not dialog._deactivate_btn.isHidden()

    def test_on_failed_shows_error_label(self, dialog: ActivationDialog) -> None:
        dialog._on_started()
        dialog._on_failed("invalid_key")
        assert dialog._progress.isHidden()
        assert not dialog._error_label.isHidden()
        assert "invalid_key" in dialog._error_label.text()

    def test_on_failed_re_enables_activate_for_long_key(self, dialog: ActivationDialog) -> None:
        dialog._key_input.setText("ASTRO-1234-5678-ABCD")
        dialog._on_started()
        dialog._on_failed("error")
        assert dialog._activate_btn.isEnabled()


class TestActivationDialogDeactivate:
    @pytest.fixture()
    def adapter(self) -> MagicMock:
        return _make_adapter(activated=True)

    @pytest.fixture()
    def dialog(self, qtbot, adapter) -> ActivationDialog:
        dlg = ActivationDialog(adapter)
        qtbot.addWidget(dlg)
        return dlg

    def test_deactivate_visible_when_activated(self, dialog: ActivationDialog) -> None:
        assert not dialog._deactivate_btn.isHidden()

    def test_deactivate_calls_adapter(self, dialog: ActivationDialog, adapter: MagicMock) -> None:
        dialog._on_deactivate()
        adapter.deactivate.assert_called_once()

    def test_deactivate_hides_deactivate_btn(self, dialog: ActivationDialog) -> None:
        dialog._on_deactivate()
        assert dialog._deactivate_btn.isHidden()

    def test_deactivate_clears_input(self, dialog: ActivationDialog) -> None:
        dialog._key_input.setText("ASTRO-1234-5678-ABCD")
        dialog._on_deactivate()
        assert dialog._key_input.text() == ""

    def test_deactivate_disables_activate_btn(self, dialog: ActivationDialog) -> None:
        dialog._key_input.setText("ASTRO-1234-5678-ABCD")
        dialog._on_deactivate()
        assert not dialog._activate_btn.isEnabled()

    def test_close_btn_rejects(self, dialog: ActivationDialog, qtbot) -> None:
        with qtbot.waitSignal(dialog.rejected, timeout=500):
            dialog._close_btn.click()
