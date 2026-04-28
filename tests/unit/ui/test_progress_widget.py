"""Tests for ProgressWidget."""
from __future__ import annotations

import pytest

from astroai.ui.widgets.progress_widget import ProgressWidget


@pytest.fixture()
def widget(qtbot):
    w = ProgressWidget()
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_label_default_text(self, widget):
        assert widget._label.text() == "Bereit"

    def test_bar_starts_at_zero(self, widget):
        assert widget._bar.value() == 0

    def test_bar_range(self, widget):
        assert widget._bar.maximum() == 1000

    def test_cancel_button_disabled_initially(self, widget):
        assert not widget._cancel_btn.isEnabled()


class TestSetStatus:
    def test_set_status_updates_label(self, widget):
        widget.set_status("Lädt...")
        assert widget._label.text() == "Lädt..."

    def test_set_status_empty_string(self, widget):
        widget.set_status("")
        assert widget._label.text() == ""


class TestSetProgress:
    def test_set_progress_half(self, widget):
        widget.set_progress(0.5)
        assert widget._bar.value() == 500

    def test_set_progress_full(self, widget):
        widget.set_progress(1.0)
        assert widget._bar.value() == 1000

    def test_set_progress_zero(self, widget):
        widget.set_progress(0.0)
        assert widget._bar.value() == 0

    def test_set_progress_quarter(self, widget):
        widget.set_progress(0.25)
        assert widget._bar.value() == 250


class TestSetCancellable:
    def test_set_cancellable_true(self, widget):
        widget.set_cancellable(True)
        assert widget._cancel_btn.isEnabled()

    def test_set_cancellable_false(self, widget):
        widget.set_cancellable(True)
        widget.set_cancellable(False)
        assert not widget._cancel_btn.isEnabled()


class TestCancelSignal:
    def test_cancel_button_emits_signal(self, widget, qtbot):
        widget.set_cancellable(True)
        with qtbot.waitSignal(widget.cancel_requested):
            widget._cancel_btn.click()


class TestReset:
    def test_reset_restores_label(self, widget):
        widget.set_status("Running")
        widget.reset()
        assert widget._label.text() == "Bereit"

    def test_reset_zeroes_bar(self, widget):
        widget.set_progress(0.8)
        widget.reset()
        assert widget._bar.value() == 0

    def test_reset_disables_cancel(self, widget):
        widget.set_cancellable(True)
        widget.reset()
        assert not widget._cancel_btn.isEnabled()


class TestIndeterminate:
    def test_set_indeterminate(self, widget):
        widget.set_indeterminate()
        assert widget._bar.minimum() == 0
        assert widget._bar.maximum() == 0

    def test_set_determinate_restores_range(self, widget):
        widget.set_indeterminate()
        widget.set_determinate()
        assert widget._bar.maximum() == 1000
