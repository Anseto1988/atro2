"""Tests for LogWidget and WidgetLogHandler."""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from astroai.ui.widgets.log_widget import LogWidget, WidgetLogHandler


class TestLogWidget:
    @pytest.fixture()
    def widget(self, qtbot) -> LogWidget:  # type: ignore[no-untyped-def]
        w = LogWidget()
        qtbot.addWidget(w)
        return w

    def test_initial_state_empty(self, widget: LogWidget) -> None:
        assert widget.get_plain_text() == ""

    def test_append_message_info(self, widget: LogWidget) -> None:
        widget.append_message("Test info", logging.INFO)
        assert "Test info" in widget.get_plain_text()

    def test_append_message_error(self, widget: LogWidget) -> None:
        widget.append_message("Fehler aufgetreten", logging.ERROR)
        assert "Fehler aufgetreten" in widget.get_plain_text()

    def test_clear_removes_all(self, widget: LogWidget) -> None:
        widget.append_message("Nachricht 1", logging.INFO)
        widget.append_message("Nachricht 2", logging.WARNING)
        widget.clear()
        assert widget.get_plain_text() == ""

    def test_create_handler_returns_handler(self, widget: LogWidget) -> None:
        handler = widget.create_handler()
        assert isinstance(handler, WidgetLogHandler)
        assert handler.level == logging.DEBUG

    def test_handler_emits_to_widget(self, widget: LogWidget, qtbot) -> None:  # type: ignore[no-untyped-def]
        handler = widget.create_handler(logging.WARNING)
        logger = logging.getLogger("test.log_widget")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            with qtbot.waitSignal(widget.message_logged, timeout=1000):
                logger.warning("Warnung test")
            assert "Warnung test" in widget.get_plain_text()
        finally:
            logger.removeHandler(handler)

    def test_handler_respects_level(self, widget: LogWidget) -> None:
        handler = widget.create_handler(logging.ERROR)
        logger = logging.getLogger("test.log_widget.level")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.info("Should not appear")
            assert "Should not appear" not in widget.get_plain_text()
        finally:
            logger.removeHandler(handler)

    def test_export_no_crash_when_empty(self, widget: LogWidget) -> None:
        with patch("astroai.ui.widgets.log_widget.QFileDialog.getSaveFileName", return_value=("", "")):
            widget._on_export()
        assert widget.get_plain_text() == ""
