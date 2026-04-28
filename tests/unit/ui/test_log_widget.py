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

    def test_export_writes_file_when_path_given(self, widget: LogWidget, tmp_path) -> None:
        widget.append_message("export content", logging.INFO)
        out = str(tmp_path / "log.txt")
        with patch("astroai.ui.widgets.log_widget.QFileDialog.getSaveFileName", return_value=(out, "")):
            widget._on_export()
        assert "export content" in (tmp_path / "log.txt").read_text(encoding="utf-8")

    def test_export_dialog_cancelled_no_write(self, widget: LogWidget, tmp_path) -> None:
        widget.append_message("some text", logging.INFO)
        with patch("astroai.ui.widgets.log_widget.QFileDialog.getSaveFileName", return_value=("", "")):
            widget._on_export()

    def test_install_root_handler_adds_to_root_logger(self, widget: LogWidget) -> None:
        root = logging.getLogger()
        before = list(root.handlers)
        handler = widget.install_root_handler()
        assert handler in root.handlers
        root.removeHandler(handler)
        assert handler not in root.handlers

    def test_append_message_unknown_level_fallback(self, widget: LogWidget) -> None:
        widget.append_message("mystery level", 9999)
        assert "mystery level" in widget.get_plain_text()
