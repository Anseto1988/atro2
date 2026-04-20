"""Fehler-Log-Panel mit farbcodiertem Logging und Export-Funktion."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from logging import LogRecord

__all__ = ["LogWidget", "WidgetLogHandler"]

_LEVEL_COLORS: dict[int, QColor] = {
    logging.DEBUG: QColor("#888888"),
    logging.INFO: QColor("#CCCCCC"),
    logging.WARNING: QColor("#FFA500"),
    logging.ERROR: QColor("#FF4444"),
    logging.CRITICAL: QColor("#FF0000"),
}


class _LogSignalBridge(QObject):
    """Thread-safe bridge: emits signal so GUI updates happen on the main thread."""

    log_received = Signal(str, int)


class WidgetLogHandler(logging.Handler):
    """Python logging handler that forwards records to LogWidget."""

    def __init__(self, widget: LogWidget) -> None:
        super().__init__()
        self._widget = widget
        self._bridge = _LogSignalBridge()
        self._bridge.log_received.connect(self._widget.append_message)

    def emit(self, record: LogRecord) -> None:
        msg = self.format(record)
        self._bridge.log_received.emit(msg, record.levelno)


class LogWidget(QWidget):
    """QTextEdit-basiertes Log-Panel mit farbcodierten Log-Levels."""

    message_logged = Signal(str, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()
        self._log_handler: WidgetLogHandler | None = None

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setAccessibleName("Fehler-Log")
        self._text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._text_edit, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)

        self._export_btn = QPushButton("Exportieren")
        self._export_btn.setObjectName("exportLogButton")
        self._export_btn.setFixedWidth(100)
        self._export_btn.setAccessibleName("Log exportieren")
        self._export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self._export_btn)

        self._clear_btn = QPushButton("Leeren")
        self._clear_btn.setObjectName("clearLogButton")
        self._clear_btn.setFixedWidth(80)
        self._clear_btn.setAccessibleName("Log leeren")
        self._clear_btn.clicked.connect(self.clear)
        btn_layout.addWidget(self._clear_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    @Slot(str, int)
    def append_message(self, message: str, level: int = logging.INFO) -> None:
        color = _LEVEL_COLORS.get(level, _LEVEL_COLORS[logging.INFO])
        fmt = QTextCharFormat()
        fmt.setForeground(color)

        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(message + "\n", fmt)
        self._text_edit.setTextCursor(cursor)
        self._text_edit.ensureCursorVisible()
        self.message_logged.emit(message, level)

    @Slot()
    def clear(self) -> None:
        self._text_edit.clear()

    def get_plain_text(self) -> str:
        return self._text_edit.toPlainText()

    def create_handler(self, level: int = logging.DEBUG) -> WidgetLogHandler:
        handler = WidgetLogHandler(self)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        self._log_handler = handler
        return handler

    def install_root_handler(self, level: int = logging.DEBUG) -> WidgetLogHandler:
        handler = self.create_handler(level)
        logging.getLogger().addHandler(handler)
        return handler

    @Slot()
    def _on_export(self) -> None:
        text = self.get_plain_text()
        if not text.strip():
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"astroai_log_{timestamp}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Log exportieren", default_name, "Textdatei (*.txt);;Alle (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
