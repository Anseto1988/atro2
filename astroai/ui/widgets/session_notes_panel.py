"""Session notes dock panel — free-text notes persisted to project metadata."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

__all__ = ["SessionNotesPanel"]


class SessionNotesPanel(QWidget):
    """Free-text notes for the imaging session, persisted in project metadata.description."""

    text_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Session-Notizen"))
        self._editor = QTextEdit()
        self._editor.setPlaceholderText("Notizen zur Beobachtungsnacht...")
        self._editor.setAcceptRichText(False)
        layout.addWidget(self._editor)
        self._editor.textChanged.connect(self._on_text_changed)

    @property
    def notes(self) -> str:
        return self._editor.toPlainText()

    def set_notes(self, text: str) -> None:
        self._editor.blockSignals(True)
        self._editor.setPlainText(text)
        self._editor.blockSignals(False)

    def _on_text_changed(self) -> None:
        self.text_changed.emit(self._editor.toPlainText())
