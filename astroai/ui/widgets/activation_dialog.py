"""License key activation dialog."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from astroai.ui.license_adapter import QLicenseAdapter

__all__ = ["ActivationDialog"]

_KEY_PATTERN = "ASTRO-XXXX-XXXX-XXXX"


class ActivationDialog(QDialog):
    """Modal dialog for entering and activating a license key."""

    activation_requested = Signal(str)  # license key

    def __init__(self, adapter: QLicenseAdapter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._adapter = adapter
        self.setWindowTitle("Lizenz aktivieren")
        self.setMinimumWidth(420)
        self.setModal(True)
        self._build_ui()
        self._connect()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Lizenzschlüssel eingeben")
        title.setObjectName("sectionHeader")
        layout.addWidget(title)

        hint = QLabel(f"Format: {_KEY_PATTERN}")
        hint.setObjectName("activationHint")
        layout.addWidget(hint)

        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText(_KEY_PATTERN)
        self._key_input.setMaxLength(24)
        layout.addWidget(self._key_input)

        self._error_label = QLabel()
        self._error_label.setObjectName("activationError")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        layout.addWidget(self._error_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._success_label = QLabel()
        self._success_label.setObjectName("activationSuccess")
        self._success_label.hide()
        layout.addWidget(self._success_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._deactivate_btn = QPushButton("Deaktivieren")
        self._deactivate_btn.setObjectName("deactivateButton")
        self._deactivate_btn.setVisible(self._adapter.is_activated)
        btn_row.addWidget(self._deactivate_btn)

        self._activate_btn = QPushButton("Aktivieren")
        self._activate_btn.setEnabled(False)
        btn_row.addWidget(self._activate_btn)

        self._close_btn = QPushButton("Schließen")
        btn_row.addWidget(self._close_btn)

        layout.addLayout(btn_row)

    def _connect(self) -> None:
        self._key_input.textChanged.connect(self._on_key_changed)
        self._activate_btn.clicked.connect(self._on_activate)
        self._deactivate_btn.clicked.connect(self._on_deactivate)
        self._close_btn.clicked.connect(self.reject)
        self._adapter.activation_started.connect(self._on_started)
        self._adapter.activation_succeeded.connect(self._on_succeeded)
        self._adapter.activation_failed.connect(self._on_failed)

    @Slot(str)
    def _on_key_changed(self, text: str) -> None:
        self._activate_btn.setEnabled(len(text.strip()) >= 10)
        self._error_label.hide()
        self._success_label.hide()

    @Slot()
    def _on_activate(self) -> None:
        key = self._key_input.text().strip()
        if not key:
            return
        self.activation_requested.emit(key)
        self._adapter.activate_async(key)

    @Slot()
    def _on_deactivate(self) -> None:
        self._adapter.deactivate()
        self._deactivate_btn.setVisible(False)
        self._success_label.hide()
        self._error_label.hide()
        self._key_input.clear()
        self._key_input.setEnabled(True)
        self._activate_btn.setEnabled(False)

    @Slot()
    def _on_started(self) -> None:
        self._activate_btn.setEnabled(False)
        self._key_input.setEnabled(False)
        self._error_label.hide()
        self._success_label.hide()
        self._progress.show()

    @Slot(object)
    def _on_succeeded(self, status: object) -> None:
        self._progress.hide()
        self._key_input.setEnabled(True)
        self._success_label.setText("Lizenz erfolgreich aktiviert!")
        self._success_label.show()
        self._deactivate_btn.setVisible(True)

    @Slot(str)
    def _on_failed(self, error: str) -> None:
        self._progress.hide()
        self._key_input.setEnabled(True)
        self._activate_btn.setEnabled(len(self._key_input.text().strip()) >= 10)
        self._error_label.setText(error)
        self._error_label.show()
