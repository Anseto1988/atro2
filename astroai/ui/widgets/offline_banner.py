"""Warning banner shown when offline grace period is critically low (<48h)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from astroai.licensing.models import LicenseStatus

__all__ = ["OfflineBanner"]

_THRESHOLD_DAYS = 2


class OfflineBanner(QWidget):
    """Horizontal warning banner — auto-shows when grace period < 48 hours."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("offlineBanner")
        self.setVisible(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)

        self._icon = QLabel("\u26A0")
        self._icon.setObjectName("offlineBannerIcon")
        layout.addWidget(self._icon)

        self._message = QLabel()
        self._message.setObjectName("offlineBannerText")
        self._message.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._message, stretch=1)

    @Slot(object)
    def on_status_changed(self, status: LicenseStatus) -> None:
        if not status.activated:
            self.setVisible(False)
            return

        remaining = status.grace_remaining_days
        if remaining <= 0:
            self._message.setText(
                "Offline-Zeitraum abgelaufen — bitte mit dem Internet verbinden, "
                "um die Lizenz zu erneuern."
            )
            self.setVisible(True)
        elif remaining < _THRESHOLD_DAYS:
            hours = remaining * 24
            self._message.setText(
                f"Noch ca. {hours:.0f}h Offline-Zeit verbleibend. "
                "Bitte bald eine Internetverbindung herstellen."
            )
            self.setVisible(True)
        else:
            self.setVisible(False)
