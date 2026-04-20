"""Modal dialog shown when a feature requires a higher license tier."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from astroai.licensing.models import LicenseTier

__all__ = ["UpgradeDialog"]

_TIER_LABELS = {
    LicenseTier.FREE: "Free",
    LicenseTier.PRO_MONTHLY: "Pro",
    LicenseTier.PRO_ANNUAL: "Pro",
    LicenseTier.FOUNDING_MEMBER: "Founding",
}


class UpgradeDialog(QDialog):
    """Prompts the user to upgrade when a tier-restricted feature is accessed."""

    upgrade_requested = Signal()
    activate_requested = Signal()

    def __init__(
        self,
        feature_name: str,
        required_tier: LicenseTier,
        current_tier: LicenseTier,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._feature = feature_name
        self._required = required_tier
        self._current = current_tier
        self.setWindowTitle("Upgrade erforderlich")
        self.setMinimumWidth(380)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Feature nicht verfügbar")
        title.setObjectName("sectionHeader")
        layout.addWidget(title)

        current_label = _TIER_LABELS.get(self._current, "Free")
        required_label = _TIER_LABELS.get(self._required, "Pro")
        body = QLabel(
            f'Das Feature „{self._feature}" erfordert mindestens den '
            f"<b>{required_label}</b>-Tier.\n\n"
            f"Dein aktueller Tier: <b>{current_label}</b>"
        )
        body.setObjectName("upgradeBody")
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(body)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        if self._current == LicenseTier.FREE:
            activate_btn = QPushButton("Lizenz eingeben")
            activate_btn.clicked.connect(self._on_activate)
            btn_row.addWidget(activate_btn)

        upgrade_btn = QPushButton("Upgrade-Info")
        upgrade_btn.setObjectName("upgradeButton")
        upgrade_btn.clicked.connect(self._on_upgrade)
        btn_row.addWidget(upgrade_btn)

        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

    def _on_upgrade(self) -> None:
        self.upgrade_requested.emit()

    def _on_activate(self) -> None:
        self.activate_requested.emit()
        self.accept()
