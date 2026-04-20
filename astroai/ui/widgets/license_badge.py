"""Compact license-tier badge for the status bar."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel, QWidget

from astroai.licensing.models import LicenseStatus, LicenseTier

__all__ = ["LicenseBadge"]

_TIER_LABELS = {
    LicenseTier.FREE: "Free",
    LicenseTier.PRO_MONTHLY: "Pro",
    LicenseTier.PRO_ANNUAL: "Pro",
    LicenseTier.FOUNDING_MEMBER: "Founding",
}

_TIER_OBJECTS = {
    LicenseTier.FREE: "licenseBadgeFree",
    LicenseTier.PRO_MONTHLY: "licenseBadgePro",
    LicenseTier.PRO_ANNUAL: "licenseBadgePro",
    LicenseTier.FOUNDING_MEMBER: "licenseBadgeFounding",
}


class LicenseBadge(QLabel):
    """Small tier badge displayed in the application status bar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tier = LicenseTier.FREE
        self._apply(LicenseTier.FREE)

    @Slot(object)
    def on_status_changed(self, status: LicenseStatus) -> None:
        tier = status.token.tier if status.token else LicenseTier.FREE
        if tier != self._tier:
            self._tier = tier
            self._apply(tier)

    def _apply(self, tier: LicenseTier) -> None:
        self.setText(f"  {_TIER_LABELS.get(tier, 'Free')}  ")
        self.setObjectName(_TIER_OBJECTS.get(tier, "licenseBadgeFree"))
        self.style().unpolish(self)
        self.style().polish(self)
        self.setToolTip(f"Lizenz-Tier: {_TIER_LABELS.get(tier, 'Free')}")
