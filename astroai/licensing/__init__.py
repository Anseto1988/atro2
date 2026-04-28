"""AstroAI Licensing — LicenseManager facade."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from astroai.licensing.client import LicenseClient
from astroai.licensing.exceptions import (
    GracePeriodExpired,
    LicenseError,
    NotActivated,
    TierInsufficientError,
)
from astroai.licensing.models import LicenseStatus, LicenseToken
from astroai.licensing.store import LicenseStore
from astroai.licensing.validator import GRACE_PERIOD_DAYS, decode_token, validate_offline

__all__ = [
    "LicenseManager",
    "LicenseStatus",
    "LicenseToken",
    "LicenseError",
    "NotActivated",
    "GracePeriodExpired",
    "TierInsufficientError",
]

logger = logging.getLogger(__name__)


class LicenseManager:
    """High-level facade for license operations.

    Orchestrates store, validator, and HTTP client into a single entry point.
    """

    def __init__(
        self,
        store_dir: Path | None = None,
        base_url: str = "https://api.astroai.app",
        public_key: str | None = None,
    ) -> None:
        self._store = LicenseStore(base_dir=store_dir)
        self._client = LicenseClient(base_url=base_url)
        self._public_key = public_key

    def activate(self, license_key: str, app_version: str) -> LicenseToken:
        """Activate a license key on this machine."""
        raw_jwt, attestation = self._client.activate(license_key, app_version)
        token = decode_token(raw_jwt, self._public_key)
        now = datetime.now(timezone.utc)
        self._store.save(raw_jwt, now, attestation_raw=attestation)
        logger.info("License activated: tier=%s, expires=%s", token.tier.value, token.exp)
        return token

    def verify(self) -> LicenseStatus:
        """Verify current license status with online refresh attempt.

        1. Load token from encrypted store.
        2. Try online refresh (5s timeout).
        3. On network failure: check offline grace period.
        4. Returns LicenseStatus describing current state.
        """
        stored = self._store.load()
        if stored is None:
            raise NotActivated("No license found — please activate first")

        raw_jwt, last_online_at, attestation_raw, start_counter = stored
        now = datetime.now(timezone.utc)

        # Attempt online refresh
        try:
            new_jwt, new_attestation = self._client.refresh(raw_jwt)
            token = decode_token(new_jwt, self._public_key)
            self._store.save(new_jwt, now, attestation_raw=new_attestation, start_counter=0)
            logger.debug("License refreshed online")
            return LicenseStatus(
                token=token,
                last_online_at=now,
                activated=True,
                grace_remaining_days=GRACE_PERIOD_DAYS,
                allowed_plugins=list(token.plugins),
            )
        except LicenseError:
            logger.debug("Online refresh failed, checking grace period")

        # Offline validation with grace period
        token = validate_offline(
            raw_jwt, last_online_at, self._public_key, now,
            raw_attestation=attestation_raw, start_counter=start_counter,
        )
        self._store.increment_start_counter()
        days_offline = (now - last_online_at).days
        grace_remaining = max(0, GRACE_PERIOD_DAYS - days_offline)

        return LicenseStatus(
            token=token,
            last_online_at=last_online_at,
            activated=True,
            grace_remaining_days=grace_remaining,
            allowed_plugins=list(token.plugins),
        )

    def deactivate(self) -> None:
        """Deactivate license on this machine and clear local store."""
        stored = self._store.load()
        if stored is not None:
            raw_jwt, *_ = stored
            try:
                self._client.deactivate(raw_jwt)
            except LicenseError:
                logger.warning("Server deactivation failed — clearing local store anyway")
        self._store.clear()
        logger.info("License deactivated and local store cleared")

    def get_status(self) -> LicenseStatus:
        """Get current license status without triggering refresh."""
        stored = self._store.load()
        if stored is None:
            return LicenseStatus()

        raw_jwt, last_online_at, *_ = stored
        try:
            token = decode_token(raw_jwt, self._public_key)
        except LicenseError:
            return LicenseStatus()

        now = datetime.now(timezone.utc)
        days_offline = (now - last_online_at).days
        grace_remaining = max(0, GRACE_PERIOD_DAYS - days_offline)

        return LicenseStatus(
            token=token,
            last_online_at=last_online_at,
            activated=True,
            grace_remaining_days=grace_remaining,
            allowed_plugins=list(token.plugins),
        )
