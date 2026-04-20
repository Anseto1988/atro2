"""License data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence


class LicenseTier(str, Enum):
    """Available subscription tiers."""

    FREE = "free"
    PRO_MONTHLY = "pro_monthly"
    PRO_ANNUAL = "pro_annual"
    FOUNDING_MEMBER = "founding_member"


@dataclass(frozen=True, slots=True)
class LicenseToken:
    """Decoded JWT license token claims."""

    sub: str
    jti: str
    iat: datetime
    exp: datetime
    tier: LicenseTier
    plugins: tuple[str, ...]
    machine_id: str
    seats_used: int
    seats_max: int

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.exp

    def has_plugin(self, plugin_name: str) -> bool:
        return plugin_name in self.plugins


@dataclass(frozen=True, slots=True)
class TimeAttestation:
    """Server-signed proof of last online timestamp (RS256-verified)."""

    user_id: str
    machine_id: str
    last_online_at: datetime  # server-authoritative UTC timestamp
    iat: datetime


@dataclass(slots=True)
class LicenseStatus:
    """Runtime license state (persisted alongside the token)."""

    token: LicenseToken | None = None
    attestation: TimeAttestation | None = None  # server-signed time proof
    last_online_at: datetime | None = None
    activated: bool = False
    grace_remaining_days: int = 7
    allowed_plugins: Sequence[str] = field(default_factory=list)
    start_counter: int = 0  # offline app-start count (resets on online refresh)
