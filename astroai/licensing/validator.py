"""JWT RS256 offline verification and grace-period logic."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jwt

from astroai.licensing.exceptions import (
    GracePeriodExpired,
    LicenseError,
    NotActivated,
    OfflineStartLimitExceeded,
    TimeRollbackDetected,
)
from astroai.licensing.machine import verify_machine_id
from astroai.licensing.models import LicenseToken, LicenseTier, TimeAttestation


GRACE_PERIOD_DAYS = 7
MAX_OFFLINE_STARTS = 50
_PUBLIC_KEY_PATH = Path(__file__).parent / "keys" / "public.pem"


def _load_public_key(key_path: Path | None = None) -> str:
    path = key_path or _PUBLIC_KEY_PATH
    if not path.exists():
        raise LicenseError(f"Public key not found: {path}")
    return path.read_text(encoding="utf-8")


def decode_token(raw_jwt: str, public_key: str | None = None) -> LicenseToken:
    """Decode and verify a JWT RS256 license token."""
    key = public_key or _load_public_key()
    try:
        payload: dict[str, Any] = jwt.decode(
            raw_jwt,
            key,
            algorithms=["RS256"],
            options={"require": ["sub", "jti", "iat", "exp", "tier", "machine_id"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise LicenseError("License token has expired") from e
    except jwt.InvalidTokenError as e:
        raise LicenseError(f"Invalid license token: {e}") from e

    return LicenseToken(
        sub=payload["sub"],
        jti=payload["jti"],
        iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        tier=LicenseTier(payload["tier"]),
        plugins=tuple(payload.get("plugins", [])),
        machine_id=payload["machine_id"],
        seats_used=int(payload.get("seats_used", 1)),
        seats_max=int(payload.get("seats_max", 1)),
    )


def decode_attestation(raw_jwt: str, public_key: str | None = None) -> TimeAttestation:
    """Decode and verify a server-signed time attestation (RS256, no expiry check)."""
    key = public_key or _load_public_key()
    try:
        payload: dict[str, Any] = jwt.decode(
            raw_jwt,
            key,
            algorithms=["RS256"],
            options={
                "require": ["user_id", "machine_id", "last_online_at", "iat"],
                "verify_exp": False,
                "verify_iat": False,
            },
        )
    except jwt.InvalidTokenError as e:
        raise LicenseError(f"Invalid time attestation: {e}") from e

    return TimeAttestation(
        user_id=payload["user_id"],
        machine_id=payload["machine_id"],
        last_online_at=datetime.fromtimestamp(payload["last_online_at"], tz=timezone.utc),
        iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
    )


def validate_offline(
    raw_jwt: str,
    last_online_at: datetime | None,
    public_key: str | None = None,
    now: datetime | None = None,
    raw_attestation: str | None = None,
    start_counter: int = 0,
) -> LicenseToken:
    """Validate a stored token offline with server-attested grace-period enforcement.

    Security properties:
    - Uses server-signed attestation for last_online_at when available (tamper-proof)
    - Detects system clock rollback attacks
    - Enforces monotone offline start counter (max MAX_OFFLINE_STARTS)

    Returns the decoded token if valid.
    Raises GracePeriodExpired, OfflineStartLimitExceeded, TimeRollbackDetected, or NotActivated.
    """
    if last_online_at is None and raw_attestation is None:
        raise NotActivated("No activation record found")

    current = now or datetime.now(timezone.utc)

    token = decode_token(raw_jwt, public_key)

    if not verify_machine_id(token.machine_id):
        raise LicenseError("Machine ID mismatch — license bound to a different device")

    # Prefer server-attested last_online_at over client-stored value
    trusted_last_online: datetime
    if raw_attestation is not None:
        attestation = decode_attestation(raw_attestation, public_key)
        if not verify_machine_id(attestation.machine_id):
            raise LicenseError("Attestation machine ID mismatch")
        trusted_last_online = attestation.last_online_at
    elif last_online_at is not None:
        trusted_last_online = last_online_at
    else:  # pragma: no cover
        raise NotActivated("No activation record found")

    # Clock rollback detection: system time must not be before last server sync
    rollback_delta = (trusted_last_online - current).total_seconds()
    if rollback_delta > 60:  # 60s tolerance for clock drift
        raise TimeRollbackDetected(rollback_delta)

    # Grace period check (based on server-authoritative timestamp)
    days_offline = (current - trusted_last_online).days
    if days_offline >= GRACE_PERIOD_DAYS:
        raise GracePeriodExpired(days_offline)

    # Offline start counter enforcement
    if start_counter >= MAX_OFFLINE_STARTS:
        raise OfflineStartLimitExceeded(start_counter, MAX_OFFLINE_STARTS)

    return token
