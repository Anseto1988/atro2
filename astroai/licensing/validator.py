"""JWT RS256 offline verification and grace-period logic."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jwt

from astroai.licensing.exceptions import GracePeriodExpired, LicenseError, NotActivated
from astroai.licensing.machine import verify_machine_id
from astroai.licensing.models import LicenseToken, LicenseTier


GRACE_PERIOD_DAYS = 7
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


def validate_offline(
    raw_jwt: str,
    last_online_at: datetime | None,
    public_key: str | None = None,
    now: datetime | None = None,
) -> LicenseToken:
    """Validate a stored token offline with grace-period enforcement.

    Returns the decoded token if valid.
    Raises GracePeriodExpired if offline longer than GRACE_PERIOD_DAYS.
    Raises NotActivated if no token or last_online_at is provided.
    """
    if last_online_at is None:
        raise NotActivated("No activation record found")

    current = now or datetime.now(timezone.utc)

    token = decode_token(raw_jwt, public_key)

    if not verify_machine_id(token.machine_id):
        raise LicenseError("Machine ID mismatch — license bound to a different device")

    days_offline = (current - last_online_at).days
    if days_offline >= GRACE_PERIOD_DAYS:
        raise GracePeriodExpired(days_offline)

    return token
