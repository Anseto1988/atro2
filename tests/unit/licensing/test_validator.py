"""Unit tests for JWT validation and grace-period logic."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from astroai.licensing.exceptions import GracePeriodExpired, LicenseError, NotActivated
from astroai.licensing.machine import get_machine_id
from astroai.licensing.models import LicenseTier
from astroai.licensing.validator import GRACE_PERIOD_DAYS, decode_token, validate_offline


@pytest.fixture(scope="module")
def rsa_keypair() -> tuple[str, str]:
    """Generate a test RSA key pair."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


def _make_token(
    private_key: str,
    tier: str = "pro_annual",
    machine_id: str | None = None,
    exp_delta: timedelta = timedelta(days=30),
) -> str:
    now = int(time.time())
    payload = {
        "sub": "user@test.com",
        "jti": "test-jti-001",
        "iat": now,
        "exp": now + int(exp_delta.total_seconds()),
        "tier": tier,
        "plugins": ["denoise_pro", "starnet_pro"],
        "machine_id": machine_id or get_machine_id(),
        "seats_used": 1,
        "seats_max": 1,
    }
    return pyjwt.encode(payload, private_key, algorithm="RS256")


class TestDecodeToken:
    def test_valid_token(self, rsa_keypair: tuple[str, str]) -> None:
        private_key, public_key = rsa_keypair
        raw = _make_token(private_key)
        token = decode_token(raw, public_key)

        assert token.sub == "user@test.com"
        assert token.tier == LicenseTier.PRO_ANNUAL
        assert "denoise_pro" in token.plugins
        assert token.machine_id == get_machine_id()

    def test_expired_token_raises(self, rsa_keypair: tuple[str, str]) -> None:
        private_key, public_key = rsa_keypair
        raw = _make_token(private_key, exp_delta=timedelta(seconds=-10))
        with pytest.raises(LicenseError, match="expired"):
            decode_token(raw, public_key)

    def test_invalid_signature_raises(self, rsa_keypair: tuple[str, str]) -> None:
        private_key, _ = rsa_keypair
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        other_pub = other_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        raw = _make_token(private_key)
        with pytest.raises(LicenseError, match="Invalid"):
            decode_token(raw, other_pub)


class TestGracePeriod:
    def test_online_success(self, rsa_keypair: tuple[str, str]) -> None:
        """Token validates when last_online_at is recent."""
        private_key, public_key = rsa_keypair
        raw = _make_token(private_key)
        now = datetime.now(timezone.utc)
        last_online = now - timedelta(hours=2)

        token = validate_offline(raw, last_online, public_key, now)
        assert token.sub == "user@test.com"

    def test_offline_within_grace(self, rsa_keypair: tuple[str, str]) -> None:
        """Token validates when offline less than 7 days."""
        private_key, public_key = rsa_keypair
        raw = _make_token(private_key)
        now = datetime.now(timezone.utc)
        last_online = now - timedelta(days=6)

        token = validate_offline(raw, last_online, public_key, now)
        assert token.tier == LicenseTier.PRO_ANNUAL

    def test_offline_expired_grace(self, rsa_keypair: tuple[str, str]) -> None:
        """GracePeriodExpired raised when offline >= 7 days."""
        private_key, public_key = rsa_keypair
        raw = _make_token(private_key)
        now = datetime.now(timezone.utc)
        last_online = now - timedelta(days=GRACE_PERIOD_DAYS)

        with pytest.raises(GracePeriodExpired) as exc_info:
            validate_offline(raw, last_online, public_key, now)
        assert exc_info.value.days_offline >= GRACE_PERIOD_DAYS

    def test_no_last_online_raises_not_activated(self, rsa_keypair: tuple[str, str]) -> None:
        """NotActivated raised when last_online_at is None."""
        private_key, public_key = rsa_keypair
        raw = _make_token(private_key)
        with pytest.raises(NotActivated):
            validate_offline(raw, None, public_key)

    def test_machine_mismatch_raises(self, rsa_keypair: tuple[str, str]) -> None:
        """LicenseError raised when machine_id doesn't match."""
        private_key, public_key = rsa_keypair
        raw = _make_token(private_key, machine_id="sha256:ffffffffffffffffffffffffffffffff")
        now = datetime.now(timezone.utc)
        last_online = now - timedelta(hours=1)

        with pytest.raises(LicenseError, match="Machine ID mismatch"):
            validate_offline(raw, last_online, public_key, now)
