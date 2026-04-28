"""Unit tests for JWT validation and grace-period logic."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from astroai.licensing.exceptions import (
    GracePeriodExpired,
    LicenseError,
    NotActivated,
    OfflineStartLimitExceeded,
    TimeRollbackDetected,
)
from astroai.licensing.machine import get_machine_id
from astroai.licensing.models import LicenseTier
from astroai.licensing.validator import (
    GRACE_PERIOD_DAYS,
    MAX_OFFLINE_STARTS,
    _load_public_key,
    decode_attestation,
    decode_token,
    validate_offline,
)


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


def _make_attestation(
    private_key: str,
    machine_id: str | None = None,
    last_online_offset: timedelta = timedelta(hours=0),
    now: datetime | None = None,
) -> str:
    """Create a signed time attestation JWT for testing."""
    base = now or datetime.now(timezone.utc)
    ts = int((base + last_online_offset).timestamp())
    payload = {
        "user_id": "user@test.com",
        "machine_id": machine_id or get_machine_id(),
        "last_online_at": ts,
        "iat": ts,
    }
    return pyjwt.encode(payload, private_key, algorithm="RS256")


class TestLoadPublicKey:
    def test_missing_key_file_raises_license_error(self, tmp_path: Path) -> None:
        """_load_public_key raises LicenseError when key file absent (lines 29-30)."""
        missing = tmp_path / "nonexistent" / "public.pem"
        with pytest.raises(LicenseError, match="Public key not found"):
            _load_public_key(missing)

    def test_existing_key_file_returns_content(self, tmp_path: Path) -> None:
        """_load_public_key reads and returns key file content (line 31)."""
        key_file = tmp_path / "public.pem"
        key_file.write_text("-----BEGIN PUBLIC KEY-----\ntest\n", encoding="utf-8")
        content = _load_public_key(key_file)
        assert "BEGIN PUBLIC KEY" in content


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


class TestDecodeAttestation:
    def test_valid_attestation(self, rsa_keypair: tuple[str, str]) -> None:
        private_key, public_key = rsa_keypair
        raw = _make_attestation(private_key)
        att = decode_attestation(raw, public_key)

        assert att.user_id == "user@test.com"
        assert att.machine_id == get_machine_id()
        assert att.last_online_at.tzinfo is not None

    def test_invalid_signature_raises(self, rsa_keypair: tuple[str, str]) -> None:
        private_key, _ = rsa_keypair
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        other_pub = other_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        raw = _make_attestation(private_key)
        with pytest.raises(LicenseError, match="Invalid time attestation"):
            decode_attestation(raw, other_pub)


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
        """NotActivated raised when last_online_at is None and no attestation."""
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


class TestAttestation:
    def test_attestation_used_for_last_online(self, rsa_keypair: tuple[str, str]) -> None:
        """Attestation last_online_at overrides client-provided value."""
        private_key, public_key = rsa_keypair
        now = datetime.now(timezone.utc)
        raw_token = _make_token(private_key)
        # Attestation says 2h ago (safe) but client-stored says 8 days ago (would expire)
        raw_att = _make_attestation(private_key, last_online_offset=timedelta(hours=-2), now=now)
        stale_last_online = now - timedelta(days=8)

        # Should pass because attestation overrides the stale client value
        token = validate_offline(raw_token, stale_last_online, public_key, now, raw_attestation=raw_att)
        assert token.sub == "user@test.com"

    def test_attestation_grace_expired(self, rsa_keypair: tuple[str, str]) -> None:
        """GracePeriodExpired when attestation's last_online_at is too old."""
        private_key, public_key = rsa_keypair
        now = datetime.now(timezone.utc)
        raw_token = _make_token(private_key)
        raw_att = _make_attestation(
            private_key,
            last_online_offset=timedelta(days=-GRACE_PERIOD_DAYS),
            now=now,
        )
        with pytest.raises(GracePeriodExpired):
            validate_offline(raw_token, None, public_key, now, raw_attestation=raw_att)

    def test_time_rollback_detected(self, rsa_keypair: tuple[str, str]) -> None:
        """TimeRollbackDetected when system clock is before attestation timestamp."""
        private_key, public_key = rsa_keypair
        now = datetime.now(timezone.utc)
        raw_token = _make_token(private_key)
        # Attestation says 1h in the future (clock rollback: current time < last_online_at)
        raw_att = _make_attestation(private_key, last_online_offset=timedelta(hours=1), now=now)

        with pytest.raises(TimeRollbackDetected):
            validate_offline(raw_token, None, public_key, now, raw_attestation=raw_att)

    def test_start_counter_limit(self, rsa_keypair: tuple[str, str]) -> None:
        """OfflineStartLimitExceeded when start_counter >= MAX_OFFLINE_STARTS."""
        private_key, public_key = rsa_keypair
        now = datetime.now(timezone.utc)
        raw_token = _make_token(private_key)
        raw_att = _make_attestation(private_key, last_online_offset=timedelta(hours=-1), now=now)

        with pytest.raises(OfflineStartLimitExceeded) as exc_info:
            validate_offline(
                raw_token,
                None,
                public_key,
                now,
                raw_attestation=raw_att,
                start_counter=MAX_OFFLINE_STARTS,
            )
        assert exc_info.value.starts == MAX_OFFLINE_STARTS
        assert exc_info.value.limit == MAX_OFFLINE_STARTS

    def test_start_counter_below_limit_passes(self, rsa_keypair: tuple[str, str]) -> None:
        """Start counter below limit does not block access."""
        private_key, public_key = rsa_keypair
        now = datetime.now(timezone.utc)
        raw_token = _make_token(private_key)
        raw_att = _make_attestation(private_key, last_online_offset=timedelta(hours=-1), now=now)

        token = validate_offline(
            raw_token,
            None,
            public_key,
            now,
            raw_attestation=raw_att,
            start_counter=MAX_OFFLINE_STARTS - 1,
        )
        assert token.sub == "user@test.com"

    def test_attestation_machine_mismatch_raises(self, rsa_keypair: tuple[str, str]) -> None:
        """LicenseError when attestation machine_id doesn't match current machine."""
        private_key, public_key = rsa_keypair
        now = datetime.now(timezone.utc)
        raw_token = _make_token(private_key)
        raw_att = _make_attestation(
            private_key,
            machine_id="sha256:ffffffffffffffffffffffffffffffff",
            last_online_offset=timedelta(hours=-1),
            now=now,
        )
        with pytest.raises(LicenseError, match="Attestation machine ID mismatch"):
            validate_offline(raw_token, None, public_key, now, raw_attestation=raw_att)
