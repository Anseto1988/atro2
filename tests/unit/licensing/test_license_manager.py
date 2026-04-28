"""Unit tests for LicenseManager — attestation plumbing, offline counter, status."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from astroai.licensing import LicenseManager, LicenseStatus
from astroai.licensing.exceptions import (
    GracePeriodExpired,
    LicenseError,
    NotActivated,
)
from astroai.licensing.machine import get_machine_id


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rsa_keypair() -> tuple[str, str]:
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


def _make_token(private_key: str, exp_delta: timedelta = timedelta(days=30)) -> str:
    now = int(time.time())
    return pyjwt.encode({
        "sub": "user@test.com",
        "jti": "test-jti-001",
        "iat": now,
        "exp": now + int(exp_delta.total_seconds()),
        "tier": "pro_annual",
        "plugins": ["denoise_pro"],
        "machine_id": get_machine_id(),
        "seats_used": 1,
        "seats_max": 1,
    }, private_key, algorithm="RS256")


def _make_attestation(
    private_key: str,
    offset_hours: float = 0.0,
    machine_id: str | None = None,
) -> str:
    now = int(time.time()) + int(offset_hours * 3600)
    return pyjwt.encode({
        "user_id": "user@test.com",
        "machine_id": machine_id or get_machine_id(),
        "last_online_at": now,
        "iat": now,
    }, private_key, algorithm="RS256")


@pytest.fixture()
def tmp_store(tmp_path: Path) -> Path:
    return tmp_path / "license"


# ---------------------------------------------------------------------------
# Activate: attestation must be persisted
# ---------------------------------------------------------------------------

class TestActivateAttestationPlumbing:
    def test_activate_stores_attestation_when_server_provides_it(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        raw_jwt = _make_token(private_key)
        raw_att = _make_attestation(private_key)

        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        mgr._client.activate = MagicMock(return_value=(raw_jwt, raw_att))  # type: ignore[method-assign]

        mgr.activate("KEY-123", "2.1.0")

        # Load store directly and verify attestation was persisted
        stored = mgr._store.load()
        assert stored is not None
        _, _, stored_att, _ = stored
        assert stored_att == raw_att

    def test_activate_stores_none_attestation_for_legacy_server(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        raw_jwt = _make_token(private_key)

        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        mgr._client.activate = MagicMock(return_value=(raw_jwt, None))  # type: ignore[method-assign]

        mgr.activate("KEY-123", "2.1.0")

        stored = mgr._store.load()
        assert stored is not None
        _, _, stored_att, _ = stored
        assert stored_att is None

    def test_activate_returns_decoded_token(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        raw_jwt = _make_token(private_key)

        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        mgr._client.activate = MagicMock(return_value=(raw_jwt, None))  # type: ignore[method-assign]

        token = mgr.activate("KEY-XXXX", "2.1.0")
        assert token.sub == "user@test.com"
        assert token.tier.value == "pro_annual"


# ---------------------------------------------------------------------------
# Verify (online): attestation and counter reset
# ---------------------------------------------------------------------------

class TestVerifyOnlineAttestationAndCounterReset:
    def test_online_refresh_stores_new_attestation(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        raw_jwt = _make_token(private_key)
        new_jwt = _make_token(private_key)
        new_att = _make_attestation(private_key)

        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        # Seed the store with an existing token + old counter
        mgr._store.save(raw_jwt, datetime.now(timezone.utc), attestation_raw=None, start_counter=10)

        mgr._client.refresh = MagicMock(return_value=(new_jwt, new_att))  # type: ignore[method-assign]

        mgr.verify()

        stored = mgr._store.load()
        assert stored is not None
        _, _, stored_att, stored_counter = stored
        assert stored_att == new_att
        assert stored_counter == 0  # counter must be reset on successful online refresh

    def test_online_refresh_resets_start_counter_to_zero(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        raw_jwt = _make_token(private_key)
        new_jwt = _make_token(private_key)

        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        mgr._store.save(raw_jwt, datetime.now(timezone.utc), start_counter=42)

        mgr._client.refresh = MagicMock(return_value=(new_jwt, None))  # type: ignore[method-assign]
        mgr.verify()

        stored = mgr._store.load()
        assert stored is not None
        _, _, _, counter = stored
        assert counter == 0

    def test_online_refresh_returns_full_grace_period(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        raw_jwt = _make_token(private_key)
        new_jwt = _make_token(private_key)

        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        mgr._store.save(raw_jwt, datetime.now(timezone.utc) - timedelta(days=6))

        mgr._client.refresh = MagicMock(return_value=(new_jwt, None))  # type: ignore[method-assign]
        status = mgr.verify()

        assert status.grace_remaining_days == 7


# ---------------------------------------------------------------------------
# Verify (offline): attestation forwarded + counter incremented
# ---------------------------------------------------------------------------

class TestVerifyOfflineAttestationForwarding:
    def _setup_offline_mgr(
        self,
        tmp_store: Path,
        private_key: str,
        public_key: str,
        last_online_days_ago: int = 1,
        start_counter: int = 0,
        with_attestation: bool = True,
    ) -> LicenseManager:
        raw_jwt = _make_token(private_key)
        raw_att = _make_attestation(
            private_key, offset_hours=-(last_online_days_ago * 24),
        ) if with_attestation else None
        last_online = datetime.now(timezone.utc) - timedelta(days=last_online_days_ago)

        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        mgr._store.save(raw_jwt, last_online, attestation_raw=raw_att, start_counter=start_counter)
        # Make online refresh fail to force offline path
        mgr._client.refresh = MagicMock(side_effect=LicenseError("network error"))  # type: ignore[method-assign]
        return mgr

    def test_offline_path_increments_start_counter(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        mgr = self._setup_offline_mgr(tmp_store, private_key, public_key, start_counter=5)

        mgr.verify()

        stored = mgr._store.load()
        assert stored is not None
        _, _, _, counter = stored
        assert counter == 6  # incremented from 5

    def test_offline_with_attestation_uses_server_time(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        mgr = self._setup_offline_mgr(
            tmp_store, private_key, public_key,
            last_online_days_ago=2, with_attestation=True,
        )

        status = mgr.verify()
        assert status.activated is True
        assert status.grace_remaining_days <= 5  # 7 - 2

    def test_offline_without_attestation_falls_back_to_stored_time(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        mgr = self._setup_offline_mgr(
            tmp_store, private_key, public_key,
            last_online_days_ago=1, with_attestation=False,
        )

        # Should succeed since we're within grace period
        status = mgr.verify()
        assert status.activated is True

    def test_offline_grace_expired_raises(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        mgr = self._setup_offline_mgr(
            tmp_store, private_key, public_key,
            last_online_days_ago=8, with_attestation=False,
        )

        with pytest.raises(GracePeriodExpired):
            mgr.verify()


# ---------------------------------------------------------------------------
# No activation: NotActivated
# ---------------------------------------------------------------------------

class TestVerifyNoActivation:
    def test_raises_not_activated_when_store_empty(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        _, public_key = rsa_keypair
        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)

        with pytest.raises(NotActivated):
            mgr.verify()


# ---------------------------------------------------------------------------
# get_status: no network, no side effects
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_returns_empty_status_when_not_activated(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        _, public_key = rsa_keypair
        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        status = mgr.get_status()
        assert status.activated is False
        assert status.token is None

    def test_returns_status_when_activated(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        raw_jwt = _make_token(private_key)
        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        mgr._store.save(raw_jwt, datetime.now(timezone.utc))

        status = mgr.get_status()
        assert status.activated is True
        assert status.token is not None
        assert status.token.tier.value == "pro_annual"


# ---------------------------------------------------------------------------
# Deactivate
# ---------------------------------------------------------------------------

class TestDeactivate:
    def test_clears_store_after_deactivation(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        raw_jwt = _make_token(private_key)
        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        mgr._store.save(raw_jwt, datetime.now(timezone.utc))
        mgr._client.deactivate = MagicMock(return_value=1)  # type: ignore[method-assign]

        mgr.deactivate()

        assert not mgr._store.exists

    def test_deactivate_clears_store_even_when_server_fails(
        self, rsa_keypair: tuple[str, str], tmp_store: Path,
    ) -> None:
        private_key, public_key = rsa_keypair
        raw_jwt = _make_token(private_key)
        mgr = LicenseManager(store_dir=tmp_store, public_key=public_key)
        mgr._store.save(raw_jwt, datetime.now(timezone.utc))
        mgr._client.deactivate = MagicMock(side_effect=LicenseError("server down"))  # type: ignore[method-assign]

        mgr.deactivate()

        assert not mgr._store.exists
