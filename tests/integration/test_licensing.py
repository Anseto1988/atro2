"""Integration tests for the AstroAI plugin licensing backend.

Tests Offline-Grace-Period, Seat-Limits, and Tier-Enforcement as specified
in VER-164 architecture plan and VER-172 test scenarios.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from astroai.licensing import LicenseManager
from astroai.licensing.client import LicenseClient
from astroai.licensing.exceptions import (
    ActivationError,
    GracePeriodExpired,
    LicenseError,
    NotActivated,
    RefreshError,
)
from astroai.licensing.machine import get_machine_id, verify_machine_id
from astroai.licensing.models import LicenseStatus, LicenseTier, LicenseToken
from astroai.licensing.store import LicenseStore
from astroai.licensing.validator import GRACE_PERIOD_DAYS, decode_token, validate_offline


# --- RSA key pair fixture for JWT signing in tests ---


@pytest.fixture(scope="module")
def rsa_keys() -> tuple[str, str]:
    """Generate an RSA key pair for test JWT signing/verification."""
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


# --- JWT helper ---


_JTI_COUNTER = 0


def _sign_jwt(
    private_key: str,
    tier: str = "pro_annual",
    plugins: list[str] | None = None,
    machine_id: str | None = None,
    seats_used: int = 1,
    seats_max: int = 1,
    exp_delta_days: int = 30,
    jti_suffix: str = "",
) -> str:
    global _JTI_COUNTER
    _JTI_COUNTER += 1
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": "user@example.com",
        "jti": f"test-jti-{now}-{_JTI_COUNTER}{jti_suffix}",
        "iat": now,
        "exp": now + (exp_delta_days * 86400),
        "tier": tier,
        "plugins": plugins if plugins is not None else ["denoise_pro", "starnet_pro", "stretch_ai"],
        "machine_id": machine_id or get_machine_id(),
        "seats_used": seats_used,
        "seats_max": seats_max,
    }
    return pyjwt.encode(payload, private_key, algorithm="RS256")


def _make_manager_with_mocks(
    tmp_path: Path,
    public_key: str,
    store_data: tuple[str, datetime] | None = None,
    client_mock: MagicMock | None = None,
) -> LicenseManager:
    """Create a LicenseManager with patched internal store and client."""
    manager = LicenseManager(store_dir=tmp_path, public_key=public_key)
    if store_data is not None:
        manager._store.save(store_data[0], store_data[1])
    if client_mock is not None:
        manager._client = client_mock
    return manager


# =============================================================================
# Offline Grace Period Tests
# =============================================================================


class TestOfflineGracePeriod:
    """Tests for the 7-day offline grace period logic.

    Spec (VER-164 §4):
    - Online activation → network loss → app start within 1 day → must work
    - Network disconnected > 7 days → AI features locked, warning displayed
    - Network restored → refresh successful → grace period resets
    """

    def test_online_verify_succeeds_and_resets_grace(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Successful online refresh returns ACTIVE status with full grace."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)
        refreshed_jwt = _sign_jwt(priv, jti_suffix="-refreshed")
        now = datetime.now(timezone.utc)

        client = MagicMock(spec=LicenseClient)
        client.refresh.return_value = refreshed_jwt

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, now), client_mock=client
        )
        status = manager.verify()

        assert status.activated is True
        assert status.grace_remaining_days == GRACE_PERIOD_DAYS
        assert len(status.allowed_plugins) > 0
        client.refresh.assert_called_once()

    def test_offline_within_grace_period_allows_features(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """App works normally when offline for less than 7 days."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)
        last_online = datetime.now(timezone.utc) - timedelta(days=1)

        client = MagicMock(spec=LicenseClient)
        client.refresh.side_effect = LicenseError("Network error")

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, last_online), client_mock=client
        )
        status = manager.verify()

        assert status.activated is True
        assert status.grace_remaining_days == GRACE_PERIOD_DAYS - 1
        assert len(status.allowed_plugins) > 0

    def test_offline_at_boundary_6_days_23h_still_allowed(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Grace period boundary: 6 days 23 hours is still within grace."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)
        last_online = datetime.now(timezone.utc) - timedelta(days=6, hours=23)

        client = MagicMock(spec=LicenseClient)
        client.refresh.side_effect = LicenseError("Network error")

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, last_online), client_mock=client
        )
        status = manager.verify()

        assert status.activated is True
        assert status.grace_remaining_days > 0

    def test_offline_beyond_grace_period_locks_features(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """AI features must be locked after 7+ days offline."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)
        last_online = datetime.now(timezone.utc) - timedelta(days=8)

        client = MagicMock(spec=LicenseClient)
        client.refresh.side_effect = LicenseError("Network error")

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, last_online), client_mock=client
        )

        with pytest.raises(GracePeriodExpired) as exc_info:
            manager.verify()
        assert exc_info.value.days_offline >= 8

    def test_offline_exactly_7_days_locks_features(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Boundary: exactly 7 days offline triggers grace period expiry."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)
        last_online = datetime.now(timezone.utc) - timedelta(days=7)

        client = MagicMock(spec=LicenseClient)
        client.refresh.side_effect = LicenseError("Network error")

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, last_online), client_mock=client
        )

        with pytest.raises(GracePeriodExpired) as exc_info:
            manager.verify()
        assert exc_info.value.days_offline >= GRACE_PERIOD_DAYS

    def test_network_restored_resets_grace_period(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """After reconnecting, successful refresh resets grace period."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)
        last_online = datetime.now(timezone.utc) - timedelta(days=5)
        refreshed_jwt = _sign_jwt(priv, jti_suffix="-restored")

        client = MagicMock(spec=LicenseClient)
        client.refresh.return_value = refreshed_jwt

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, last_online), client_mock=client
        )
        status = manager.verify()

        assert status.activated is True
        assert status.grace_remaining_days == GRACE_PERIOD_DAYS

    def test_no_stored_token_raises_not_activated(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """App without any stored token raises NotActivated."""
        _, pub = rsa_keys
        client = MagicMock(spec=LicenseClient)
        manager = LicenseManager(store_dir=tmp_path, public_key=pub)
        manager._client = client

        with pytest.raises(NotActivated):
            manager.verify()

    def test_expired_jwt_offline_raises_license_error(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """JWT already expired + offline → LicenseError (token expired)."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv, exp_delta_days=-1)
        last_online = datetime.now(timezone.utc) - timedelta(days=2)

        client = MagicMock(spec=LicenseClient)
        client.refresh.side_effect = LicenseError("Network error")

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, last_online), client_mock=client
        )

        with pytest.raises(LicenseError):
            manager.verify()


# =============================================================================
# Offline Grace Period — Direct validator tests
# =============================================================================


class TestValidateOfflineDirect:
    """Direct tests for validate_offline() function."""

    def test_validate_offline_within_grace(self, rsa_keys: tuple[str, str]) -> None:
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)
        last_online = datetime.now(timezone.utc) - timedelta(days=3)

        with patch("astroai.licensing.validator.verify_machine_id", return_value=True):
            token = validate_offline(raw_jwt, last_online, pub)
        assert token.tier == LicenseTier.PRO_ANNUAL

    def test_validate_offline_expired_raises(self, rsa_keys: tuple[str, str]) -> None:
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)
        last_online = datetime.now(timezone.utc) - timedelta(days=10)

        with patch("astroai.licensing.validator.verify_machine_id", return_value=True):
            with pytest.raises(GracePeriodExpired):
                validate_offline(raw_jwt, last_online, pub)

    def test_validate_offline_no_last_online_raises(self, rsa_keys: tuple[str, str]) -> None:
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)

        with pytest.raises(NotActivated):
            validate_offline(raw_jwt, None, pub)

    def test_validate_offline_machine_mismatch_raises(self, rsa_keys: tuple[str, str]) -> None:
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv, machine_id="sha256:wrongmachine0000000000000000")
        last_online = datetime.now(timezone.utc) - timedelta(days=1)

        with pytest.raises(LicenseError, match="Machine ID mismatch"):
            validate_offline(raw_jwt, last_online, pub)


# =============================================================================
# Seat Limits Tests
# =============================================================================


class TestSeatLimits:
    """Tests for per-tier seat limits.

    Spec (VER-164 §2, §5):
    - pro_monthly/pro_annual: 1 seat → 2nd activation rejected (409 max_seats_reached)
    - founding_member: 2 seats → 3rd activation rejected, 1st/2nd work
    - Deactivation frees a seat for reuse
    """

    def test_pro_tier_single_seat_activation_succeeds(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Pro tier: first activation on a machine succeeds."""
        priv, pub = rsa_keys
        jwt_token = _sign_jwt(priv, tier="pro_annual", seats_used=1, seats_max=1)

        client = MagicMock(spec=LicenseClient)
        client.activate.return_value = jwt_token

        manager = LicenseManager(store_dir=tmp_path, public_key=pub)
        manager._client = client

        token = manager.activate("ASTRO-TEST-KEY1-0001", "0.2.0")

        assert token.tier == LicenseTier.PRO_ANNUAL
        assert token.seats_used == 1
        assert token.seats_max == 1
        client.activate.assert_called_once_with("ASTRO-TEST-KEY1-0001", "0.2.0")

    def test_pro_tier_second_activation_rejected(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Pro tier (1 seat): 2nd activation on different machine is rejected."""
        _, pub = rsa_keys

        client = MagicMock(spec=LicenseClient)
        client.activate.side_effect = ActivationError("max_seats_reached")

        manager = LicenseManager(store_dir=tmp_path, public_key=pub)
        manager._client = client

        with pytest.raises(ActivationError, match="max_seats_reached"):
            manager.activate("ASTRO-TEST-KEY1-0001", "0.2.0")

    def test_founding_member_first_activation_succeeds(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Founding member (2 seats): 1st activation succeeds."""
        priv, pub = rsa_keys
        jwt_token = _sign_jwt(priv, tier="founding_member", seats_used=1, seats_max=2)

        client = MagicMock(spec=LicenseClient)
        client.activate.return_value = jwt_token

        manager = LicenseManager(store_dir=tmp_path, public_key=pub)
        manager._client = client

        token = manager.activate("ASTRO-FOUND-KEY1-0001", "0.2.0")
        assert token.tier == LicenseTier.FOUNDING_MEMBER
        assert token.seats_used == 1
        assert token.seats_max == 2

    def test_founding_member_second_activation_succeeds(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Founding member (2 seats): 2nd activation succeeds."""
        priv, pub = rsa_keys
        jwt_token = _sign_jwt(priv, tier="founding_member", seats_used=2, seats_max=2)

        client = MagicMock(spec=LicenseClient)
        client.activate.return_value = jwt_token

        manager = LicenseManager(store_dir=tmp_path / "machine_b", public_key=pub)
        manager._client = client

        token = manager.activate("ASTRO-FOUND-KEY1-0001", "0.2.0")
        assert token.seats_used == 2
        assert token.seats_max == 2

    def test_founding_member_third_activation_rejected(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Founding member (2 seats): 3rd activation is rejected."""
        _, pub = rsa_keys

        client = MagicMock(spec=LicenseClient)
        client.activate.side_effect = ActivationError("max_seats_reached")

        manager = LicenseManager(store_dir=tmp_path, public_key=pub)
        manager._client = client

        with pytest.raises(ActivationError, match="max_seats_reached"):
            manager.activate("ASTRO-FOUND-KEY1-0001", "0.2.0")

    def test_deactivation_clears_local_store(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Deactivating clears local store and calls server."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)
        now = datetime.now(timezone.utc)

        client = MagicMock(spec=LicenseClient)
        client.deactivate.return_value = 1

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, now), client_mock=client
        )
        manager.deactivate()

        client.deactivate.assert_called_once()
        assert manager._store.load() is None

    def test_deactivation_then_reactivation_works(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """After deactivation, re-activation on a new machine works."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv)
        now = datetime.now(timezone.utc)

        client = MagicMock(spec=LicenseClient)
        client.deactivate.return_value = 1

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, now), client_mock=client
        )
        manager.deactivate()
        assert manager._store.load() is None

        new_jwt = _sign_jwt(priv, seats_used=1, seats_max=1)
        client.activate.return_value = new_jwt
        token = manager.activate("ASTRO-TEST-KEY1-0001", "0.2.0")
        assert token.tier == LicenseTier.PRO_ANNUAL

    def test_free_tier_single_seat(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Free tier: activation succeeds but with empty plugin list."""
        priv, pub = rsa_keys
        jwt_token = _sign_jwt(priv, tier="free", plugins=[], seats_used=1, seats_max=1)

        client = MagicMock(spec=LicenseClient)
        client.activate.return_value = jwt_token

        manager = LicenseManager(store_dir=tmp_path, public_key=pub)
        manager._client = client

        token = manager.activate("ASTRO-FREE-KEY1-0001", "0.2.0")
        assert token.tier == LicenseTier.FREE
        assert len(token.plugins) == 0


# =============================================================================
# Tier Enforcement Tests
# =============================================================================


class TestTierEnforcement:
    """Tests for tier-based feature gating.

    Spec (VER-164 §2, §5):
    - Free tier cannot load pro plugins
    - Pro tier can load all current plugins
    - Expired subscription → pro features locked after next refresh
    """

    def test_free_tier_cannot_load_pro_plugins(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Free tier token has no plugins → pro plugin access denied."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv, tier="free", plugins=[])
        refreshed_jwt = _sign_jwt(priv, tier="free", plugins=[], jti_suffix="-free-ref")
        now = datetime.now(timezone.utc)

        client = MagicMock(spec=LicenseClient)
        client.refresh.return_value = refreshed_jwt

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, now), client_mock=client
        )
        status = manager.verify()

        assert "denoise_pro" not in status.allowed_plugins
        assert "starnet_pro" not in status.allowed_plugins
        assert "stretch_ai" not in status.allowed_plugins

    def test_pro_tier_can_load_all_current_plugins(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Pro tier has full plugin access."""
        priv, pub = rsa_keys
        pro_plugins = ["denoise_pro", "starnet_pro", "stretch_ai"]
        raw_jwt = _sign_jwt(priv, tier="pro_annual", plugins=pro_plugins)
        refreshed_jwt = _sign_jwt(priv, tier="pro_annual", plugins=pro_plugins, jti_suffix="-pro-ref")
        now = datetime.now(timezone.utc)

        client = MagicMock(spec=LicenseClient)
        client.refresh.return_value = refreshed_jwt

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, now), client_mock=client
        )
        status = manager.verify()

        for plugin in pro_plugins:
            assert plugin in status.allowed_plugins, f"Pro tier should access {plugin}"

    def test_founding_member_has_all_plugins(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Founding member has lifetime access to all plugins."""
        priv, pub = rsa_keys
        all_plugins = ["denoise_pro", "starnet_pro", "stretch_ai"]
        raw_jwt = _sign_jwt(priv, tier="founding_member", plugins=all_plugins, seats_max=2)
        refreshed_jwt = _sign_jwt(
            priv, tier="founding_member", plugins=all_plugins, seats_max=2, jti_suffix="-fm-ref"
        )
        now = datetime.now(timezone.utc)

        client = MagicMock(spec=LicenseClient)
        client.refresh.return_value = refreshed_jwt

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, now), client_mock=client
        )
        status = manager.verify()

        for plugin in all_plugins:
            assert plugin in status.allowed_plugins

    def test_expired_subscription_falls_through_to_offline(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """FINDING: RefreshError('subscription_expired') is caught by verify() and
        falls through to offline grace period instead of blocking access immediately.
        Per spec (VER-164 §5): expired subscription should lock pro features after
        next refresh. Current behavior: user keeps old token's plugins via grace period.
        """
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv, tier="pro_annual")
        now = datetime.now(timezone.utc)

        client = MagicMock(spec=LicenseClient)
        client.refresh.side_effect = RefreshError("subscription_expired")

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, now), client_mock=client
        )
        status = manager.verify()
        assert status.activated is True
        assert "denoise_pro" in status.allowed_plugins

    def test_revoked_license_falls_through_to_offline(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """FINDING: RefreshError('license_revoked') is caught by verify() and
        falls through to offline grace period. Per spec: revoked licenses should
        immediately block all access. Current behavior: user keeps plugins via
        offline grace period until it expires (up to 7 days).
        """
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv, tier="pro_annual")
        now = datetime.now(timezone.utc)

        client = MagicMock(spec=LicenseClient)
        client.refresh.side_effect = RefreshError("license_revoked")

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, now), client_mock=client
        )
        status = manager.verify()
        assert status.activated is True

    def test_unknown_plugin_not_in_allowed_list(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """Plugin not in token's list is not in allowed_plugins."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv, plugins=["denoise_pro"])
        refreshed_jwt = _sign_jwt(priv, plugins=["denoise_pro"], jti_suffix="-unk-ref")
        now = datetime.now(timezone.utc)

        client = MagicMock(spec=LicenseClient)
        client.refresh.return_value = refreshed_jwt

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(raw_jwt, now), client_mock=client
        )
        status = manager.verify()

        assert "denoise_pro" in status.allowed_plugins
        assert "nonexistent_plugin" not in status.allowed_plugins

    def test_tier_downgrade_removes_plugin_access(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        """When refresh returns a downgraded tier, plugins are removed."""
        priv, pub = rsa_keys
        original_jwt = _sign_jwt(priv, tier="pro_annual", plugins=["denoise_pro", "starnet_pro"])
        downgraded_jwt = _sign_jwt(priv, tier="free", plugins=[], jti_suffix="-downgraded")
        now = datetime.now(timezone.utc)

        client = MagicMock(spec=LicenseClient)
        client.refresh.return_value = downgraded_jwt

        manager = _make_manager_with_mocks(
            tmp_path, pub, store_data=(original_jwt, now), client_mock=client
        )
        status = manager.verify()

        assert "denoise_pro" not in status.allowed_plugins
        assert "starnet_pro" not in status.allowed_plugins

    def test_token_has_plugin_method(self, rsa_keys: tuple[str, str]) -> None:
        """LicenseToken.has_plugin() correctly checks plugin membership."""
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv, plugins=["denoise_pro", "starnet_pro"])
        token = decode_token(raw_jwt, pub)

        assert token.has_plugin("denoise_pro") is True
        assert token.has_plugin("starnet_pro") is True
        assert token.has_plugin("stretch_ai") is False
        assert token.has_plugin("nonexistent") is False


# =============================================================================
# Machine ID Tests
# =============================================================================


class TestMachineId:
    """Machine fingerprint must be deterministic and not contain PII."""

    def test_machine_id_is_deterministic(self) -> None:
        id1 = get_machine_id()
        id2 = get_machine_id()
        assert id1 == id2

    def test_machine_id_format(self) -> None:
        mid = get_machine_id()
        assert mid.startswith("sha256:")
        hex_part = mid[7:]
        assert len(hex_part) == 32
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_machine_id_no_plaintext_hostname(self) -> None:
        import platform
        mid = get_machine_id()
        assert platform.node().lower() not in mid.lower()

    def test_verify_machine_id_matches_current(self) -> None:
        assert verify_machine_id(get_machine_id()) is True

    def test_verify_machine_id_rejects_wrong(self) -> None:
        assert verify_machine_id("sha256:0000000000000000000000000000dead") is False


# =============================================================================
# License Store Tests
# =============================================================================


class TestLicenseStore:
    """Fernet-encrypted token persistence."""

    def test_store_roundtrip(self, tmp_path: Path, rsa_keys: tuple[str, str]) -> None:
        priv, _ = rsa_keys
        raw_jwt = _sign_jwt(priv)
        now = datetime.now(timezone.utc)

        store = LicenseStore(base_dir=tmp_path)
        store.save(raw_jwt, now)
        loaded = store.load()

        assert loaded is not None
        loaded_jwt, loaded_time = loaded
        assert loaded_jwt == raw_jwt
        assert abs((loaded_time - now).total_seconds()) < 2

    def test_store_empty_returns_none(self, tmp_path: Path) -> None:
        store = LicenseStore(base_dir=tmp_path / "empty_store")
        assert store.load() is None

    def test_store_clear_removes_data(self, tmp_path: Path, rsa_keys: tuple[str, str]) -> None:
        priv, _ = rsa_keys
        raw_jwt = _sign_jwt(priv)
        now = datetime.now(timezone.utc)

        store = LicenseStore(base_dir=tmp_path)
        store.save(raw_jwt, now)
        store.clear()
        assert store.load() is None

    def test_store_exists_property(self, tmp_path: Path, rsa_keys: tuple[str, str]) -> None:
        priv, _ = rsa_keys
        store = LicenseStore(base_dir=tmp_path)
        assert store.exists is False

        store.save(_sign_jwt(priv), datetime.now(timezone.utc))
        assert store.exists is True

        store.clear()
        assert store.exists is False


# =============================================================================
# LicenseStatus / get_status Tests
# =============================================================================


class TestGetStatus:
    """LicenseManager.get_status() without triggering refresh."""

    def test_get_status_no_license(self, tmp_path: Path, rsa_keys: tuple[str, str]) -> None:
        _, pub = rsa_keys
        manager = LicenseManager(store_dir=tmp_path, public_key=pub)
        status = manager.get_status()
        assert status.activated is False
        assert status.token is None

    def test_get_status_with_valid_license(
        self, tmp_path: Path, rsa_keys: tuple[str, str]
    ) -> None:
        priv, pub = rsa_keys
        raw_jwt = _sign_jwt(priv, tier="pro_annual")
        now = datetime.now(timezone.utc)

        manager = LicenseManager(store_dir=tmp_path, public_key=pub)
        manager._store.save(raw_jwt, now)

        status = manager.get_status()
        assert status.activated is True
        assert status.token is not None
        assert status.token.tier == LicenseTier.PRO_ANNUAL
        assert status.grace_remaining_days == GRACE_PERIOD_DAYS
