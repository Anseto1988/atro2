"""Tests for QLicenseAdapter — verify, deactivate, check_plugin, activate_async."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from astroai.licensing.exceptions import GracePeriodExpired, LicenseError, NotActivated
from astroai.licensing.models import LicenseStatus, LicenseTier, LicenseToken
from astroai.ui.license_adapter import QLicenseAdapter, _TIER_LABELS


def _make_token(tier: LicenseTier = LicenseTier.PRO_ANNUAL) -> LicenseToken:
    now = datetime.now(timezone.utc)
    return LicenseToken(
        sub="user@test.com",
        jti="test-jti",
        iat=now,
        exp=now.replace(year=now.year + 1),
        tier=tier,
        plugins=("denoise_pro", "starnet_pro"),
        machine_id="sha256:abc",
        seats_used=1,
        seats_max=1,
    )


def _make_status(tier: LicenseTier = LicenseTier.PRO_ANNUAL, activated: bool = True) -> LicenseStatus:
    token = _make_token(tier) if activated else None
    return LicenseStatus(
        token=token,
        last_online_at=datetime.now(timezone.utc) if activated else None,
        activated=activated,
        grace_remaining_days=7,
        allowed_plugins=list(token.plugins) if token else [],
    )


@pytest.fixture()
def adapter(qtbot) -> QLicenseAdapter:
    with patch("astroai.ui.license_adapter.LicenseManager") as MockMgr:
        instance = MockMgr.return_value
        instance.verify.return_value = LicenseStatus()
        instance.get_status.return_value = LicenseStatus()
        a = QLicenseAdapter()
        a._mgr = instance
        return a


class TestQLicenseAdapterProperties:
    def test_initial_status(self, adapter: QLicenseAdapter) -> None:
        assert adapter.status is not None
        assert adapter.status.activated is False

    def test_tier_free_when_no_token(self, adapter: QLicenseAdapter) -> None:
        assert adapter.tier is LicenseTier.FREE

    def test_tier_label_free(self, adapter: QLicenseAdapter) -> None:
        assert adapter.tier_label == "Free"

    def test_is_activated_false(self, adapter: QLicenseAdapter) -> None:
        assert adapter.is_activated is False

    def test_grace_remaining_days(self, adapter: QLicenseAdapter) -> None:
        assert adapter.grace_remaining_days >= 0

    def test_tier_returns_token_tier(self, adapter: QLicenseAdapter) -> None:
        adapter._status = _make_status(LicenseTier.FOUNDING_MEMBER)
        assert adapter.tier is LicenseTier.FOUNDING_MEMBER

    def test_tier_label_pro(self, adapter: QLicenseAdapter) -> None:
        adapter._status = _make_status(LicenseTier.PRO_ANNUAL)
        assert adapter.tier_label == "Pro"


class TestQLicenseAdapterVerify:
    def test_verify_success(self, adapter: QLicenseAdapter, qtbot) -> None:
        status = _make_status()
        adapter._mgr.verify.return_value = status
        with qtbot.waitSignal(adapter.status_changed, timeout=500):
            result = adapter.verify()
        assert result.activated is True

    def test_verify_grace_period_expired(self, adapter: QLicenseAdapter, qtbot) -> None:
        adapter._mgr.verify.side_effect = GracePeriodExpired("expired")
        expired_status = _make_status()
        expired_status.grace_remaining_days = 5
        adapter._mgr.get_status.return_value = expired_status
        with qtbot.waitSignal(adapter.status_changed, timeout=500):
            result = adapter.verify()
        assert result.grace_remaining_days == 0

    def test_verify_not_activated(self, adapter: QLicenseAdapter, qtbot) -> None:
        adapter._mgr.verify.side_effect = NotActivated("not activated")
        with qtbot.waitSignal(adapter.status_changed, timeout=500):
            result = adapter.verify()
        assert result.activated is False

    def test_verify_license_error(self, adapter: QLicenseAdapter, qtbot) -> None:
        adapter._mgr.verify.side_effect = LicenseError("network error")
        with qtbot.waitSignal(adapter.status_changed, timeout=500):
            result = adapter.verify()
        assert result.activated is False


class TestQLicenseAdapterRefresh:
    def test_refresh_status(self, adapter: QLicenseAdapter, qtbot) -> None:
        status = _make_status()
        adapter._mgr.get_status.return_value = status
        with qtbot.waitSignal(adapter.status_changed, timeout=500):
            result = adapter.refresh_status()
        assert result.activated is True


class TestQLicenseAdapterDeactivate:
    def test_deactivate_success(self, adapter: QLicenseAdapter, qtbot) -> None:
        adapter._status = _make_status()
        with qtbot.waitSignal(adapter.status_changed, timeout=500):
            adapter.deactivate()
        assert adapter.status.activated is False
        adapter._mgr.deactivate.assert_called_once()

    def test_deactivate_handles_error(self, adapter: QLicenseAdapter, qtbot) -> None:
        adapter._mgr.deactivate.side_effect = LicenseError("fail")
        adapter._status = _make_status()
        with qtbot.waitSignal(adapter.status_changed, timeout=500):
            adapter.deactivate()
        assert adapter.status.activated is False


class TestQLicenseAdapterCheckPlugin:
    def test_check_plugin_no_token(self, adapter: QLicenseAdapter) -> None:
        assert adapter.check_plugin("denoise_pro") is False

    def test_check_plugin_with_token(self, adapter: QLicenseAdapter) -> None:
        adapter._status = _make_status()
        assert adapter.check_plugin("denoise_pro") is True

    def test_check_plugin_missing(self, adapter: QLicenseAdapter) -> None:
        adapter._status = _make_status()
        assert adapter.check_plugin("nonexistent") is False


class TestQLicenseAdapterActivateAsync:
    def test_activate_async_emits_started(self, adapter: QLicenseAdapter, qtbot) -> None:
        with qtbot.waitSignal(adapter.activation_started, timeout=500):
            adapter.activate_async("ASTRO-KEY-1234")
        if adapter._thread:
            adapter._thread.quit()
            adapter._thread.wait(1000)

    def test_activate_async_skips_if_running(self, adapter: QLicenseAdapter) -> None:
        adapter._thread = MagicMock()
        adapter._thread.isRunning.return_value = True
        adapter.activate_async("KEY")
        adapter._thread.start.assert_not_called()

    def test_on_ok_updates_status(self, adapter: QLicenseAdapter, qtbot) -> None:
        status = _make_status()
        with qtbot.waitSignal(adapter.activation_succeeded, timeout=500):
            adapter._on_ok(status)
        assert adapter._status == status

    def test_on_fail_emits_error(self, adapter: QLicenseAdapter, qtbot) -> None:
        with qtbot.waitSignal(adapter.activation_failed, timeout=500):
            adapter._on_fail("invalid key")

    def test_cleanup(self, adapter: QLicenseAdapter) -> None:
        adapter._thread = MagicMock()
        adapter._worker = MagicMock()
        adapter._cleanup()
        assert adapter._thread is None
        assert adapter._worker is None
