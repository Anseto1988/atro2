"""Tests for license UI widgets (LicenseBadge, ActivationDialog, OfflineBanner, UpgradeDialog)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import Qt

from astroai.licensing.models import LicenseStatus, LicenseTier, LicenseToken
from astroai.ui.widgets.activation_dialog import ActivationDialog
from astroai.ui.widgets.license_badge import LicenseBadge
from astroai.ui.widgets.offline_banner import OfflineBanner
from astroai.ui.widgets.upgrade_dialog import UpgradeDialog


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


def _make_status(
    tier: LicenseTier = LicenseTier.FREE,
    activated: bool = False,
    grace_days: int = 7,
) -> LicenseStatus:
    token = _make_token(tier) if activated else None
    return LicenseStatus(
        token=token,
        last_online_at=datetime.now(timezone.utc) if activated else None,
        activated=activated,
        grace_remaining_days=grace_days,
        allowed_plugins=list(token.plugins) if token else [],
    )


class TestLicenseBadge:
    @pytest.fixture()
    def badge(self, qtbot) -> LicenseBadge:
        w = LicenseBadge()
        qtbot.addWidget(w)
        return w

    def test_initial_state_is_free(self, badge: LicenseBadge) -> None:
        assert "Free" in badge.text()
        assert badge.objectName() == "licenseBadgeFree"

    def test_updates_to_pro(self, badge: LicenseBadge) -> None:
        status = _make_status(LicenseTier.PRO_ANNUAL, activated=True)
        badge.on_status_changed(status)
        assert "Pro" in badge.text()
        assert badge.objectName() == "licenseBadgePro"

    def test_updates_to_founding(self, badge: LicenseBadge) -> None:
        status = _make_status(LicenseTier.FOUNDING_MEMBER, activated=True)
        badge.on_status_changed(status)
        assert "Founding" in badge.text()
        assert badge.objectName() == "licenseBadgeFounding"

    def test_resets_to_free(self, badge: LicenseBadge) -> None:
        badge.on_status_changed(_make_status(LicenseTier.PRO_ANNUAL, activated=True))
        badge.on_status_changed(_make_status())
        assert "Free" in badge.text()
        assert badge.objectName() == "licenseBadgeFree"

    def test_no_update_when_same_tier(self, badge: LicenseBadge) -> None:
        badge.on_status_changed(_make_status())
        assert badge.objectName() == "licenseBadgeFree"

    def test_tooltip_shows_tier(self, badge: LicenseBadge) -> None:
        badge.on_status_changed(_make_status(LicenseTier.FOUNDING_MEMBER, activated=True))
        assert "Founding" in badge.toolTip()


class TestOfflineBanner:
    @pytest.fixture()
    def banner(self, qtbot) -> OfflineBanner:
        w = OfflineBanner()
        qtbot.addWidget(w)
        return w

    def test_hidden_initially(self, banner: OfflineBanner) -> None:
        assert not banner.isVisible()

    def test_hidden_when_not_activated(self, banner: OfflineBanner) -> None:
        banner.on_status_changed(_make_status())
        assert not banner.isVisible()

    def test_hidden_when_grace_sufficient(self, banner: OfflineBanner) -> None:
        banner.on_status_changed(_make_status(LicenseTier.PRO_ANNUAL, activated=True, grace_days=5))
        assert not banner.isVisible()

    def test_visible_when_grace_low(self, banner: OfflineBanner) -> None:
        banner.on_status_changed(_make_status(LicenseTier.PRO_ANNUAL, activated=True, grace_days=1))
        assert banner.isVisible()
        assert "verbleibend" in banner._message.text().lower() or "h" in banner._message.text()

    def test_visible_when_grace_expired(self, banner: OfflineBanner) -> None:
        banner.on_status_changed(_make_status(LicenseTier.PRO_ANNUAL, activated=True, grace_days=0))
        assert banner.isVisible()
        assert "abgelaufen" in banner._message.text().lower()

    def test_hides_again_when_online(self, banner: OfflineBanner) -> None:
        banner.on_status_changed(_make_status(LicenseTier.PRO_ANNUAL, activated=True, grace_days=0))
        assert banner.isVisible()
        banner.on_status_changed(_make_status(LicenseTier.PRO_ANNUAL, activated=True, grace_days=7))
        assert not banner.isVisible()


class TestActivationDialog:
    @pytest.fixture()
    def adapter(self) -> MagicMock:
        mock = MagicMock()
        mock.is_activated = False
        mock.activation_started = MagicMock()
        mock.activation_succeeded = MagicMock()
        mock.activation_failed = MagicMock()
        mock.activation_started.connect = MagicMock()
        mock.activation_succeeded.connect = MagicMock()
        mock.activation_failed.connect = MagicMock()
        return mock

    @pytest.fixture()
    def dialog(self, qtbot, adapter) -> ActivationDialog:
        dlg = ActivationDialog(adapter)
        qtbot.addWidget(dlg)
        return dlg

    def test_activate_button_disabled_initially(self, dialog: ActivationDialog) -> None:
        assert not dialog._activate_btn.isEnabled()

    def test_activate_button_enables_with_key(self, dialog: ActivationDialog) -> None:
        dialog._key_input.setText("ASTRO-1234-5678-ABCD")
        assert dialog._activate_btn.isEnabled()

    def test_activate_button_disabled_for_short_key(self, dialog: ActivationDialog) -> None:
        dialog._key_input.setText("ASTRO")
        assert not dialog._activate_btn.isEnabled()

    def test_deactivate_hidden_when_not_activated(self, dialog: ActivationDialog) -> None:
        assert dialog._deactivate_btn.isHidden()

    def test_deactivate_visible_when_activated(self, qtbot, adapter) -> None:
        adapter.is_activated = True
        dlg = ActivationDialog(adapter)
        qtbot.addWidget(dlg)
        assert not dlg._deactivate_btn.isHidden()

    def test_on_started_shows_progress(self, dialog: ActivationDialog) -> None:
        dialog._on_started()
        assert not dialog._progress.isHidden()
        assert not dialog._activate_btn.isEnabled()
        assert not dialog._key_input.isEnabled()

    def test_on_succeeded_shows_success(self, dialog: ActivationDialog) -> None:
        dialog._on_started()
        dialog._on_succeeded(_make_status(LicenseTier.PRO_ANNUAL, activated=True))
        assert dialog._progress.isHidden()
        assert not dialog._success_label.isHidden()
        assert "erfolgreich" in dialog._success_label.text().lower()

    def test_on_failed_shows_error(self, dialog: ActivationDialog) -> None:
        dialog._on_started()
        dialog._on_failed("Activation failed: invalid_key")
        assert dialog._progress.isHidden()
        assert not dialog._error_label.isHidden()
        assert "invalid_key" in dialog._error_label.text()

    def test_error_clears_on_new_input(self, dialog: ActivationDialog) -> None:
        dialog._on_failed("Some error")
        assert not dialog._error_label.isHidden()
        dialog._key_input.setText("ASTRO-NEW-KEY-HERE")
        assert dialog._error_label.isHidden()


class TestUpgradeDialog:
    @pytest.fixture()
    def dialog(self, qtbot) -> UpgradeDialog:
        dlg = UpgradeDialog(
            feature_name="Denoise Pro",
            required_tier=LicenseTier.PRO_ANNUAL,
            current_tier=LicenseTier.FREE,
        )
        qtbot.addWidget(dlg)
        return dlg

    def test_shows_feature_name(self, dialog: UpgradeDialog) -> None:
        body_text = dialog.findChild(type(dialog.findChildren(type(dialog))[0] if dialog.findChildren(type(dialog)) else dialog))
        found = False
        for label in dialog.findChildren(type(dialog._icon if hasattr(dialog, "_icon") else dialog)):
            pass
        from PySide6.QtWidgets import QLabel
        labels = dialog.findChildren(QLabel)
        texts = [l.text() for l in labels]
        assert any("Denoise Pro" in t for t in texts)

    def test_shows_tier_info(self, dialog: UpgradeDialog) -> None:
        from PySide6.QtWidgets import QLabel
        labels = dialog.findChildren(QLabel)
        texts = " ".join(l.text() for l in labels)
        assert "Pro" in texts
        assert "Free" in texts

    def test_upgrade_signal(self, dialog: UpgradeDialog, qtbot) -> None:
        with qtbot.waitSignal(dialog.upgrade_requested, timeout=500):
            dialog._on_upgrade()

    def test_activate_signal_for_free_user(self, dialog: UpgradeDialog, qtbot) -> None:
        with qtbot.waitSignal(dialog.activate_requested, timeout=500):
            dialog._on_activate()

    def test_no_activate_button_for_paid_user(self, qtbot) -> None:
        from PySide6.QtWidgets import QPushButton
        dlg = UpgradeDialog(
            feature_name="Test Feature",
            required_tier=LicenseTier.FOUNDING_MEMBER,
            current_tier=LicenseTier.PRO_ANNUAL,
        )
        qtbot.addWidget(dlg)
        buttons = dlg.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert "Lizenz eingeben" not in button_texts

    def test_has_activate_button_for_free_user(self, dialog: UpgradeDialog) -> None:
        from PySide6.QtWidgets import QPushButton
        buttons = dialog.findChildren(QPushButton)
        button_texts = [b.text() for b in buttons]
        assert "Lizenz eingeben" in button_texts
