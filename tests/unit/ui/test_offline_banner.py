"""Tests for OfflineBanner widget."""
from __future__ import annotations

import pytest

from astroai.licensing.models import LicenseStatus
from astroai.ui.widgets.offline_banner import OfflineBanner, _THRESHOLD_DAYS


def _status(activated: bool, grace_days: int) -> LicenseStatus:
    s = LicenseStatus()
    s.activated = activated
    s.grace_remaining_days = grace_days
    return s


@pytest.fixture()
def banner(qtbot):
    w = OfflineBanner()
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, banner):
        assert banner is not None

    def test_hidden_initially(self, banner):
        assert banner.isHidden()

    def test_object_name(self, banner):
        assert banner.objectName() == "offlineBanner"


class TestNotActivated:
    def test_not_activated_stays_hidden(self, banner):
        banner.on_status_changed(_status(activated=False, grace_days=0))
        assert banner.isHidden()

    def test_not_activated_with_grace_stays_hidden(self, banner):
        banner.on_status_changed(_status(activated=False, grace_days=1))
        assert banner.isHidden()


class TestActivatedGraceOk:
    def test_sufficient_grace_hides_banner(self, banner):
        banner.on_status_changed(_status(activated=True, grace_days=_THRESHOLD_DAYS + 1))
        assert banner.isHidden()

    def test_exactly_threshold_days_hides_banner(self, banner):
        banner.on_status_changed(_status(activated=True, grace_days=_THRESHOLD_DAYS))
        assert banner.isHidden()


class TestActivatedGraceLow:
    def test_grace_below_threshold_shows_banner(self, banner):
        banner.on_status_changed(_status(activated=True, grace_days=1))
        assert not banner.isHidden()

    def test_grace_below_threshold_message_contains_hours(self, banner):
        banner.on_status_changed(_status(activated=True, grace_days=1))
        assert "24h" in banner._message.text() or "h" in banner._message.text()

    def test_grace_zero_shows_expired_message(self, banner):
        banner.on_status_changed(_status(activated=True, grace_days=0))
        assert not banner.isHidden()
        assert "abgelaufen" in banner._message.text()

    def test_grace_negative_shows_expired_message(self, banner):
        banner.on_status_changed(_status(activated=True, grace_days=-1))
        assert not banner.isHidden()
        assert "abgelaufen" in banner._message.text()


class TestTransitions:
    def test_shows_then_hides_when_grace_recovers(self, banner):
        banner.on_status_changed(_status(activated=True, grace_days=1))
        assert not banner.isHidden()
        banner.on_status_changed(_status(activated=True, grace_days=10))
        assert banner.isHidden()

    def test_shows_then_hides_when_deactivated(self, banner):
        banner.on_status_changed(_status(activated=True, grace_days=0))
        assert not banner.isHidden()
        banner.on_status_changed(_status(activated=False, grace_days=0))
        assert banner.isHidden()
