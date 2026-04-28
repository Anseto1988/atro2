"""Unit tests for licensing exception hierarchy."""
from __future__ import annotations
import pytest
from astroai.licensing.exceptions import (
    ActivationError,
    GracePeriodExpired,
    LicenseError,
    NotActivated,
    OfflineStartLimitExceeded,
    RefreshError,
    TierInsufficientError,
    TimeRollbackDetected,
)


class TestLicenseExceptionHierarchy:
    def test_not_activated_is_license_error(self) -> None:
        assert issubclass(NotActivated, LicenseError)

    def test_grace_period_expired_is_license_error(self) -> None:
        assert issubclass(GracePeriodExpired, LicenseError)

    def test_activation_error_is_license_error(self) -> None:
        assert issubclass(ActivationError, LicenseError)

    def test_refresh_error_is_license_error(self) -> None:
        assert issubclass(RefreshError, LicenseError)

    def test_offline_start_limit_is_license_error(self) -> None:
        assert issubclass(OfflineStartLimitExceeded, LicenseError)

    def test_time_rollback_is_license_error(self) -> None:
        assert issubclass(TimeRollbackDetected, LicenseError)

    def test_tier_insufficient_is_license_error(self) -> None:
        assert issubclass(TierInsufficientError, LicenseError)

    def test_license_error_is_exception(self) -> None:
        assert issubclass(LicenseError, Exception)


class TestGracePeriodExpired:
    def test_days_offline_stored(self) -> None:
        exc = GracePeriodExpired(days_offline=8)
        assert exc.days_offline == 8

    def test_message_contains_days(self) -> None:
        exc = GracePeriodExpired(days_offline=10)
        assert "10" in str(exc)

    def test_is_catchable_as_license_error(self) -> None:
        with pytest.raises(LicenseError):
            raise GracePeriodExpired(days_offline=7)


class TestActivationError:
    def test_code_stored(self) -> None:
        exc = ActivationError("max_seats_reached")
        assert exc.code == "max_seats_reached"

    def test_message_contains_code(self) -> None:
        exc = ActivationError("max_seats_reached")
        assert "max_seats_reached" in str(exc)

    def test_detail_appended(self) -> None:
        exc = ActivationError("invalid_key", detail="Key not found")
        assert "Key not found" in str(exc)

    def test_empty_detail_no_extra_dash(self) -> None:
        exc = ActivationError("invalid_key")
        assert "—" not in str(exc)


class TestRefreshError:
    def test_code_stored(self) -> None:
        exc = RefreshError("subscription_expired")
        assert exc.code == "subscription_expired"

    def test_message_contains_code(self) -> None:
        exc = RefreshError("license_revoked", detail="Revoked by admin")
        assert "license_revoked" in str(exc)
        assert "Revoked by admin" in str(exc)


class TestOfflineStartLimitExceeded:
    def test_fields(self) -> None:
        exc = OfflineStartLimitExceeded(starts=11, limit=10)
        assert exc.starts == 11
        assert exc.limit == 10

    def test_message(self) -> None:
        exc = OfflineStartLimitExceeded(starts=11, limit=10)
        assert "11" in str(exc)
        assert "10" in str(exc)


class TestTimeRollbackDetected:
    def test_delta_stored(self) -> None:
        exc = TimeRollbackDetected(delta_seconds=3600.0)
        assert exc.delta_seconds == pytest.approx(3600.0)

    def test_message(self) -> None:
        exc = TimeRollbackDetected(delta_seconds=120.0)
        assert "120" in str(exc)


class TestTierInsufficientError:
    def test_fields(self) -> None:
        exc = TierInsufficientError(model_name="starnet_pro", required_tier="pro_annual")
        assert exc.model_name == "starnet_pro"
        assert exc.required_tier == "pro_annual"

    def test_message(self) -> None:
        exc = TierInsufficientError(model_name="denoise_pro", required_tier="founding_member")
        assert "denoise_pro" in str(exc)
        assert "founding_member" in str(exc)
