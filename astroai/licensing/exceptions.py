"""License-related exceptions."""

from __future__ import annotations


class LicenseError(Exception):
    """Base exception for all licensing errors."""


class NotActivated(LicenseError):
    """Raised when no valid license is found on this machine."""


class GracePeriodExpired(LicenseError):
    """Raised when offline grace period (7 days) has elapsed."""

    def __init__(self, days_offline: int) -> None:
        self.days_offline = days_offline
        super().__init__(
            f"Offline grace period expired ({days_offline} days without server contact)"
        )


class ActivationError(LicenseError):
    """Raised when the activation API call fails."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"Activation failed: {code}" + (f" — {detail}" if detail else ""))


class RefreshError(LicenseError):
    """Raised when token refresh fails (subscription expired, revoked, etc.)."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"Refresh failed: {code}" + (f" — {detail}" if detail else ""))


class OfflineStartLimitExceeded(LicenseError):
    """Raised when the offline app-start counter exceeds the maximum allowed."""

    def __init__(self, starts: int, limit: int) -> None:
        self.starts = starts
        self.limit = limit
        super().__init__(f"Offline start limit reached ({starts}/{limit}) — please connect to internet")


class TimeRollbackDetected(LicenseError):
    """Raised when system clock is behind the server-attested last_online_at."""

    def __init__(self, delta_seconds: float) -> None:
        self.delta_seconds = delta_seconds
        super().__init__(
            f"System clock rollback detected ({delta_seconds:.0f}s behind last server sync)"
        )


class TierInsufficientError(LicenseError):
    """Raised when the user's tier does not grant access to a model."""

    def __init__(self, model_name: str, required_tier: str) -> None:
        self.model_name = model_name
        self.required_tier = required_tier
        super().__init__(
            f"Tier insufficient for model '{model_name}': requires '{required_tier}'"
        )
