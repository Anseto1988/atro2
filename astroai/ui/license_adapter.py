"""Qt adapter for LicenseManager — adds signals and background threading."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal, Slot

from astroai import __version__
from astroai.licensing import LicenseManager
from astroai.licensing.exceptions import GracePeriodExpired, LicenseError, NotActivated
from astroai.licensing.models import LicenseStatus, LicenseTier

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["QLicenseAdapter"]

_TIER_LABELS = {
    LicenseTier.FREE: "Free",
    LicenseTier.PRO_MONTHLY: "Pro",
    LicenseTier.PRO_ANNUAL: "Pro",
    LicenseTier.FOUNDING_MEMBER: "Founding",
}


class _ActivateWorker(QObject):
    succeeded = Signal(object)  # LicenseStatus
    failed = Signal(str)

    def __init__(self, manager: LicenseManager, key: str) -> None:
        super().__init__()
        self._mgr = manager
        self._key = key

    @Slot()
    def run(self) -> None:
        try:
            self._mgr.activate(self._key, __version__)
            status = self._mgr.get_status()
            self.succeeded.emit(status)
        except LicenseError as exc:
            self.failed.emit(str(exc))


class QLicenseAdapter(QObject):
    """Reactive Qt wrapper around the pure-Python LicenseManager."""

    status_changed = Signal(object)  # LicenseStatus
    activation_started = Signal()
    activation_succeeded = Signal(object)  # LicenseStatus
    activation_failed = Signal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        store_dir: Path | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(parent)
        kwargs: dict = {}
        if store_dir is not None:
            kwargs["store_dir"] = store_dir
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._mgr = LicenseManager(**kwargs)
        self._status = LicenseStatus()
        self._thread: QThread | None = None
        self._worker: _ActivateWorker | None = None

    @property
    def status(self) -> LicenseStatus:
        return self._status

    @property
    def tier(self) -> LicenseTier:
        if self._status.token is None:
            return LicenseTier.FREE
        return self._status.token.tier

    @property
    def tier_label(self) -> str:
        return _TIER_LABELS.get(self.tier, "Free")

    @property
    def is_activated(self) -> bool:
        return self._status.activated

    @property
    def grace_remaining_days(self) -> int:
        return self._status.grace_remaining_days

    def verify(self) -> LicenseStatus:
        """Synchronous verify — call once at startup."""
        try:
            self._status = self._mgr.verify()
        except GracePeriodExpired:
            self._status = self._mgr.get_status()
            self._status.grace_remaining_days = 0
        except (NotActivated, LicenseError):
            self._status = LicenseStatus()
        self.status_changed.emit(self._status)
        return self._status

    def refresh_status(self) -> LicenseStatus:
        """Re-read local store without network call."""
        self._status = self._mgr.get_status()
        self.status_changed.emit(self._status)
        return self._status

    def activate_async(self, license_key: str) -> None:
        """Start activation in a background thread."""
        if self._thread is not None and self._thread.isRunning():
            return

        self.activation_started.emit()
        self._thread = QThread(self)
        self._worker = _ActivateWorker(self._mgr, license_key)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.succeeded.connect(self._on_ok)
        self._worker.failed.connect(self._on_fail)
        self._worker.succeeded.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def deactivate(self) -> None:
        try:
            self._mgr.deactivate()
        except LicenseError:
            pass
        self._status = LicenseStatus()
        self.status_changed.emit(self._status)

    def check_plugin(self, plugin: str) -> bool:
        if self._status.token is None:
            return False
        return self._status.token.has_plugin(plugin)

    @Slot(object)
    def _on_ok(self, status: LicenseStatus) -> None:
        self._status = status
        self.status_changed.emit(self._status)
        self.activation_succeeded.emit(self._status)

    @Slot(str)
    def _on_fail(self, error: str) -> None:
        self.activation_failed.emit(error)

    def _cleanup(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
