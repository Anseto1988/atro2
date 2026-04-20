"""HTTP client for the AstroAI license activation server."""

from __future__ import annotations

from typing import Any

import httpx

from astroai.licensing.exceptions import (
    ActivationError,
    LicenseError,
    RefreshError,
    TierInsufficientError,
)
from astroai.licensing.machine import get_machine_id


_DEFAULT_BASE_URL = "https://api.astroai.app"
_TIMEOUT = 5.0
_DOWNLOAD_TIMEOUT = 10.0


class LicenseClient:
    """Synchronous HTTP client for license server API calls."""

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, timeout: float = _TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def activate(self, license_key: str, app_version: str) -> str:
        """Activate a license key on this machine. Returns the raw JWT token."""
        payload: dict[str, str] = {
            "license_key": license_key,
            "machine_id": get_machine_id(),
            "app_version": app_version,
        }
        try:
            resp = httpx.post(
                self._url("/api/v1/license/activate"),
                json=payload,
                timeout=self._timeout,
            )
        except httpx.RequestError as e:
            raise LicenseError(f"Network error during activation: {e}") from e

        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            token: str = data["token"]
            return token

        error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        code = error_data.get("error", f"http_{resp.status_code}")
        detail = error_data.get("detail", "")
        raise ActivationError(code, detail)

    def refresh(self, current_token: str) -> str:
        """Refresh a license token. Returns the new raw JWT."""
        try:
            resp = httpx.post(
                self._url("/api/v1/license/refresh"),
                headers={"Authorization": f"Bearer {current_token}"},
                timeout=self._timeout,
            )
        except httpx.RequestError as e:
            raise LicenseError(f"Network error during refresh: {e}") from e

        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            token: str = data["token"]
            return token

        error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        code = error_data.get("error", f"http_{resp.status_code}")
        detail = error_data.get("detail", "")
        raise RefreshError(code, detail)

    def deactivate(self, current_token: str) -> int:
        """Deactivate license on this machine. Returns seats released."""
        try:
            resp = httpx.post(
                self._url("/api/v1/license/deactivate"),
                headers={"Authorization": f"Bearer {current_token}"},
                timeout=self._timeout,
            )
        except httpx.RequestError as e:
            raise LicenseError(f"Network error during deactivation: {e}") from e

        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            seats: int = data.get("seats_released", 1)
            return seats

        raise LicenseError(f"Deactivation failed with status {resp.status_code}")

    def get_model_manifest(self, token: str) -> list[dict[str, Any]]:
        """Fetch the tier-filtered model manifest from the server."""
        try:
            resp = httpx.get(
                self._url("/api/v1/models/manifest"),
                headers={"Authorization": f"Bearer {token}"},
                timeout=_DOWNLOAD_TIMEOUT,
            )
        except httpx.RequestError as e:
            raise LicenseError(f"Network error fetching manifest: {e}") from e

        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            models: list[dict[str, Any]] = data.get("models", [])
            return models

        raise LicenseError(f"Manifest fetch failed with status {resp.status_code}")

    def get_download_url(self, token: str, model_name: str) -> str:
        """Request a presigned R2 download URL for a model."""
        try:
            resp = httpx.post(
                self._url("/api/v1/models/download-url"),
                headers={"Authorization": f"Bearer {token}"},
                json={"model_name": model_name},
                timeout=_DOWNLOAD_TIMEOUT,
            )
        except httpx.RequestError as e:
            raise LicenseError(f"Network error requesting download URL: {e}") from e

        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            url: str = data["url"]
            return url

        if resp.status_code == 403:
            error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            required_tier = error_data.get("required_tier", "unknown")
            raise TierInsufficientError(model_name, required_tier)

        if resp.status_code == 404:
            raise LicenseError(f"Model not found: {model_name}")

        raise LicenseError(f"Download URL request failed with status {resp.status_code}")
