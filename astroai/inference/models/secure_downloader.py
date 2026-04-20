"""Secure model downloader with R2 presigned URL support via license server."""

from __future__ import annotations

import logging
from pathlib import Path

from astroai.core.pipeline.base import ProgressCallback
from astroai.inference.models.downloader import ModelDownloader, ModelManifestEntry
from astroai.licensing.client import LicenseClient
from astroai.licensing.exceptions import LicenseError, NotActivated
from astroai.licensing.store import LicenseStore

__all__ = ["SecureModelDownloader"]

logger = logging.getLogger(__name__)


class SecureModelDownloader(ModelDownloader):
    """Downloads models via R2 presigned URLs obtained from the license server.

    Falls back to the base ModelDownloader manifest for free-tier models
    when the license server is unreachable.
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        progress: ProgressCallback | None = None,
        license_client: LicenseClient | None = None,
        license_store: LicenseStore | None = None,
    ) -> None:
        super().__init__(models_dir=models_dir, progress=progress)
        self._license_client = license_client or LicenseClient()
        self._license_store = license_store or LicenseStore()
        self._server_manifest: dict[str, ModelManifestEntry] | None = None

    def _get_token(self) -> str:
        """Retrieve the current raw JWT from the license store."""
        stored = self._license_store.load()
        if stored is None:
            raise NotActivated("No license found — please activate first")
        raw_jwt, *_ = stored
        return raw_jwt

    def get_manifest(self) -> dict[str, ModelManifestEntry]:
        """Fetch manifest from the license server (tier-filtered).

        Caches result for the lifetime of this instance.
        Falls back to parent manifest on network failure.
        """
        if self._server_manifest is not None:
            return self._server_manifest

        try:
            token = self._get_token()
            raw_models = self._license_client.get_model_manifest(token)
            entries: dict[str, ModelManifestEntry] = {}
            for m in raw_models:
                entries[m["name"]] = ModelManifestEntry(
                    name=m["name"],
                    url="",  # URL resolved at download time via presigned URL
                    sha256=m.get("sha256", ""),
                    filename=m["filename"],
                    description=m.get("description", ""),
                )
            self._server_manifest = entries
            logger.debug("Server manifest loaded: %d models", len(entries))
            return entries
        except (LicenseError, KeyError) as exc:
            logger.warning("Server manifest unavailable (%s), using local fallback", exc)
            return super().get_manifest()

    def ensure_model(self, name: str) -> Path:
        """Download model via presigned R2 URL if not already cached."""
        manifest = self.get_manifest()
        if name not in manifest:
            raise KeyError(f"Unknown model: {name}")

        entry = manifest[name]

        if not entry.sha256:
            raise ValueError(
                f"Model '{name}' has no SHA-256 checksum. "
                "Refusing download — integrity cannot be verified."
            )

        safe_filename = Path(entry.filename).name
        if safe_filename != entry.filename:
            raise ValueError(
                f"Path traversal detected in filename '{entry.filename}'. "
                "Only plain filenames are accepted."
            )
        target = self._models_dir / safe_filename

        if target.exists() and self._verify_checksum(target, entry.sha256):
            logger.info("Model '%s' already cached at %s", name, target)
            return target

        presigned_url = self._resolve_download_url(name)
        secure_entry = ModelManifestEntry(
            name=entry.name,
            url=presigned_url,
            sha256=entry.sha256,
            filename=safe_filename,
            description=entry.description,
        )

        try:
            self._download(secure_entry, target)
        except Exception as exc:
            logger.warning("Download failed for '%s': %s", name, exc)
            raise

        if not self._verify_checksum(target, entry.sha256):
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for '{name}'. File deleted. "
                "Re-run to retry download."
            )

        logger.info("Model '%s' downloaded and verified at %s", name, target)
        return target

    def _resolve_download_url(self, model_name: str) -> str:
        """Request a presigned download URL from the license server."""
        token = self._get_token()
        return self._license_client.get_download_url(token, model_name)

    def invalidate_manifest_cache(self) -> None:
        """Force re-fetch of server manifest on next access."""
        self._server_manifest = None
