"""Tests for SecureModelDownloader with mocked license server."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest

from astroai.inference.models.downloader import ModelManifestEntry
from astroai.inference.models.secure_downloader import SecureModelDownloader
from astroai.licensing.client import LicenseClient
from astroai.licensing.exceptions import LicenseError, NotActivated, TierInsufficientError
from astroai.licensing.store import LicenseStore


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture()
def mock_store() -> MagicMock:
    store = MagicMock(spec=LicenseStore)
    store.load.return_value = ("fake.jwt.token", datetime(2026, 1, 1, tzinfo=timezone.utc))
    return store


@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock(spec=LicenseClient)
    client.get_model_manifest.return_value = [
        {
            "name": "nafnet_denoise",
            "filename": "nafnet_denoise.onnx",
            "sha256": "a" * 64,
            "description": "NAFNet denoising model",
            "size_bytes": 1024,
            "min_tier": "free",
        },
        {
            "name": "nafnet_denoise_pro",
            "filename": "nafnet_denoise_pro.onnx",
            "sha256": "b" * 64,
            "description": "NAFNet Pro model",
            "size_bytes": 2048,
            "min_tier": "pro_monthly",
        },
    ]
    client.get_download_url.return_value = "https://bucket.r2.cloudflarestorage.com/models/nafnet_denoise.onnx?signed=1"
    return client


@pytest.fixture()
def downloader(models_dir: Path, mock_client: MagicMock, mock_store: MagicMock) -> SecureModelDownloader:
    return SecureModelDownloader(
        models_dir=models_dir,
        license_client=mock_client,
        license_store=mock_store,
    )


class TestSecureModelDownloaderManifest:
    def test_get_manifest_from_server(self, downloader: SecureModelDownloader, mock_client: MagicMock) -> None:
        manifest = downloader.get_manifest()
        assert "nafnet_denoise" in manifest
        assert "nafnet_denoise_pro" in manifest
        mock_client.get_model_manifest.assert_called_once_with("fake.jwt.token")

    def test_manifest_cached_after_first_fetch(self, downloader: SecureModelDownloader, mock_client: MagicMock) -> None:
        downloader.get_manifest()
        downloader.get_manifest()
        mock_client.get_model_manifest.assert_called_once()

    def test_invalidate_cache_forces_refetch(self, downloader: SecureModelDownloader, mock_client: MagicMock) -> None:
        downloader.get_manifest()
        downloader.invalidate_manifest_cache()
        downloader.get_manifest()
        assert mock_client.get_model_manifest.call_count == 2

    def test_manifest_falls_back_on_network_error(self, downloader: SecureModelDownloader, mock_client: MagicMock) -> None:
        mock_client.get_model_manifest.side_effect = LicenseError("timeout")
        manifest = downloader.get_manifest()
        assert "nafnet_denoise" in manifest
        assert "nafnet_denoise_pro" not in manifest

    def test_manifest_falls_back_when_not_activated(self, models_dir: Path, mock_client: MagicMock) -> None:
        store = MagicMock(spec=LicenseStore)
        store.load.return_value = None
        dl = SecureModelDownloader(models_dir=models_dir, license_client=mock_client, license_store=store)
        manifest = dl.get_manifest()
        assert "nafnet_denoise" in manifest
        mock_client.get_model_manifest.assert_not_called()


class TestSecureModelDownloaderDownload:
    def test_ensure_model_uses_presigned_url(self, downloader: SecureModelDownloader, mock_client: MagicMock, models_dir: Path) -> None:
        content = b"model data"
        sha = hashlib.sha256(content).hexdigest()

        mock_client.get_model_manifest.return_value = [
            {"name": "test_model", "filename": "test.onnx", "sha256": sha, "description": "test"},
        ]
        mock_client.get_download_url.return_value = "https://r2.example.com/test.onnx?signed=1"

        with patch.object(downloader, "_download") as mock_download:
            mock_download.side_effect = lambda entry, target: target.write_bytes(content)
            downloader.invalidate_manifest_cache()
            path = downloader.ensure_model("test_model")

        assert path == models_dir / "test.onnx"
        mock_client.get_download_url.assert_called_once_with("fake.jwt.token", "test_model")
        call_entry = mock_download.call_args[0][0]
        assert call_entry.url == "https://r2.example.com/test.onnx?signed=1"

    def test_ensure_model_skips_download_if_cached(self, downloader: SecureModelDownloader, mock_client: MagicMock, models_dir: Path) -> None:
        content = b"cached model"
        sha = hashlib.sha256(content).hexdigest()

        mock_client.get_model_manifest.return_value = [
            {"name": "cached", "filename": "cached.onnx", "sha256": sha, "description": ""},
        ]
        downloader.invalidate_manifest_cache()
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "cached.onnx").write_bytes(content)

        path = downloader.ensure_model("cached")
        assert path == models_dir / "cached.onnx"
        mock_client.get_download_url.assert_not_called()

    def test_ensure_model_raises_on_unknown_model(self, downloader: SecureModelDownloader) -> None:
        with pytest.raises(KeyError, match="Unknown model"):
            downloader.ensure_model("nonexistent")

    def test_checksum_mismatch_deletes_file(self, downloader: SecureModelDownloader, mock_client: MagicMock, models_dir: Path) -> None:
        mock_client.get_model_manifest.return_value = [
            {"name": "bad", "filename": "bad.onnx", "sha256": "f" * 64, "description": ""},
        ]
        mock_client.get_download_url.return_value = "https://r2.example.com/bad.onnx?signed=1"
        downloader.invalidate_manifest_cache()

        with patch.object(downloader, "_download") as mock_dl:
            mock_dl.side_effect = lambda entry, target: target.write_bytes(b"wrong content")
            with pytest.raises(RuntimeError, match="Checksum mismatch"):
                downloader.ensure_model("bad")

        assert not (models_dir / "bad.onnx").exists()


class TestSecureModelDownloaderTierBlocking:
    def test_tier_insufficient_raises(self, downloader: SecureModelDownloader, mock_client: MagicMock) -> None:
        mock_client.get_download_url.side_effect = TierInsufficientError("starnet_pro", "pro_monthly")
        mock_client.get_model_manifest.return_value = [
            {"name": "starnet_pro", "filename": "starnet_pro.onnx", "sha256": "c" * 64, "description": ""},
        ]
        downloader.invalidate_manifest_cache()

        with pytest.raises(TierInsufficientError) as exc_info:
            downloader.ensure_model("starnet_pro")
        assert exc_info.value.model_name == "starnet_pro"
        assert exc_info.value.required_tier == "pro_monthly"

    def test_not_activated_raises_on_download(self, models_dir: Path, mock_client: MagicMock) -> None:
        store = MagicMock(spec=LicenseStore)
        store.load.return_value = None
        dl = SecureModelDownloader(models_dir=models_dir, license_client=mock_client, license_store=store)

        with pytest.raises(NotActivated):
            dl.ensure_model("nafnet_denoise")


class TestSecureModelDownloaderNoSha:
    def test_skips_checksum_when_sha_empty(self, downloader: SecureModelDownloader, mock_client: MagicMock, models_dir: Path) -> None:
        mock_client.get_model_manifest.return_value = [
            {"name": "nosha", "filename": "nosha.onnx", "sha256": "", "description": ""},
        ]
        mock_client.get_download_url.return_value = "https://r2.example.com/nosha.onnx?signed=1"
        downloader.invalidate_manifest_cache()

        with patch.object(downloader, "_download") as mock_dl:
            mock_dl.side_effect = lambda entry, target: target.write_bytes(b"data")
            path = downloader.ensure_model("nosha")

        assert path == models_dir / "nosha.onnx"
        assert path.exists()
