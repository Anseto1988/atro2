"""Unit tests for Fernet-encrypted license store."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from astroai.licensing.exceptions import LicenseError
from astroai.licensing.store import LicenseStore, _get_or_create_key


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    return tmp_path / "astroai_test"


@pytest.fixture()
def store(store_dir: Path) -> LicenseStore:
    return LicenseStore(base_dir=store_dir)


class TestLicenseStore:
    def test_roundtrip(self, store: LicenseStore) -> None:
        """Save and load must return identical data."""
        token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.test_payload"
        ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)

        store.save(token, ts)
        result = store.load()

        assert result is not None
        loaded_token, loaded_ts, loaded_att, loaded_counter = result
        assert loaded_token == token
        assert loaded_ts == ts
        assert loaded_att is None
        assert loaded_counter == 0

    def test_empty_store_returns_none(self, store: LicenseStore) -> None:
        """Load on empty store returns None."""
        assert store.load() is None

    def test_exists_property(self, store: LicenseStore) -> None:
        """exists reflects whether a license file is stored."""
        assert store.exists is False
        store.save("tok", datetime.now(timezone.utc))
        assert store.exists is True

    def test_clear(self, store: LicenseStore) -> None:
        """clear removes the stored license."""
        store.save("tok", datetime.now(timezone.utc))
        store.clear()
        assert store.exists is False
        assert store.load() is None

    def test_corrupt_data_raises(self, store: LicenseStore, store_dir: Path) -> None:
        """Corrupt file raises LicenseError."""
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "license.dat").write_bytes(b"not encrypted data")
        with pytest.raises(LicenseError, match="Corrupt license store"):
            store.load()

    def test_different_stores_isolated(self, tmp_path: Path) -> None:
        """Two stores in different dirs are independent."""
        s1 = LicenseStore(base_dir=tmp_path / "a")
        s2 = LicenseStore(base_dir=tmp_path / "b")

        s1.save("token_a", datetime.now(timezone.utc))
        assert s2.load() is None


class TestKeyProtectionFallback:
    def test_fallback_creates_key_file(self, store_dir: Path) -> None:
        """Without DPAPI/keyring, falls back to plain file."""
        store = LicenseStore(base_dir=store_dir)
        key_path = store_dir / "license.key"
        assert key_path.exists()

    def test_key_file_reused_on_second_init(self, store_dir: Path) -> None:
        """Key file is reused, not regenerated."""
        s1 = LicenseStore(base_dir=store_dir)
        s2 = LicenseStore(base_dir=store_dir)
        assert s1._key == s2._key


class TestIncrementStartCounter:
    def test_increment_no_data_raises(self, store: LicenseStore) -> None:
        with pytest.raises(LicenseError, match="No license data"):
            store.increment_start_counter()

    def test_increment_increases_counter(self, store: LicenseStore) -> None:
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        store.save("tok", ts)
        assert store.increment_start_counter() == 1
        assert store.increment_start_counter() == 2

    def test_load_naive_datetime_adds_utc(self, store: LicenseStore, store_dir: Path) -> None:
        """Timestamps stored without tzinfo are auto-assigned UTC on load (line 140)."""
        import json

        # Build a payload with naive datetime (no +00:00)
        key = store._key
        f = Fernet(key)
        payload = json.dumps({
            "token": "raw_tok",
            "last_online_at": "2026-03-15T10:00:00",  # naive
            "machine_id": "any",
            "attestation": None,
            "start_counter": 0,
        }).encode()
        store._file.write_bytes(f.encrypt(payload))

        result = store.load()
        assert result is not None
        _, ts, _, _ = result
        assert ts.tzinfo is not None


# ---------------------------------------------------------------------------
# _get_or_create_key — platform-specific branches
# ---------------------------------------------------------------------------

class TestGetOrCreateKey:
    def test_win32_dpapi_import_error_falls_back_to_file(self, tmp_path: Path) -> None:
        """ImportError in win32 DPAPI block → falls through to file fallback (lines 66-67)."""
        with patch("sys.platform", "win32"):
            with patch("astroai.licensing.store._protect_key_dpapi", side_effect=ImportError):
                key = _get_or_create_key(tmp_path)
        assert (tmp_path / "license.key").exists()
        assert isinstance(key, bytes)

    def test_win32_dpapi_exception_falls_back_to_file(self, tmp_path: Path) -> None:
        """General Exception in win32 DPAPI block → falls through to file fallback (lines 68-69)."""
        with patch("sys.platform", "win32"):
            with patch("astroai.licensing.store._protect_key_dpapi", side_effect=RuntimeError("DPAPI fail")):
                key = _get_or_create_key(tmp_path)
        assert isinstance(key, bytes)

    def test_non_win32_keyring_stores_new_key(self, tmp_path: Path) -> None:
        """Non-win32: keyring with no existing key generates and stores new key (lines 78-81)."""
        mock_kr = MagicMock()
        mock_kr.get_password.return_value = None
        generated_key = Fernet.generate_key()
        with patch("sys.platform", "linux"):
            with patch.dict(sys.modules, {"keyring": mock_kr}):
                with patch("cryptography.fernet.Fernet.generate_key", return_value=generated_key):
                    key = _get_or_create_key(tmp_path)
        assert key == generated_key
        mock_kr.set_password.assert_called_once()

    def test_non_win32_keyring_returns_existing_key(self, tmp_path: Path) -> None:
        """Non-win32: keyring with existing key returns it (lines 76-77)."""
        stored = Fernet.generate_key()
        mock_kr = MagicMock()
        mock_kr.get_password.return_value = stored.decode()
        with patch("sys.platform", "linux"):
            with patch.dict(sys.modules, {"keyring": mock_kr}):
                key = _get_or_create_key(tmp_path)
        assert key == stored

    def test_non_win32_keyring_import_error_falls_back_to_file(self, tmp_path: Path) -> None:
        """Non-win32: keyring ImportError → file fallback (lines 82-83)."""
        with patch("sys.platform", "linux"):
            with patch.dict(sys.modules, {"keyring": None}):
                key = _get_or_create_key(tmp_path)
        assert (tmp_path / "license.key").exists()
        assert isinstance(key, bytes)

    def test_non_win32_keyring_exception_falls_back_to_file(self, tmp_path: Path) -> None:
        """Non-win32: keyring Exception → file fallback (lines 84-85)."""
        mock_kr = MagicMock()
        mock_kr.get_password.side_effect = RuntimeError("keyring broken")
        with patch("sys.platform", "linux"):
            with patch.dict(sys.modules, {"keyring": mock_kr}):
                key = _get_or_create_key(tmp_path)
        assert isinstance(key, bytes)

    def test_file_fallback_reads_existing_key(self, tmp_path: Path) -> None:
        """File fallback reads existing key without regenerating (lines 87-88)."""
        existing = Fernet.generate_key()
        (tmp_path / "license.key").write_bytes(existing)
        with patch("sys.platform", "linux"):
            with patch.dict(sys.modules, {"keyring": None}):
                key = _get_or_create_key(tmp_path)
        assert key == existing

    def test_file_fallback_chmod_oserror_swallowed(self, tmp_path: Path) -> None:
        """OSError from os.chmod is silently swallowed (lines 93-94)."""
        with patch("sys.platform", "linux"):
            with patch.dict(sys.modules, {"keyring": None}):
                with patch("astroai.licensing.store.os.chmod", side_effect=OSError("perm")):
                    key = _get_or_create_key(tmp_path)
        assert isinstance(key, bytes)
