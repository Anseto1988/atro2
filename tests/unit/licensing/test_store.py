"""Unit tests for Fernet-encrypted license store."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from astroai.licensing.exceptions import LicenseError
from astroai.licensing.store import LicenseStore


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
        loaded_token, loaded_ts = result
        assert loaded_token == token
        assert loaded_ts == ts

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
