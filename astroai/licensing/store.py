"""Fernet-encrypted local license storage (~/.astroai/license.dat)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from astroai.licensing.exceptions import LicenseError
from astroai.licensing.machine import get_machine_id


_DEFAULT_DIR = Path.home() / ".astroai"
_LICENSE_FILE = "license.dat"
_KEY_FILE = "license.key"


def _get_store_dir(base_dir: Path | None = None) -> Path:
    d = base_dir or _DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_or_create_key(store_dir: Path) -> bytes:
    """Derive or load per-machine Fernet key."""
    key_path = store_dir / _KEY_FILE
    if key_path.exists():
        return key_path.read_bytes()
    key = Fernet.generate_key()
    key_path.write_bytes(key)
    os.chmod(str(key_path), 0o600)
    return key


class LicenseStore:
    """Encrypted on-disk license persistence."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._dir = _get_store_dir(base_dir)
        self._key = _get_or_create_key(self._dir)
        self._fernet = Fernet(self._key)
        self._file = self._dir / _LICENSE_FILE

    def save(
        self,
        token_raw: str,
        last_online_at: datetime,
        attestation_raw: str | None = None,
        start_counter: int = 0,
    ) -> None:
        """Encrypt and persist license data including server attestation."""
        payload: dict[str, Any] = {
            "token": token_raw,
            "last_online_at": last_online_at.isoformat(),
            "machine_id": get_machine_id(),
            "attestation": attestation_raw,
            "start_counter": start_counter,
        }
        data = json.dumps(payload).encode()
        encrypted = self._fernet.encrypt(data)
        self._file.write_bytes(encrypted)

    def load(self) -> tuple[str, datetime, str | None, int] | None:
        """Load and decrypt stored license data.

        Returns (raw_jwt, last_online_at, raw_attestation, start_counter) or None.
        """
        if not self._file.exists():
            return None
        try:
            encrypted = self._file.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            payload: dict[str, Any] = json.loads(decrypted)
            token_raw: str = payload["token"]
            last_online_at = datetime.fromisoformat(payload["last_online_at"])
            if last_online_at.tzinfo is None:
                last_online_at = last_online_at.replace(tzinfo=timezone.utc)
            attestation_raw: str | None = payload.get("attestation")
            start_counter: int = int(payload.get("start_counter", 0))
            return token_raw, last_online_at, attestation_raw, start_counter
        except (InvalidToken, KeyError, json.JSONDecodeError, ValueError) as e:
            raise LicenseError(f"Corrupt license store: {e}") from e

    def increment_start_counter(self) -> int:
        """Increment offline start counter and persist. Returns new count."""
        loaded = self.load()
        if loaded is None:
            raise LicenseError("No license data to increment start counter")
        token_raw, last_online_at, attestation_raw, start_counter = loaded
        new_count = start_counter + 1
        self.save(token_raw, last_online_at, attestation_raw, new_count)
        return new_count

    def clear(self) -> None:
        """Remove stored license data."""
        if self._file.exists():
            self._file.unlink()

    @property
    def exists(self) -> bool:
        return self._file.exists()
