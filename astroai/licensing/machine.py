"""Hardware fingerprint generation (no PII stored)."""

from __future__ import annotations

import hashlib
import platform
import uuid


_HASH_PREFIX = "sha256:"


def get_machine_id() -> str:
    """Generate a deterministic, one-way hardware fingerprint.

    Combines OS node name and MAC-derived UUID into a SHA-256 hash.
    The result is not reversible to PII.
    """
    raw = f"{platform.node()}:{uuid.getnode()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"{_HASH_PREFIX}{digest}"


def verify_machine_id(expected: str) -> bool:
    """Check whether the current machine matches the expected fingerprint."""
    return get_machine_id() == expected
