"""Unit tests for machine fingerprint generation."""

from __future__ import annotations

from astroai.licensing.machine import get_machine_id, verify_machine_id


class TestGetMachineId:
    def test_deterministic(self) -> None:
        """Machine ID must be identical across multiple calls."""
        id1 = get_machine_id()
        id2 = get_machine_id()
        assert id1 == id2

    def test_sha256_prefix(self) -> None:
        """Machine ID must start with the sha256: prefix."""
        mid = get_machine_id()
        assert mid.startswith("sha256:")

    def test_hash_length(self) -> None:
        """Hash portion must be exactly 32 hex chars."""
        mid = get_machine_id()
        hash_part = mid.removeprefix("sha256:")
        assert len(hash_part) == 32
        assert all(c in "0123456789abcdef" for c in hash_part)

    def test_no_pii_in_output(self) -> None:
        """Output must not contain raw hostname or MAC address."""
        import platform
        import uuid

        mid = get_machine_id()
        assert platform.node() not in mid
        assert str(uuid.getnode()) not in mid


class TestVerifyMachineId:
    def test_matches_current(self) -> None:
        """verify_machine_id returns True for the current machine."""
        assert verify_machine_id(get_machine_id()) is True

    def test_rejects_different(self) -> None:
        """verify_machine_id returns False for a foreign fingerprint."""
        assert verify_machine_id("sha256:00000000000000000000000000000000") is False
