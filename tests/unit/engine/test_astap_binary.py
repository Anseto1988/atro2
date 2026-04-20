"""Tests for ASTAP binary platform detection and path resolution."""

from __future__ import annotations

import os
import platform
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from astroai.engine.platesolving.astap_binary import (
    AstapNotFoundError,
    _detect_platform_key,
    _is_executable,
    get_astap_path,
    verify_astap,
)


class TestDetectPlatformKey:
    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="x86_64")
    def test_linux_x86_64(self, _m: object, _s: object) -> None:
        assert _detect_platform_key() == "linux-x86_64"

    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="amd64")
    def test_linux_amd64(self, _m: object, _s: object) -> None:
        assert _detect_platform_key() == "linux-x86_64"

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    def test_darwin_arm64(self, _m: object, _s: object) -> None:
        assert _detect_platform_key() == "darwin-arm64"

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="x86_64")
    def test_darwin_x86_64_maps_to_arm64(self, _m: object, _s: object) -> None:
        assert _detect_platform_key() == "darwin-arm64"

    @patch("platform.system", return_value="Windows")
    @patch("platform.machine", return_value="AMD64")
    def test_windows_amd64(self, _m: object, _s: object) -> None:
        assert _detect_platform_key() == "win32-x86_64"

    @patch("platform.system", return_value="FreeBSD")
    @patch("platform.machine", return_value="x86_64")
    def test_unsupported_raises(self, _m: object, _s: object) -> None:
        with pytest.raises(AstapNotFoundError, match="Unsupported platform"):
            _detect_platform_key()

    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="aarch64")
    def test_linux_arm_unsupported(self, _m: object, _s: object) -> None:
        with pytest.raises(AstapNotFoundError, match="Unsupported platform"):
            _detect_platform_key()


class TestGetAstapPath:
    def test_env_override(self, tmp_path: Path) -> None:
        binary = tmp_path / "astap"
        binary.write_bytes(b"#!/bin/sh\necho test")
        if platform.system() != "Windows":
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

        with patch.dict(os.environ, {"ASTAP_BINARY_PATH": str(binary)}):
            result = get_astap_path()
            assert result == binary

    def test_env_override_missing_raises(self) -> None:
        with patch.dict(os.environ, {"ASTAP_BINARY_PATH": "/nonexistent/astap"}):
            with pytest.raises(AstapNotFoundError, match="not executable"):
                get_astap_path()

    def test_bundled_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        key = _detect_platform_key()
        ext = ".exe" if platform.system() == "Windows" else ""
        binary_dir = tmp_path / key
        binary_dir.mkdir(parents=True)
        binary = binary_dir / f"astap{ext}"
        binary.write_bytes(b"fake binary")
        if platform.system() != "Windows":
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            "astroai.engine.platesolving.astap_binary._BUNDLED_DIR", tmp_path
        )
        monkeypatch.delenv("ASTAP_BINARY_PATH", raising=False)
        result = get_astap_path()
        assert result == binary

    def test_not_found_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "astroai.engine.platesolving.astap_binary._BUNDLED_DIR", tmp_path / "empty1"
        )
        monkeypatch.setattr(
            "astroai.engine.platesolving.astap_binary._USER_DIR", tmp_path / "empty2"
        )
        monkeypatch.delenv("ASTAP_BINARY_PATH", raising=False)
        with patch("shutil.which", return_value=None):
            with pytest.raises(AstapNotFoundError, match="not found"):
                get_astap_path()


class TestIsExecutable:
    def test_nonexistent(self) -> None:
        assert _is_executable(Path("/nonexistent/astap")) is False

    def test_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test"
        f.write_bytes(b"x")
        if platform.system() == "Windows":
            assert _is_executable(f) is True
        else:
            assert _is_executable(f) is False
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
            assert _is_executable(f) is True


class TestVerifyAstap:
    def test_returns_none_when_not_found(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "astroai.engine.platesolving.astap_binary._BUNDLED_DIR", tmp_path / "e1"
        )
        monkeypatch.setattr(
            "astroai.engine.platesolving.astap_binary._USER_DIR", tmp_path / "e2"
        )
        monkeypatch.delenv("ASTAP_BINARY_PATH", raising=False)
        with patch("shutil.which", return_value=None):
            assert verify_astap() is None
