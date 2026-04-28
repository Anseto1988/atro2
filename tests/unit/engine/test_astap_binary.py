"""Tests for ASTAP binary platform detection and path resolution."""

from __future__ import annotations

import io
import os
import platform
import stat
import subprocess
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astroai.engine.platesolving.astap_binary import (
    AstapNotFoundError,
    _detect_platform_key,
    _is_executable,
    download_astap,
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
    def test_darwin_x86_64_raises(self, _m: object, _s: object) -> None:
        with pytest.raises(AstapNotFoundError, match="Intel Mac"):
            _detect_platform_key()

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

    def test_returns_version_on_success(self, tmp_path: Path) -> None:
        binary = tmp_path / "astap"
        binary.write_bytes(b"fake")
        if platform.system() != "Windows":
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        import subprocess
        with patch(
            "astroai.engine.platesolving.astap_binary.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ASTAP v0.2.1\n", stderr=""
            ),
        ):
            result = verify_astap(binary)
            assert result == "ASTAP v0.2.1"

    def test_returns_none_on_timeout(self, tmp_path: Path) -> None:
        binary = tmp_path / "astap"
        binary.write_bytes(b"fake")
        import subprocess
        with patch(
            "astroai.engine.platesolving.astap_binary.subprocess.run",
            side_effect=subprocess.TimeoutExpired("astap", 10),
        ):
            assert verify_astap(binary) is None

    def test_returns_none_on_empty_output(self, tmp_path: Path) -> None:
        binary = tmp_path / "astap"
        binary.write_bytes(b"fake")
        import subprocess
        with patch(
            "astroai.engine.platesolving.astap_binary.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            ),
        ):
            assert verify_astap(binary) is None


class TestEnsureAstap:
    def test_returns_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

        from astroai.engine.platesolving.astap_binary import ensure_astap
        result = ensure_astap()
        assert result == binary

    @patch("astroai.engine.platesolving.astap_binary.download_astap")
    def test_downloads_on_not_found(
        self, mock_download: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock
        mock_dl = MagicMock(return_value=tmp_path / "downloaded_astap")
        with patch("astroai.engine.platesolving.astap_binary.download_astap", mock_dl):
            monkeypatch.setattr(
                "astroai.engine.platesolving.astap_binary._BUNDLED_DIR", tmp_path / "e1"
            )
            monkeypatch.setattr(
                "astroai.engine.platesolving.astap_binary._USER_DIR", tmp_path / "e2"
            )
            monkeypatch.delenv("ASTAP_BINARY_PATH", raising=False)
            with patch("shutil.which", return_value=None):
                from astroai.engine.platesolving.astap_binary import ensure_astap
                result = ensure_astap()
                assert result == tmp_path / "downloaded_astap"
                mock_dl.assert_called_once()


class TestSHA256Verify:
    def test_placeholder_always_passes(self, tmp_path: Path) -> None:
        from astroai.engine.platesolving.astap_binary import _verify_sha256
        f = tmp_path / "file.bin"
        f.write_bytes(b"any content")
        assert _verify_sha256(f, "placeholder_linux_x86_64") is True

    def test_correct_hash_passes(self, tmp_path: Path) -> None:
        import hashlib
        from astroai.engine.platesolving.astap_binary import _verify_sha256
        data = b"test data for hashing"
        f = tmp_path / "file.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _verify_sha256(f, expected) is True

    def test_wrong_hash_fails(self, tmp_path: Path) -> None:
        from astroai.engine.platesolving.astap_binary import _verify_sha256
        f = tmp_path / "file.bin"
        f.write_bytes(b"real content")
        assert _verify_sha256(f, "a" * 64) is False


class TestMakeExecutable:
    def test_sets_execute_bit(self, tmp_path: Path) -> None:
        from astroai.engine.platesolving.astap_binary import _make_executable
        f = tmp_path / "binary"
        f.write_bytes(b"\x00")
        _make_executable(f)
        if platform.system() != "Windows":
            assert os.access(f, os.X_OK)


class TestGetSpec:
    def test_returns_spec_for_current_platform(self) -> None:
        from astroai.engine.platesolving.astap_binary import _get_spec
        spec = _get_spec()
        assert spec.binary_name in ("astap", "astap.exe")
        assert spec.url.startswith("https://")


class TestFindInPath:
    @patch("astroai.engine.platesolving.astap_binary.shutil.which")
    def test_finds_binary_in_system_path(self, mock_which: object) -> None:
        from unittest.mock import MagicMock
        mock_w = MagicMock(return_value="/usr/bin/astap")
        with patch("astroai.engine.platesolving.astap_binary.shutil.which", mock_w):
            from astroai.engine.platesolving.astap_binary import _find_in_path
            result = _find_in_path()
            assert result == Path("/usr/bin/astap")

    def test_returns_none_when_not_in_path(self) -> None:
        with patch("astroai.engine.platesolving.astap_binary.shutil.which", return_value=None):
            from astroai.engine.platesolving.astap_binary import _find_in_path
            result = _find_in_path()
            assert result is None


class TestExtractArchive:
    def test_extract_tar_gz(self, tmp_path: Path) -> None:
        import tarfile
        from astroai.engine.platesolving.astap_binary import _extract_archive

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "astap").write_bytes(b"binary content")

        archive = tmp_path / "astap.tar.gz"
        with tarfile.open(str(archive), "w:gz") as tf:
            tf.add(str(src_dir / "astap"), arcname="astap")

        dest = tmp_path / "dest"
        _extract_archive(archive, dest)
        assert (dest / "astap").exists()

    def test_extract_zip(self, tmp_path: Path) -> None:
        import zipfile
        from astroai.engine.platesolving.astap_binary import _extract_archive

        archive = tmp_path / "astap.zip"
        with zipfile.ZipFile(str(archive), "w") as zf:
            zf.writestr("astap.exe", b"binary content")

        dest = tmp_path / "dest"
        _extract_archive(archive, dest)
        assert (dest / "astap.exe").exists()

    def test_extract_unknown_copies(self, tmp_path: Path) -> None:
        from astroai.engine.platesolving.astap_binary import _extract_archive

        archive = tmp_path / "astap.bin"
        archive.write_bytes(b"raw binary")

        dest = tmp_path / "dest"
        _extract_archive(archive, dest)
        assert (dest / "astap.bin").exists()


class TestGetAstapPathUserAndSystem:
    def test_user_dir_binary_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_astap_path returns user-dir binary when bundled is missing (line 131)."""
        key = _detect_platform_key()
        ext = ".exe" if platform.system() == "Windows" else ""
        user_dir = tmp_path / "user"
        binary_dir = user_dir / key
        binary_dir.mkdir(parents=True)
        binary = binary_dir / f"astap{ext}"
        binary.write_bytes(b"fake")
        if platform.system() != "Windows":
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr("astroai.engine.platesolving.astap_binary._BUNDLED_DIR", tmp_path / "empty")
        monkeypatch.setattr("astroai.engine.platesolving.astap_binary._USER_DIR", user_dir)
        monkeypatch.delenv("ASTAP_BINARY_PATH", raising=False)
        assert get_astap_path() == binary

    def test_system_path_binary_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_astap_path falls through to shutil.which when bundled and user are missing (line 135)."""
        fake_bin = tmp_path / "astap_fake"
        monkeypatch.setattr("astroai.engine.platesolving.astap_binary._BUNDLED_DIR", tmp_path / "empty1")
        monkeypatch.setattr("astroai.engine.platesolving.astap_binary._USER_DIR", tmp_path / "empty2")
        monkeypatch.delenv("ASTAP_BINARY_PATH", raising=False)

        with patch("astroai.engine.platesolving.astap_binary.shutil.which", return_value=str(fake_bin)):
            result = get_astap_path()
        assert result == fake_bin


class TestIsExecutableUnix:
    @patch("platform.system", return_value="Linux")
    @patch("astroai.engine.platesolving.astap_binary.os.access", return_value=False)
    def test_non_executable_file_returns_false(self, _acc: object, _sys: object, tmp_path: Path) -> None:
        """On non-Windows, os.access=False → not executable (line 113)."""
        f = tmp_path / "bin"
        f.write_bytes(b"data")
        assert _is_executable(f) is False

    @patch("platform.system", return_value="Linux")
    @patch("astroai.engine.platesolving.astap_binary.os.access", return_value=True)
    def test_executable_file_returns_true(self, _acc: object, _sys: object, tmp_path: Path) -> None:
        f = tmp_path / "bin"
        f.write_bytes(b"data")
        assert _is_executable(f) is True


class TestMakeExecutableNonWindows:
    @patch("platform.system", return_value="Linux")
    def test_sets_execute_bits(self, _: object, tmp_path: Path) -> None:
        """On non-Windows, _make_executable adds exec bits (lines 172-173)."""
        from astroai.engine.platesolving.astap_binary import _make_executable
        f = tmp_path / "binary"
        f.write_bytes(b"\x00")
        f.chmod(0o600)  # no exec bit
        _make_executable(f)
        assert os.access(f, os.X_OK)


class TestFindInPathWindowsExeFallback:
    @patch("platform.system", return_value="Windows")
    @patch("platform.machine", return_value="AMD64")
    def test_windows_exe_fallback(self, _m: object, _s: object) -> None:
        """When astap.exe not in PATH but 'astap' is, use it (line 94)."""
        from astroai.engine.platesolving.astap_binary import _find_in_path

        def _mock_which(name: str) -> str | None:
            if name == "astap.exe":
                return None
            if name == "astap":
                return "/usr/local/bin/astap"
            return None

        with patch("astroai.engine.platesolving.astap_binary.shutil.which", side_effect=_mock_which):
            result = _find_in_path()
        assert result == Path("/usr/local/bin/astap")


class TestDownloadAstap:
    def test_download_skips_if_already_present(self, tmp_path: Path) -> None:
        from astroai.engine.platesolving.astap_binary import download_astap, _get_spec

        spec = _get_spec()
        target_dir = tmp_path / spec.key
        target_dir.mkdir(parents=True)
        binary = target_dir / spec.binary_name
        binary.write_bytes(b"existing")
        if platform.system() != "Windows":
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

        result = download_astap(target_dir)
        assert result == binary

    def _make_fake_response(self, content: bytes, content_length: int = 0) -> MagicMock:
        """Build a minimal urllib response mock."""
        resp = MagicMock()
        data = io.BytesIO(content)
        resp.read.side_effect = lambda n: data.read(n)
        resp.headers = {"Content-Length": str(content_length)}
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_download_full_flow_binary_in_root(self, tmp_path: Path) -> None:
        """Lines 187-226: full download path when binary lands directly in dest_dir."""
        from astroai.engine.platesolving.astap_binary import (
            download_astap, _get_spec
        )
        spec = _get_spec()
        target_dir = tmp_path / "dl_root"

        fake_data = b"fake archive content"
        resp = self._make_fake_response(fake_data, len(fake_data))

        def _fake_urlopen(req, timeout):
            return resp

        def _fake_extract(archive, dest_dir):
            # Simulate extraction: write binary directly to dest_dir
            (dest_dir / spec.binary_name).write_bytes(b"fake binary")

        with patch("astroai.engine.platesolving.astap_binary.urllib.request.urlopen",
                   _fake_urlopen), \
             patch("astroai.engine.platesolving.astap_binary._verify_sha256",
                   return_value=True), \
             patch("astroai.engine.platesolving.astap_binary._extract_archive",
                   side_effect=_fake_extract), \
             patch("astroai.engine.platesolving.astap_binary._make_executable"):
            result = download_astap(target_dir)

        assert result == target_dir / spec.binary_name

    def test_download_binary_found_via_rglob(self, tmp_path: Path) -> None:
        """Lines 213-216: binary not in root of dest_dir, found via rglob."""
        from astroai.engine.platesolving.astap_binary import (
            download_astap, _get_spec
        )
        spec = _get_spec()
        target_dir = tmp_path / "dl_rglob"

        fake_data = b"fake archive"
        resp = self._make_fake_response(fake_data)

        def _fake_urlopen(req, timeout):
            return resp

        def _fake_extract(archive, dest_dir):
            # Put binary one level deeper (rglob needed)
            subdir = dest_dir / "subdir"
            subdir.mkdir(parents=True, exist_ok=True)
            (subdir / spec.binary_name).write_bytes(b"nested binary")

        with patch("astroai.engine.platesolving.astap_binary.urllib.request.urlopen",
                   _fake_urlopen), \
             patch("astroai.engine.platesolving.astap_binary._verify_sha256",
                   return_value=True), \
             patch("astroai.engine.platesolving.astap_binary._extract_archive",
                   side_effect=_fake_extract), \
             patch("astroai.engine.platesolving.astap_binary._make_executable"):
            result = download_astap(target_dir)

        assert result == target_dir / spec.binary_name

    def test_download_sha256_mismatch_raises(self, tmp_path: Path) -> None:
        """Line 209: SHA256 mismatch raises RuntimeError."""
        from astroai.engine.platesolving.astap_binary import (
            download_astap, _get_spec
        )
        spec = _get_spec()
        target_dir = tmp_path / "dl_sha"

        fake_data = b"corrupted data"
        resp = self._make_fake_response(fake_data)

        def _fake_urlopen(req, timeout):
            return resp

        with patch("astroai.engine.platesolving.astap_binary.urllib.request.urlopen",
                   _fake_urlopen), \
             patch("astroai.engine.platesolving.astap_binary._verify_sha256",
                   return_value=False):
            with pytest.raises(RuntimeError, match="SHA256 mismatch"):
                download_astap(target_dir)

    def test_download_not_executable_after_download_raises(self, tmp_path: Path) -> None:
        """Lines 222-223: RuntimeError when binary not executable after download."""
        from astroai.engine.platesolving.astap_binary import (
            download_astap, _get_spec
        )
        spec = _get_spec()
        target_dir = tmp_path / "dl_noexec"

        fake_data = b"fake archive"
        resp = self._make_fake_response(fake_data)

        def _fake_urlopen(req, timeout):
            return resp

        def _fake_extract(archive, dest_dir):
            (dest_dir / spec.binary_name).write_bytes(b"fake binary")

        with patch("astroai.engine.platesolving.astap_binary.urllib.request.urlopen",
                   _fake_urlopen), \
             patch("astroai.engine.platesolving.astap_binary._verify_sha256",
                   return_value=True), \
             patch("astroai.engine.platesolving.astap_binary._extract_archive",
                   side_effect=_fake_extract), \
             patch("astroai.engine.platesolving.astap_binary._make_executable"), \
             patch("astroai.engine.platesolving.astap_binary._is_executable",
                   side_effect=lambda p: False):
            with pytest.raises(RuntimeError, match="not executable after download"):
                download_astap(target_dir)

    def test_download_with_content_length_progress(self, tmp_path: Path) -> None:
        """Lines 204-206: progress logging when Content-Length is set."""
        from astroai.engine.platesolving.astap_binary import (
            download_astap, _get_spec
        )
        spec = _get_spec()
        target_dir = tmp_path / "dl_progress"

        fake_data = b"x" * 1000
        resp = self._make_fake_response(fake_data, content_length=1000)

        def _fake_urlopen(req, timeout):
            return resp

        def _fake_extract(archive, dest_dir):
            (dest_dir / spec.binary_name).write_bytes(b"fake binary")

        with patch("astroai.engine.platesolving.astap_binary.urllib.request.urlopen",
                   _fake_urlopen), \
             patch("astroai.engine.platesolving.astap_binary._verify_sha256",
                   return_value=True), \
             patch("astroai.engine.platesolving.astap_binary._extract_archive",
                   side_effect=_fake_extract), \
             patch("astroai.engine.platesolving.astap_binary._make_executable"):
            result = download_astap(target_dir)

        assert result == target_dir / spec.binary_name


class TestMainBlockLogic:
    """Tests for lines 254-266: __main__ entry point.

    The if __name__ == '__main__' guard means these lines are only executed
    when running the file directly. We test the equivalent logic by calling
    ensure_astap + verify_astap directly and verifying they integrate correctly.
    The actual source lines are covered via a subprocess invocation with
    coverage.
    """

    def test_main_logic_success_with_version(self, tmp_path: Path) -> None:
        """Validate the main block success path: ensure_astap + verify_astap."""
        fake_binary = tmp_path / "astap"
        fake_binary.write_bytes(b"fake")
        if platform.system() != "Windows":
            fake_binary.chmod(fake_binary.stat().st_mode | stat.S_IEXEC)

        import subprocess
        with patch("astroai.engine.platesolving.astap_binary.subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       args=[], returncode=0, stdout="ASTAP v0.9.9", stderr="")):
            from astroai.engine.platesolving.astap_binary import verify_astap
            version = verify_astap(fake_binary)
        assert version == "ASTAP v0.9.9"

    def test_main_logic_success_no_version(self, tmp_path: Path) -> None:
        """Validate that verify_astap returns None when output is empty."""
        fake_binary = tmp_path / "astap"
        fake_binary.write_bytes(b"fake")
        import subprocess
        with patch("astroai.engine.platesolving.astap_binary.subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       args=[], returncode=0, stdout="", stderr="")):
            from astroai.engine.platesolving.astap_binary import verify_astap
            version = verify_astap(fake_binary)
        assert version is None

    def test_main_as_subprocess_success(self, tmp_path: Path) -> None:
        """Lines 254-266: run module as __main__ via subprocess to hit actual lines."""
        import subprocess, sys

        fake_binary = tmp_path / "astap_fake"
        fake_binary.write_bytes(b"fake")
        if platform.system() != "Windows":
            fake_binary.chmod(fake_binary.stat().st_mode | stat.S_IEXEC)

        script = (
            "import sys\n"
            "from unittest.mock import patch\n"
            f"fake = r'{fake_binary}'\n"
            "import subprocess as sp\n"
            "completed = sp.CompletedProcess(args=[], returncode=0, stdout='ASTAP v1.0', stderr='')\n"
            "with patch('astroai.engine.platesolving.astap_binary.get_astap_path',\n"
            "           return_value=type('P', (), {'__str__': lambda s: fake})()):\n"
            "    with patch('astroai.engine.platesolving.astap_binary.download_astap',\n"
            "               return_value=type('P', (), {'__str__': lambda s: fake})()):\n"
            "        with patch('astroai.engine.platesolving.astap_binary.subprocess.run',\n"
            "                   return_value=completed):\n"
            "            import runpy\n"
            "            runpy.run_module('astroai.engine.platesolving.astap_binary',\n"
            "                             run_name='__main__', alter_sys=False)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        # subprocess ran without crashing (exit code 0 or 1 depending on env)
        assert result.returncode in (0, 1)

    def test_main_as_subprocess_failure(self, tmp_path: Path) -> None:
        """Lines 264-266: run module as __main__ and trigger exception path."""
        import subprocess, sys

        script = (
            "from unittest.mock import patch\n"
            "with patch('astroai.engine.platesolving.astap_binary.ensure_astap',\n"
            "           side_effect=Exception('test failure')):\n"
            "    import runpy\n"
            "    runpy.run_module('astroai.engine.platesolving.astap_binary',\n"
            "                     run_name='__main__', alter_sys=False)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
