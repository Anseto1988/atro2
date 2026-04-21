"""ASTAP binary management: platform detection, path resolution, and download."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from pathlib import Path

__all__ = ["get_astap_path", "ensure_astap", "AstapNotFoundError"]

logger = logging.getLogger(__name__)

_BUNDLED_DIR = Path(__file__).parent / "bin"
_USER_DIR = Path.home() / ".astroai" / "bin"


class AstapNotFoundError(FileNotFoundError):
    pass


class _PlatformSpec:
    __slots__ = ("key", "binary_name", "url", "sha256")

    def __init__(self, key: str, binary_name: str, url: str, sha256: str) -> None:
        self.key = key
        self.binary_name = binary_name
        self.url = url
        self.sha256 = sha256


_PLATFORM_SPECS: dict[str, _PlatformSpec] = {
    "linux-x86_64": _PlatformSpec(
        key="linux-x86_64",
        binary_name="astap",
        url="https://github.com/Anseto1988/astap-bin/releases/download/v0.2.1/astap-linux-x86_64.tar.gz",
        sha256="ce3d61573aa14c61276aec7c5b95bd44ec8b21fdb0e19b5011a0295ef2960b7a",
    ),
    "darwin-arm64": _PlatformSpec(
        key="darwin-arm64",
        binary_name="astap",
        url="https://github.com/Anseto1988/astap-bin/releases/download/v0.2.1/astap-darwin-arm64.tar.gz",
        sha256="e8b6ab8c16a1f22693d244bf8c160dbc784d3bb4f51b8a0cb6b9a1fb60db4c84",
    ),
    "win32-x86_64": _PlatformSpec(
        key="win32-x86_64",
        binary_name="astap.exe",
        url="https://github.com/Anseto1988/astap-bin/releases/download/v0.2.1/astap-win32-x86_64.zip",
        sha256="6820c42ade89fe109e8bfcf185a8e2d87c9123ec62991bbacdb8dc15f575fcf6",
    ),
}


def _detect_platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine in ("x86_64", "amd64"):
        return "linux-x86_64"
    if system == "darwin" and machine in ("arm64", "aarch64"):
        return "darwin-arm64"
    if system == "darwin" and machine in ("x86_64", "amd64"):
        return "darwin-arm64"
    if system == "windows" and machine in ("amd64", "x86_64"):
        return "win32-x86_64"

    raise AstapNotFoundError(
        f"Unsupported platform: {system}-{machine}. "
        f"Supported: {', '.join(_PLATFORM_SPECS.keys())}"
    )


def _get_spec() -> _PlatformSpec:
    return _PLATFORM_SPECS[_detect_platform_key()]


def _find_in_path() -> Path | None:
    spec = _get_spec()
    found = shutil.which(spec.binary_name)
    if found:
        return Path(found)
    if spec.binary_name == "astap.exe":
        found = shutil.which("astap")
        if found:
            return Path(found)
    return None


def _bundled_path() -> Path:
    spec = _get_spec()
    return _BUNDLED_DIR / spec.key / spec.binary_name


def _user_path() -> Path:
    spec = _get_spec()
    return _USER_DIR / spec.key / spec.binary_name


def _is_executable(path: Path) -> bool:
    if not path.is_file():
        return False
    if platform.system().lower() == "windows":
        return True
    return os.access(path, os.X_OK)


def get_astap_path() -> Path:
    """Resolve the ASTAP binary path. Checks bundled, user dir, then system PATH."""
    env_path = os.environ.get("ASTAP_BINARY_PATH")
    if env_path:
        p = Path(env_path)
        if _is_executable(p):
            return p
        raise AstapNotFoundError(f"ASTAP_BINARY_PATH set but not executable: {p}")

    bundled = _bundled_path()
    if _is_executable(bundled):
        return bundled

    user = _user_path()
    if _is_executable(user):
        return user

    system = _find_in_path()
    if system is not None:
        return system

    raise AstapNotFoundError(
        "ASTAP binary not found. Run 'python -m astroai.engine.platesolving.astap_binary' "
        "to download, or set ASTAP_BINARY_PATH."
    )


def _verify_sha256(path: Path, expected: str) -> bool:
    if expected.startswith("placeholder"):
        return True
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 * 1024):
            sha.update(chunk)
    return sha.hexdigest() == expected


def _extract_archive(archive: Path, dest_dir: Path) -> None:
    import zipfile
    import tarfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()

    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest_dir)  # noqa: S202
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    else:
        shutil.copy2(archive, dest_dir)


def _make_executable(path: Path) -> None:
    if platform.system().lower() != "windows":
        st = path.stat()
        path.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def download_astap(target_dir: Path | None = None) -> Path:
    """Download the ASTAP binary for the current platform."""
    spec = _get_spec()
    dest_dir = target_dir or (_USER_DIR / spec.key)
    dest_dir.mkdir(parents=True, exist_ok=True)
    binary_path = dest_dir / spec.binary_name

    if _is_executable(binary_path):
        logger.info("ASTAP already present at %s", binary_path)
        return binary_path

    logger.info("Downloading ASTAP for %s from %s", spec.key, spec.url)

    tmp = tempfile.NamedTemporaryFile(
        dir=dest_dir, suffix=".download", delete=False
    )
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        req = urllib.request.Request(spec.url)
        with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        logger.info("Download progress: %d%%", pct)

        if not _verify_sha256(tmp_path, spec.sha256):
            raise RuntimeError(f"SHA256 mismatch for ASTAP download ({spec.key})")

        _extract_archive(tmp_path, dest_dir)

        if not binary_path.exists():
            for candidate in dest_dir.rglob(spec.binary_name):
                shutil.move(str(candidate), str(binary_path))
                break

        _make_executable(binary_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not _is_executable(binary_path):
        raise RuntimeError(f"ASTAP binary not executable after download: {binary_path}")

    logger.info("ASTAP binary ready at %s", binary_path)
    return binary_path


def ensure_astap() -> Path:
    """Return ASTAP path, downloading if necessary."""
    try:
        return get_astap_path()
    except AstapNotFoundError:
        return download_astap()


def verify_astap(path: Path | None = None) -> str | None:
    """Run 'astap --version' and return version string, or None on failure."""
    try:
        binary = path or get_astap_path()
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (result.stdout or result.stderr).strip()
        return output if output else None
    except (AstapNotFoundError, subprocess.SubprocessError, OSError):
        return None


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        p = ensure_astap()
        print(f"ASTAP binary: {p}")
        version = verify_astap(p)
        if version:
            print(f"Version: {version}")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
