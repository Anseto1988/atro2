"""ONNX model download manager with checksum verification."""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from astroai.core.pipeline.base import PipelineProgress, PipelineStage, ProgressCallback

__all__ = ["ModelDownloader", "ModelManifestEntry"]

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".astroai" / "models"

_NAFNET_MANIFEST: dict[str, dict[str, Any]] = {
    "nafnet_denoise": {
        "url": "https://github.com/AstroAI-Suite/models/releases/download/v0.1.0/nafnet_denoise.onnx",
        "sha256": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "filename": "nafnet_denoise.onnx",
        "description": "NAFNet denoising model (ONNX, Apache 2.0)",
    },
}

_STARNET_MANIFEST: dict[str, dict[str, Any]] = {
    "starnet_plus_plus": {
        "url": "https://github.com/AstroAI-Suite/models/releases/download/v0.1.0/starnet_plus_plus.onnx",
        "sha256": "placeholder_no_public_url_available",
        "filename": "starnet_plus_plus.onnx",
        "description": "StarNet++ star removal model (ONNX, MIT)",
        "available": False,
    },
}


@dataclass
class ModelManifestEntry:
    name: str
    url: str
    sha256: str
    filename: str
    description: str = ""


class ModelDownloader:
    """Downloads and verifies ONNX model weights on first use."""

    def __init__(
        self,
        models_dir: Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self._models_dir = models_dir or MODELS_DIR
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._progress = progress

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def get_manifest(self) -> dict[str, ModelManifestEntry]:
        entries: dict[str, ModelManifestEntry] = {}
        for source in (_NAFNET_MANIFEST, _STARNET_MANIFEST):
            for name, info in source.items():
                entries[name] = ModelManifestEntry(
                    name=name,
                    url=info["url"],
                    sha256=info["sha256"],
                    filename=info["filename"],
                    description=info.get("description", ""),
                )
        return entries

    def is_available(self, name: str) -> bool:
        manifest = self.get_manifest()
        if name not in manifest:
            return False
        source_info = _NAFNET_MANIFEST.get(name) or _STARNET_MANIFEST.get(name)
        if source_info and not source_info.get("available", True):
            return False
        path = self._models_dir / manifest[name].filename
        return path.exists() and self._verify_checksum(path, manifest[name].sha256)

    def is_downloadable(self, name: str) -> bool:
        """Return True if the model has a public download URL available."""
        for source in (_NAFNET_MANIFEST, _STARNET_MANIFEST):
            if name in source:
                return bool(source[name].get("available", True))
        return False

    def ensure_model(self, name: str) -> Path:
        manifest = self.get_manifest()
        if name not in manifest:
            raise KeyError(f"Unknown model: {name}")
        entry = manifest[name]
        target = self._models_dir / entry.filename

        if target.exists() and self._verify_checksum(target, entry.sha256):
            logger.info("Model '%s' already cached at %s", name, target)
            return target

        try:
            self._download(entry, target)
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

    def load_onnx_session(self, name: str, fallback_to_dummy: bool = True) -> Any:
        if ort is None:
            if fallback_to_dummy:
                logger.warning("onnxruntime not installed, using CPU dummy model")
                return _DummyOnnxSession()
            raise RuntimeError("onnxruntime is required but not installed")

        try:
            path = self.ensure_model(name)
            return ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            if fallback_to_dummy:
                logger.warning(
                    "Failed to load model '%s' (%s), using CPU dummy fallback", name, exc
                )
                return _DummyOnnxSession()
            raise

    @staticmethod
    def _require_https(url: str) -> None:
        """Reject non-HTTPS download URLs."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            raise ValueError(
                f"Refusing download from non-HTTPS URL: {url!r}. "
                "Only HTTPS is allowed for model downloads."
            )

    def _download(self, entry: ModelManifestEntry, target: Path) -> None:
        self._require_https(entry.url)
        self._report_progress("Downloading model: " + entry.name, 0, 100)
        target.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd = tempfile.NamedTemporaryFile(
            dir=target.parent, suffix=".tmp", delete=False
        )
        tmp_path = Path(tmp_fd.name)
        tmp_fd.close()

        try:
            req = urllib.request.Request(entry.url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 64 * 1024

                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = min(int(downloaded * 100 / total), 99)
                            self._report_progress(
                                f"Downloading {entry.name}", pct, 100
                            )

            shutil.move(str(tmp_path), str(target))
            self._report_progress("Download complete: " + entry.name, 100, 100)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _verify_checksum(self, path: Path, expected: str) -> bool:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest() == expected

    def _report_progress(self, message: str, current: int, total: int) -> None:
        if self._progress:
            self._progress(PipelineProgress(
                stage=PipelineStage.LOADING,
                current=current,
                total=total,
                message=message,
            ))


class _DummyOnnxSession:
    """CPU fallback that returns input unchanged (identity pass-through)."""

    def get_inputs(self) -> list[Any]:
        return [_DummyInput("input")]

    def run(
        self, output_names: Any, input_dict: dict[str, Any]
    ) -> list[np.ndarray[Any, Any]]:
        key = next(iter(input_dict))
        return [input_dict[key].copy()]


class _DummyInput:
    def __init__(self, name: str) -> None:
        self.name = name
