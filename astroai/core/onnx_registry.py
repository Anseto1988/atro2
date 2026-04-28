"""Central ONNX model registry with versioned caching and provider selection."""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

from astroai.core.pipeline.base import PipelineProgress, PipelineStage, ProgressCallback

__all__ = ["OnnxModelRegistry", "ModelSpec"]

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".astroai" / "models"

_BUILTIN_MODELS: dict[str, dict[str, str]] = {
    "nafnet_denoise": {
        "url": "https://github.com/AstroAI-Suite/models/releases/download/v0.1.0/nafnet_denoise.onnx",
        "sha256": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "filename": "nafnet_denoise.onnx",
        "version": "0.1.0",
    },
    "starnet_plus_plus": {
        "url": "https://github.com/AstroAI-Suite/models/releases/download/v0.1.0/starnet_plus_plus.onnx",
        "sha256": "placeholder_no_public_url_available",
        "filename": "starnet_plus_plus.onnx",
        "version": "0.1.0",
        "available": "false",
    },
    "blind_deconv": {
        "url": "https://github.com/AstroAI-Suite/models/releases/download/v0.1.0/blind_deconv.onnx",
        "sha256": "placeholder_blind_deconv",
        "filename": "blind_deconv.onnx",
        "version": "0.1.0",
        "available": "false",
    },
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    filename: str
    version: str = "0.1.0"
    available: bool = True


class _DummyOnnxSession:
    """Identity pass-through when onnxruntime is unavailable."""

    def get_inputs(self) -> list[Any]:
        return [type("Input", (), {"name": "input"})()]

    def run(
        self, output_names: Any, input_dict: dict[str, Any]
    ) -> list[np.ndarray[Any, Any]]:
        key = next(iter(input_dict))
        return [input_dict[key].copy()]


class OnnxModelRegistry:
    """Thread-safe ONNX model registry with versioned cache and provider detection.

    Usage::

        registry = OnnxModelRegistry()
        session = registry.get_session("nafnet_denoise")
        print(registry.backend_label)  # "[GPU]" / "[CPU]" / "[ONNX]"
    """

    _instance: OnnxModelRegistry | None = None
    _init_done: bool = False

    def __new__(cls, cache_dir: Path | None = None) -> OnnxModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cache_dir: Path | None = None) -> None:
        if self._init_done:
            return
        self._lock = threading.Lock()
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._sessions: dict[str, Any] = {}
        self._specs: dict[str, ModelSpec] = {}
        self._progress: ProgressCallback | None = None
        self._providers = self._detect_providers()
        self._register_builtins()
        OnnxModelRegistry._init_done = True

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — for testing only."""
        cls._instance = None
        cls._init_done = False

    def set_progress(self, callback: ProgressCallback | None) -> None:
        self._progress = callback

    def _register_builtins(self) -> None:
        for name, info in _BUILTIN_MODELS.items():
            self._specs[name] = ModelSpec(
                name=name,
                url=info["url"],
                sha256=info["sha256"],
                filename=info["filename"],
                version=info.get("version", "0.1.0"),
                available=info.get("available", "true") != "false",
            )

    def register(self, spec: ModelSpec) -> None:
        with self._lock:
            self._specs[spec.name] = spec

    def list_models(self) -> list[str]:
        with self._lock:
            return list(self._specs.keys())

    def is_available(self, name: str) -> bool:
        with self._lock:
            spec = self._specs.get(name)
            if spec is None or not spec.available:
                return False
            path = self._model_path(spec)
            return path.exists() and self._verify_sha256(path, spec.sha256)

    def get_session(
        self,
        name: str,
        *,
        fallback_to_dummy: bool = True,
        progress: ProgressCallback | None = None,
    ) -> Any:
        with self._lock:
            if name in self._sessions:
                return self._sessions[name]

        session = self._load_session(name, fallback_to_dummy, progress)
        with self._lock:
            self._sessions[name] = session
        return session

    def load_from_path(self, path: str | Path) -> Any | None:
        model_path = Path(path)
        if not model_path.exists():
            logger.debug("ONNX model path does not exist: %s", path)
            return None
        if ort is None:
            logger.warning("onnxruntime not installed")
            return None
        try:
            return ort.InferenceSession(str(model_path), providers=self._providers)
        except Exception as exc:
            logger.debug("Failed to load ONNX from path (%s): %s", path, exc)
            return None

    def evict(self, name: str) -> None:
        with self._lock:
            self._sessions.pop(name, None)

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    @property
    def backend_label(self) -> str:
        if ort is None:
            return "[CPU]"
        for p in self._providers:
            if "CUDA" in p or "Dml" in p or "TensorRT" in p:
                return "[GPU]"
        return "[ONNX]"

    def _model_path(self, spec: ModelSpec) -> Path:
        return self._cache_dir / spec.name / spec.version / spec.filename

    def _load_session(
        self,
        name: str,
        fallback_to_dummy: bool,
        progress: ProgressCallback | None,
    ) -> Any:
        spec = self._specs.get(name)
        if spec is None:
            if fallback_to_dummy:
                logger.warning("Unknown model '%s', using dummy fallback", name)
                return _DummyOnnxSession()
            raise KeyError(f"Unknown model: {name}")

        if ort is None:
            if fallback_to_dummy:
                logger.warning("onnxruntime not installed, using dummy fallback")
                return _DummyOnnxSession()
            raise RuntimeError("onnxruntime is required but not installed")

        try:
            path = self._ensure_model(spec, progress)
            return ort.InferenceSession(str(path), providers=self._providers)
        except Exception as exc:
            if fallback_to_dummy:
                logger.warning("Failed to load '%s' (%s), using dummy", name, exc)
                return _DummyOnnxSession()
            raise

    def _ensure_model(self, spec: ModelSpec, progress: ProgressCallback | None) -> Path:
        target = self._model_path(spec)
        if target.exists() and self._verify_sha256(target, spec.sha256):
            logger.info("Model '%s' cached at %s", spec.name, target)
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        cb = progress or self._progress
        self._download(spec, target, cb)

        if not self._verify_sha256(target, spec.sha256):
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for '{spec.name}'. File deleted."
            )
        logger.info("Model '%s' downloaded and verified at %s", spec.name, target)
        return target

    def _download(self, spec: ModelSpec, target: Path, progress: ProgressCallback | None) -> None:
        import shutil
        import tempfile
        import urllib.request
        from urllib.parse import urlparse

        parsed = urlparse(spec.url)
        if parsed.scheme.lower() != "https":
            raise ValueError(f"Refusing non-HTTPS URL: {spec.url!r}")

        self._report(progress, f"Downloading {spec.name}", 0, 100)
        tmp = tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        try:
            with urllib.request.urlopen(
                urllib.request.Request(spec.url), timeout=120
            ) as resp:
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
                            pct = min(int(downloaded * 100 / total), 99)
                            self._report(progress, f"Downloading {spec.name}", pct, 100)
            shutil.move(str(tmp_path), str(target))
            self._report(progress, f"Download complete: {spec.name}", 100, 100)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _verify_sha256(path: Path, expected: str) -> bool:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest() == expected

    @staticmethod
    def _report(cb: ProgressCallback | None, message: str, current: int, total: int) -> None:
        if cb:
            cb(PipelineProgress(
                stage=PipelineStage.LOADING,
                current=current,
                total=total,
                message=message,
            ))

    @staticmethod
    def _detect_providers() -> list[str]:
        if ort is None:
            return ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        preferred = [
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        result = [p for p in preferred if p in available]
        return result or ["CPUExecutionProvider"]
