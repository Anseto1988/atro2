from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

__all__ = ["ModelRegistry"]


class ModelRegistry:
    """Thread-safe registry for loading and caching AI models."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}
        self._lock = threading.Lock()

    def _load(self, path: Path) -> Any:
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            if ort is None:
                raise RuntimeError(
                    "onnxruntime is required to load .onnx models"
                )
            return ort.InferenceSession(str(path))
        if suffix == ".pth":
            return torch.load(
                path, map_location="cpu", weights_only=False
            )
        raise ValueError(f"Unsupported model format: {suffix}")

    def register(self, name: str, path: Path) -> None:
        with self._lock:
            self._paths[name] = path
            self._models[name] = self._load(path)

    def get(self, name: str) -> Any:
        with self._lock:
            if name not in self._models:
                raise KeyError(f"Model '{name}' is not registered")
            return self._models[name]

    def reload(self, name: str) -> None:
        with self._lock:
            if name not in self._paths:
                raise KeyError(f"Model '{name}' is not registered")
            self._models[name] = self._load(self._paths[name])

    def unregister(self, name: str) -> None:
        with self._lock:
            self._paths.pop(name, None)
            self._models.pop(name, None)

    def list_models(self) -> list[str]:
        with self._lock:
            return list(self._models.keys())
