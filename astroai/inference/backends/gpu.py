from __future__ import annotations

from typing import Any

import torch

__all__ = ["DeviceManager"]


class DeviceManager:
    """Singleton GPU/accelerator abstraction for device selection."""

    _instance: DeviceManager | None = None
    _device: torch.device

    def __new__(cls) -> DeviceManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._device = cls._resolve_device()
        return cls._instance

    @staticmethod
    def _resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_device(self) -> torch.device:
        return self._device

    def device_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "type": self._device.type,
            "name": self._device.type,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }
        if self._device.type == "cuda":
            idx = self._device.index or 0
            info["name"] = torch.cuda.get_device_name(idx)
            mem = torch.cuda.get_device_properties(idx).total_memory
            info["vram_bytes"] = mem
            info["vram_gb"] = round(mem / (1024**3), 2)
        return info

    def to_device(
        self, tensor_or_module: torch.Tensor | torch.nn.Module,
    ) -> torch.Tensor | torch.nn.Module:
        return tensor_or_module.to(self._device)
