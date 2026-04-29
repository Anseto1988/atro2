"""NAFNet ONNX-backed tile-based denoiser."""
from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from astroai.core.onnx_registry import OnnxModelRegistry
from astroai.core.pipeline.base import ProgressCallback

__all__ = ["NAFNetDenoiser"]

logger = logging.getLogger(__name__)

_MODEL_NAME = "nafnet_denoise"


class NAFNetDenoiser:
    """Tile-based NAFNet ONNX denoiser backed by OnnxModelRegistry."""

    MODEL_NAME = _MODEL_NAME

    def __init__(
        self,
        strength: float = 1.0,
        tile_size: int = 512,
        tile_overlap: int = 64,
        registry: OnnxModelRegistry | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self._strength = float(np.clip(strength, 0.0, 1.0))
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._registry = registry if registry is not None else OnnxModelRegistry()
        self._progress = progress
        self._session: Any | None = None

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        self._strength = float(np.clip(value, 0.0, 1.0))

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @property
    def tile_overlap(self) -> int:
        return self._tile_overlap

    @property
    def is_model_loaded(self) -> bool:
        return self._session is not None

    @property
    def backend_label(self) -> str:
        return self._registry.backend_label

    def load(self) -> None:
        """Eagerly load the ONNX session (lazy if not called)."""
        self._session = self._registry.get_session(
            self.MODEL_NAME,
            fallback_to_dummy=True,
            progress=self._progress,
        )

    def _get_session(self) -> Any:
        if self._session is None:
            self.load()
        return self._session

    def denoise(self, frame: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Denoise frame (H,W) or (H,W,C). Preserves shape and dtype."""
        original_dtype = frame.dtype
        img = frame.astype(np.float64)

        if img.ndim == 3:
            channels = [img[..., c] for c in range(img.shape[2])]
            denoised = [self._denoise_channel(ch) for ch in channels]
            result = np.stack(denoised, axis=-1)
        else:
            result = self._denoise_channel(img)

        blended = img + self._strength * (result - img)
        return cast(NDArray[np.floating[Any]], blended.astype(original_dtype))

    def _denoise_channel(self, channel: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        session = self._get_session()
        h, w = channel.shape
        vmin = float(channel.min())
        vmax = float(channel.max())
        rng = vmax - vmin if vmax > vmin else 1.0
        normalized = ((channel - vmin) / rng).astype(np.float32)

        ts = self._tile_size
        step = max(1, ts - self._tile_overlap)
        result = np.zeros((h, w), dtype=np.float64)
        weight_map = np.zeros((h, w), dtype=np.float64)
        input_name = session.get_inputs()[0].name

        for y in range(0, h, step):
            for x in range(0, w, step):
                y1 = min(y + ts, h)
                x1 = min(x + ts, w)
                y0 = max(y1 - ts, 0)
                x0 = max(x1 - ts, 0)

                tile = normalized[y0:y1, x0:x1]
                ph = ts - tile.shape[0]
                pw = ts - tile.shape[1]
                if ph > 0 or pw > 0:
                    tile = np.pad(tile, ((0, ph), (0, pw)), mode="reflect")

                inp = tile[np.newaxis, np.newaxis, :, :]
                out = session.run(None, {input_name: inp})[0]
                out_crop = out.squeeze()[:y1 - y0, :x1 - x0]
                result[y0:y1, x0:x1] += out_crop.astype(np.float64)
                weight_map[y0:y1, x0:x1] += 1.0

        weight_map = np.maximum(weight_map, 1.0)
        result /= weight_map
        return cast(NDArray[np.floating[Any]], np.clip(result * rng + vmin, vmin, vmax))
