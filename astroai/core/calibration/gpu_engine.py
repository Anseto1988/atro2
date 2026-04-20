from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray

from astroai.core.calibration.matcher import CalibrationLibrary, find_best_dark, find_best_flat
from astroai.core.io.fits_io import ImageMetadata
from astroai.inference.backends.gpu import DeviceManager

LoadDataFn = Callable[..., NDArray[np.floating[Any]]]


class GPUCalibrationEngine:
    """GPU-accelerated calibration via PyTorch (CUDA / MPS / CPU fallback)."""

    def __init__(self) -> None:
        self._device = DeviceManager().get_device()

    def _to_tensor(self, arr: NDArray[np.floating[Any]]) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32)).to(self._device)

    def _to_numpy(self, t: torch.Tensor) -> NDArray[np.float32]:
        return t.cpu().numpy()

    def apply_dark_gpu(
        self,
        light_t: torch.Tensor,
        dark_t: torch.Tensor,
    ) -> torch.Tensor:
        return torch.clamp(light_t - dark_t, min=0.0)

    def apply_flat_gpu(
        self,
        light_t: torch.Tensor,
        flat_t: torch.Tensor,
    ) -> torch.Tensor:
        flat_norm = flat_t / torch.clamp(flat_t.median(), min=1e-7)
        return light_t / torch.clamp(flat_norm, min=1e-7)

    def calibrate_frame_gpu(
        self,
        light: NDArray[np.floating[Any]],
        light_meta: ImageMetadata,
        library: CalibrationLibrary,
        load_data: LoadDataFn | None = None,
    ) -> NDArray[np.float32]:
        result_t = self._to_tensor(light)

        dark_frame = find_best_dark(light_meta, library)
        if dark_frame is not None:
            dark_data = dark_frame.data
            if dark_data is None and load_data is not None:
                dark_data = load_data(dark_frame.path)
            if dark_data is not None:
                result_t = self.apply_dark_gpu(result_t, self._to_tensor(dark_data))

        flat_frame = find_best_flat(light_meta, library)
        if flat_frame is not None:
            flat_data = flat_frame.data
            if flat_data is None and load_data is not None:
                flat_data = load_data(flat_frame.path)
            if flat_data is not None:
                result_t = self.apply_flat_gpu(result_t, self._to_tensor(flat_data))

        return self._to_numpy(result_t)

    def calibrate_batch_gpu(
        self,
        frames: list[NDArray[np.floating[Any]]],
        light_meta: ImageMetadata,
        library: CalibrationLibrary,
        load_data: LoadDataFn | None = None,
    ) -> list[NDArray[np.float32]]:
        """Calibrate multiple frames; dark/flat tensors are reused across batch."""
        dark_t: torch.Tensor | None = None
        flat_t: torch.Tensor | None = None

        dark_frame = find_best_dark(light_meta, library)
        if dark_frame is not None:
            dark_data = dark_frame.data
            if dark_data is None and load_data is not None:
                dark_data = load_data(dark_frame.path)
            if dark_data is not None:
                dark_t = self._to_tensor(dark_data)

        flat_frame = find_best_flat(light_meta, library)
        if flat_frame is not None:
            flat_data = flat_frame.data
            if flat_data is None and load_data is not None:
                flat_data = load_data(flat_frame.path)
            if flat_data is not None:
                flat_t = self._to_tensor(flat_data)
                flat_norm = flat_t / torch.clamp(flat_t.median(), min=1e-7)
                flat_t = torch.clamp(flat_norm, min=1e-7)

        results: list[NDArray[np.float32]] = []
        for frame in frames:
            t = self._to_tensor(frame)
            if dark_t is not None:
                t = torch.clamp(t - dark_t, min=0.0)
            if flat_t is not None:
                t = t / flat_t
            results.append(self._to_numpy(t))

        return results

    @property
    def device_type(self) -> str:
        return self._device.type
