from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray

from astroai.core.calibration.matcher import CalibrationLibrary, find_best_dark, find_best_flat
from astroai.core.calibration.metrics import BenchmarkBackend, BenchmarkMetrics, MetricsCallback
from astroai.core.io.fits_io import ImageMetadata
from astroai.inference.backends.gpu import DeviceManager

LoadDataFn = Callable[..., NDArray[np.floating[Any]]]

_DEVICE_TO_BACKEND: dict[str, BenchmarkBackend] = {
    "cuda": BenchmarkBackend.CUDA,
    "mps": BenchmarkBackend.MPS,
    "cpu": BenchmarkBackend.CPU,
}


class GPUCalibrationEngine:
    """GPU-accelerated calibration via PyTorch (CUDA / MPS / CPU fallback)."""

    def __init__(self) -> None:
        device = DeviceManager().get_device()
        self._device = self._validate_device(device)

    @staticmethod
    def _validate_device(device: torch.device) -> torch.device:
        """Return device only if a test tensor actually runs on it; else CPU."""
        if device.type == "cpu":
            return device
        try:
            torch.zeros(1, device=device)
            return device
        except Exception:
            return torch.device("cpu")

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
        on_metrics: MetricsCallback | None = None,
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

        total = len(frames)
        backend = _DEVICE_TO_BACKEND.get(self._device.type, BenchmarkBackend.CPU)
        device_name = (
            torch.cuda.get_device_name(self._device)
            if self._device.type == "cuda"
            else self._device.type.upper()
        )

        results: list[NDArray[np.float32]] = []
        t_start = time.perf_counter()
        last_emit = 0.0

        for idx, frame in enumerate(frames):
            t = self._to_tensor(frame)
            if dark_t is not None:
                t = torch.clamp(t - dark_t, min=0.0)
            if flat_t is not None:
                t = t / flat_t
            results.append(self._to_numpy(t))

            if on_metrics is not None:
                now = time.perf_counter()
                elapsed = now - t_start
                if elapsed - last_emit >= 1.0 or idx == total - 1 or idx == 0:
                    last_emit = elapsed
                    done = idx + 1
                    fps = done / max(elapsed, 1e-9)
                    remaining = (total - done) / max(fps, 1e-9) if done < total else 0.0
                    cpu_fps = fps if backend == BenchmarkBackend.CPU else None
                    speedup = fps / cpu_fps if cpu_fps else fps
                    on_metrics(BenchmarkMetrics(
                        backend=backend,
                        device_name=device_name,
                        frames_per_second=fps,
                        speedup_factor=1.0 if backend == BenchmarkBackend.CPU else speedup,
                        current_frame=done,
                        total_frames=total,
                        eta_seconds=remaining,
                    ))

        return results

    @property
    def device_type(self) -> str:
        return self._device.type
