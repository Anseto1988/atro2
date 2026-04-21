"""Benchmark metrics emitted during GPU/CPU batch calibration."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

__all__ = ["BenchmarkBackend", "BenchmarkMetrics", "MetricsCallback"]


class BenchmarkBackend(Enum):
    CPU = "CPU"
    CUDA = "CUDA"
    MPS = "MPS"


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    backend: BenchmarkBackend
    device_name: str
    frames_per_second: float
    speedup_factor: float
    current_frame: int
    total_frames: int
    eta_seconds: float


MetricsCallback = Callable[[BenchmarkMetrics], None]
