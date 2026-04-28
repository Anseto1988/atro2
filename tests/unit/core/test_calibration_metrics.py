"""Unit tests for BenchmarkMetrics dataclass and BenchmarkBackend enum."""
from __future__ import annotations

import pytest

from astroai.core.calibration.metrics import (
    BenchmarkBackend,
    BenchmarkMetrics,
    MetricsCallback,
)


class TestBenchmarkBackend:
    def test_cpu_value(self) -> None:
        assert BenchmarkBackend.CPU.value == "CPU"

    def test_cuda_value(self) -> None:
        assert BenchmarkBackend.CUDA.value == "CUDA"

    def test_mps_value(self) -> None:
        assert BenchmarkBackend.MPS.value == "MPS"

    def test_members_count(self) -> None:
        assert len(BenchmarkBackend) == 3


class TestBenchmarkMetrics:
    def _make(self, **overrides: object) -> BenchmarkMetrics:
        defaults = dict(
            backend=BenchmarkBackend.CPU,
            device_name="Intel i9",
            frames_per_second=12.5,
            speedup_factor=1.0,
            current_frame=3,
            total_frames=10,
            eta_seconds=5.6,
        )
        defaults.update(overrides)
        return BenchmarkMetrics(**defaults)  # type: ignore[arg-type]

    def test_fields_stored(self) -> None:
        m = self._make()
        assert m.backend == BenchmarkBackend.CPU
        assert m.device_name == "Intel i9"
        assert m.frames_per_second == pytest.approx(12.5)
        assert m.speedup_factor == pytest.approx(1.0)
        assert m.current_frame == 3
        assert m.total_frames == 10
        assert m.eta_seconds == pytest.approx(5.6)

    def test_frozen(self) -> None:
        m = self._make()
        with pytest.raises(AttributeError):
            m.current_frame = 99  # type: ignore[misc]

    def test_cuda_backend(self) -> None:
        m = self._make(backend=BenchmarkBackend.CUDA, device_name="RTX 5090")
        assert m.backend == BenchmarkBackend.CUDA
        assert m.device_name == "RTX 5090"

    def test_speedup_above_one(self) -> None:
        m = self._make(speedup_factor=8.3)
        assert m.speedup_factor == pytest.approx(8.3)

    def test_eta_zero_at_completion(self) -> None:
        m = self._make(current_frame=10, total_frames=10, eta_seconds=0.0)
        assert m.eta_seconds == pytest.approx(0.0)
        assert m.current_frame == m.total_frames


class TestMetricsCallbackType:
    def test_callable_accepted(self) -> None:
        received: list[BenchmarkMetrics] = []

        def cb(m: BenchmarkMetrics) -> None:
            received.append(m)

        callback: MetricsCallback = cb
        m = BenchmarkMetrics(
            backend=BenchmarkBackend.CPU,
            device_name="x",
            frames_per_second=1.0,
            speedup_factor=1.0,
            current_frame=1,
            total_frames=1,
            eta_seconds=0.0,
        )
        callback(m)
        assert len(received) == 1
        assert received[0] is m
