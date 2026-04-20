"""Benchmark display for GPU vs. CPU calibration progress."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

__all__ = ["CalibrationBenchmarkWidget", "BenchmarkBackend", "BenchmarkMetrics"]


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


class _BackendBadge(QLabel):
    """Colored label indicating active compute backend."""

    _STYLES: dict[BenchmarkBackend, str] = {
        BenchmarkBackend.CUDA: "benchmarkBadgeCuda",
        BenchmarkBackend.MPS: "benchmarkBadgeMps",
        BenchmarkBackend.CPU: "benchmarkBadgeCpu",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("—", parent)
        self.setObjectName("benchmarkBadgeCpu")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(24)
        self.setMinimumWidth(140)

    def set_backend(self, backend: BenchmarkBackend, device_name: str) -> None:
        self.setObjectName(self._STYLES[backend])
        self.setText(f"{backend.value} ({device_name})")
        self.style().unpolish(self)
        self.style().polish(self)


class _MetricLabel(QWidget):
    """Single metric with title and value."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self._title = QLabel(title)
        self._title.setObjectName("benchmarkMetricTitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._value = QLabel("—")
        self._value.setObjectName("benchmarkMetricValue")
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class CalibrationBenchmarkWidget(QWidget):
    """Shows GPU/CPU calibration benchmark: backend, FPS, speedup, ETA."""

    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("calibrationBenchmark")
        self.setAccessibleName("Kalibrierungs-Benchmark")
        self._build_ui()
        self.setVisible(False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._badge = _BackendBadge()
        header.addWidget(self._badge)
        header.addStretch()
        self._eta_label = QLabel("")
        self._eta_label.setObjectName("benchmarkEta")
        header.addWidget(self._eta_label)
        root.addLayout(header)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setAccessibleName("Kalibrierungs-Fortschritt")
        root.addWidget(self._bar)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(16)

        self._fps_metric = _MetricLabel("Frames/s")
        self._speedup_metric = _MetricLabel("Speedup")
        self._frames_metric = _MetricLabel("Frames")

        metrics_layout.addWidget(self._fps_metric)
        metrics_layout.addWidget(self._speedup_metric)
        metrics_layout.addWidget(self._frames_metric)
        metrics_layout.addStretch()

        root.addLayout(metrics_layout)

    @Slot(object)
    def update_metrics(self, metrics: BenchmarkMetrics) -> None:
        self._badge.set_backend(metrics.backend, metrics.device_name)
        self._fps_metric.set_value(f"{metrics.frames_per_second:.1f}")

        if metrics.backend == BenchmarkBackend.CPU:
            self._speedup_metric.set_value("1.0\u00d7")
        else:
            self._speedup_metric.set_value(
                f"{metrics.speedup_factor:.1f}\u00d7 GPU"
            )

        self._frames_metric.set_value(
            f"{metrics.current_frame}/{metrics.total_frames}"
        )

        if metrics.total_frames > 0:
            fraction = metrics.current_frame / metrics.total_frames
            self._bar.setValue(int(fraction * 1000))
            self._bar.setFormat(f"{fraction * 100:.0f}%")

        self._eta_label.setText(self._format_eta(metrics.eta_seconds))

    @Slot()
    def start(self) -> None:
        self._bar.setValue(0)
        self._bar.setFormat("0%")
        self._eta_label.setText("")
        self._fps_metric.set_value("—")
        self._speedup_metric.set_value("—")
        self._frames_metric.set_value("—")
        self.setVisible(True)

    @Slot()
    def finish(self) -> None:
        self._bar.setValue(1000)
        self._bar.setFormat("100%")
        self._eta_label.setText("Fertig")

    @Slot()
    def reset(self) -> None:
        self._bar.setValue(0)
        self._bar.setFormat("")
        self._eta_label.setText("")
        self._badge.set_backend(BenchmarkBackend.CPU, "")
        self._fps_metric.set_value("—")
        self._speedup_metric.set_value("—")
        self._frames_metric.set_value("—")
        self.setVisible(False)

    @staticmethod
    def _format_eta(seconds: float) -> str:
        if seconds <= 0:
            return ""
        if seconds < 60:
            return f"ETA: {seconds:.0f}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"ETA: {minutes}m {secs:02d}s"
