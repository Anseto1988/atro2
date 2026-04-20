"""Tests for CalibrationBenchmarkWidget."""
from __future__ import annotations

import pytest

from astroai.ui.widgets.calibration_benchmark import (
    BenchmarkBackend,
    BenchmarkMetrics,
    CalibrationBenchmarkWidget,
)


class TestCalibrationBenchmarkWidget:
    @pytest.fixture()
    def widget(self, qtbot) -> CalibrationBenchmarkWidget:  # type: ignore[no-untyped-def]
        w = CalibrationBenchmarkWidget()
        qtbot.addWidget(w)
        return w

    def test_initially_hidden(self, widget: CalibrationBenchmarkWidget) -> None:
        assert not widget.isVisible()

    def test_start_shows_widget(self, widget: CalibrationBenchmarkWidget) -> None:
        widget.start()
        assert widget.isVisible()

    def test_reset_hides_widget(self, widget: CalibrationBenchmarkWidget) -> None:
        widget.start()
        widget.reset()
        assert not widget.isVisible()

    def test_update_metrics_cuda(self, widget: CalibrationBenchmarkWidget) -> None:
        widget.start()
        metrics = BenchmarkMetrics(
            backend=BenchmarkBackend.CUDA,
            device_name="RTX 4090",
            frames_per_second=42.5,
            speedup_factor=7.2,
            current_frame=15,
            total_frames=100,
            eta_seconds=120.0,
        )
        widget.update_metrics(metrics)
        assert widget._badge.text() == "CUDA (RTX 4090)"
        assert widget._fps_metric._value.text() == "42.5"
        assert "7.2" in widget._speedup_metric._value.text()
        assert widget._frames_metric._value.text() == "15/100"
        assert widget._bar.value() == 150
        assert "2m" in widget._eta_label.text()

    def test_update_metrics_mps(self, widget: CalibrationBenchmarkWidget) -> None:
        widget.start()
        metrics = BenchmarkMetrics(
            backend=BenchmarkBackend.MPS,
            device_name="Apple M2",
            frames_per_second=18.3,
            speedup_factor=3.1,
            current_frame=50,
            total_frames=50,
            eta_seconds=0.0,
        )
        widget.update_metrics(metrics)
        assert widget._badge.text() == "MPS (Apple M2)"
        assert "3.1" in widget._speedup_metric._value.text()
        assert widget._bar.value() == 1000

    def test_update_metrics_cpu_shows_1x(self, widget: CalibrationBenchmarkWidget) -> None:
        widget.start()
        metrics = BenchmarkMetrics(
            backend=BenchmarkBackend.CPU,
            device_name="",
            frames_per_second=5.8,
            speedup_factor=1.0,
            current_frame=3,
            total_frames=20,
            eta_seconds=45.0,
        )
        widget.update_metrics(metrics)
        assert widget._speedup_metric._value.text() == "1.0\u00d7"
        assert "45s" in widget._eta_label.text()

    def test_finish_sets_complete(self, widget: CalibrationBenchmarkWidget) -> None:
        widget.start()
        widget.finish()
        assert widget._bar.value() == 1000
        assert widget._eta_label.text() == "Fertig"

    def test_eta_formatting_minutes(self, widget: CalibrationBenchmarkWidget) -> None:
        result = CalibrationBenchmarkWidget._format_eta(185.0)
        assert result == "ETA: 3m 05s"

    def test_eta_formatting_seconds(self, widget: CalibrationBenchmarkWidget) -> None:
        result = CalibrationBenchmarkWidget._format_eta(30.0)
        assert result == "ETA: 30s"

    def test_eta_formatting_zero(self, widget: CalibrationBenchmarkWidget) -> None:
        result = CalibrationBenchmarkWidget._format_eta(0.0)
        assert result == ""

    def test_accessible_name(self, widget: CalibrationBenchmarkWidget) -> None:
        assert widget.accessibleName() == "Kalibrierungs-Benchmark"
