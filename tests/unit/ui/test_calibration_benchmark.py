"""Tests for CalibrationBenchmarkWidget."""
from __future__ import annotations

import pytest

from astroai.core.calibration.metrics import BenchmarkBackend, BenchmarkMetrics
from astroai.ui.widgets.calibration_benchmark import CalibrationBenchmarkWidget


def _metrics(
    backend=BenchmarkBackend.CUDA,
    device_name="RTX 4090",
    fps=10.0,
    speedup=5.0,
    current=50,
    total=100,
    eta=30.0,
) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        backend=backend,
        device_name=device_name,
        frames_per_second=fps,
        speedup_factor=speedup,
        current_frame=current,
        total_frames=total,
        eta_seconds=eta,
    )


@pytest.fixture()
def widget(qtbot):
    w = CalibrationBenchmarkWidget()
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, widget):
        assert widget is not None

    def test_hidden_initially(self, widget):
        assert widget.isHidden()

    def test_accessible_name(self, widget):
        assert widget.accessibleName() == "Kalibrierungs-Benchmark"

    def test_progress_bar_starts_at_zero(self, widget):
        assert widget._bar.value() == 0


class TestStart:
    def test_start_shows_widget(self, widget):
        widget.start()
        assert not widget.isHidden()

    def test_start_resets_bar_to_zero(self, widget):
        widget._bar.setValue(500)
        widget.start()
        assert widget._bar.value() == 0

    def test_start_clears_eta_label(self, widget):
        widget._eta_label.setText("ETA: 30s")
        widget.start()
        assert widget._eta_label.text() == ""

    def test_start_sets_metric_dashes(self, widget):
        widget.start()
        assert widget._fps_metric._value.text() == "—"
        assert widget._speedup_metric._value.text() == "—"
        assert widget._frames_metric._value.text() == "—"


class TestFinish:
    def test_finish_sets_bar_to_max(self, widget):
        widget.finish()
        assert widget._bar.value() == 1000

    def test_finish_sets_eta_fertig(self, widget):
        widget.finish()
        assert widget._eta_label.text() == "Fertig"

    def test_finish_sets_format_100(self, widget):
        widget.finish()
        assert widget._bar.format() == "100%"


class TestReset:
    def test_reset_hides_widget(self, widget):
        widget.start()
        widget.reset()
        assert widget.isHidden()

    def test_reset_clears_bar(self, widget):
        widget._bar.setValue(500)
        widget.reset()
        assert widget._bar.value() == 0

    def test_reset_clears_eta(self, widget):
        widget._eta_label.setText("ETA: 5s")
        widget.reset()
        assert widget._eta_label.text() == ""

    def test_reset_resets_metrics_to_dash(self, widget):
        widget.reset()
        assert widget._fps_metric._value.text() == "—"


class TestUpdateMetrics:
    def test_update_sets_fps(self, widget):
        widget.update_metrics(_metrics(fps=15.5))
        assert widget._fps_metric._value.text() == "15.5"

    def test_update_cuda_sets_speedup_gpu(self, widget):
        widget.update_metrics(_metrics(backend=BenchmarkBackend.CUDA, speedup=4.2))
        assert "GPU" in widget._speedup_metric._value.text()
        assert "4.2" in widget._speedup_metric._value.text()

    def test_update_cpu_sets_speedup_1x(self, widget):
        widget.update_metrics(_metrics(backend=BenchmarkBackend.CPU, speedup=1.0))
        assert widget._speedup_metric._value.text() == "1.0\u00d7"

    def test_update_sets_frames_fraction(self, widget):
        widget.update_metrics(_metrics(current=25, total=100))
        assert widget._frames_metric._value.text() == "25/100"

    def test_update_sets_progress_bar(self, widget):
        widget.update_metrics(_metrics(current=50, total=100))
        assert widget._bar.value() == 500

    def test_update_zero_total_no_crash(self, widget):
        widget.update_metrics(_metrics(current=0, total=0))

    def test_update_sets_eta_seconds(self, widget):
        widget.update_metrics(_metrics(eta=45.0))
        assert "45s" in widget._eta_label.text()

    def test_update_sets_badge_backend(self, widget):
        widget.update_metrics(_metrics(backend=BenchmarkBackend.CUDA, device_name="RTX 4090"))
        assert "RTX 4090" in widget._badge.text()

    def test_update_mps_backend(self, widget):
        widget.update_metrics(_metrics(backend=BenchmarkBackend.MPS, device_name="Apple M2"))
        assert "Apple M2" in widget._badge.text()


class TestFormatEta:
    def test_zero_eta_returns_empty(self):
        assert CalibrationBenchmarkWidget._format_eta(0) == ""

    def test_negative_eta_returns_empty(self):
        assert CalibrationBenchmarkWidget._format_eta(-5) == ""

    def test_seconds_only(self):
        result = CalibrationBenchmarkWidget._format_eta(45)
        assert "45s" in result

    def test_minutes_and_seconds(self):
        result = CalibrationBenchmarkWidget._format_eta(90)
        assert "1m" in result
        assert "30s" in result

    def test_exactly_60_seconds(self):
        result = CalibrationBenchmarkWidget._format_eta(60)
        assert "1m" in result
        assert "00s" in result
