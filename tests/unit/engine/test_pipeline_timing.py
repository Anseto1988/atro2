"""Tests for astroai.core.pipeline.timing."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from astroai.core.pipeline.timing import (
    PipelineTimer,
    StepTiming,
    TimingStore,
)


# ---------------------------------------------------------------------------
# StepTiming
# ---------------------------------------------------------------------------

class TestStepTiming:
    def test_elapsed_increases_while_running(self) -> None:
        t = StepTiming(step_type="A", start_time=time.monotonic() - 0.1)
        assert t.elapsed >= 0.1

    def test_elapsed_frozen_after_finish(self) -> None:
        t = StepTiming(step_type="A", start_time=time.monotonic() - 0.05)
        t.finish()
        e1 = t.elapsed
        time.sleep(0.02)
        e2 = t.elapsed
        assert e1 == pytest.approx(e2, abs=1e-9)

    def test_is_finished_false_before_finish(self) -> None:
        t = StepTiming(step_type="X")
        assert not t.is_finished

    def test_is_finished_true_after_finish(self) -> None:
        t = StepTiming(step_type="X")
        t.finish()
        assert t.is_finished

    def test_elapsed_non_negative(self) -> None:
        t = StepTiming(step_type="Y")
        assert t.elapsed >= 0.0


# ---------------------------------------------------------------------------
# TimingStore (in-memory, no disk)
# ---------------------------------------------------------------------------

class TestTimingStoreMemory:
    def test_record_and_eta(self) -> None:
        store = TimingStore()
        store.record("AIAlign", 10.0)
        store.record("AIAlign", 20.0)
        assert store.eta("AIAlign") == pytest.approx(15.0)

    def test_eta_none_when_empty(self) -> None:
        store = TimingStore()
        assert store.eta("Unknown") is None

    def test_window_enforced(self) -> None:
        store = TimingStore()
        for i in range(10):
            store.record("Step", float(i), window=3)
        # Only last 3: 7,8,9 → mean 8
        assert store.eta("Step") == pytest.approx(8.0)

    def test_history_returns_copy(self) -> None:
        store = TimingStore()
        store.record("S", 5.0)
        h = store.history("S")
        h.append(99.0)
        assert len(store.history("S")) == 1  # internal not mutated

    def test_all_step_types(self) -> None:
        store = TimingStore()
        store.record("A", 1.0)
        store.record("B", 2.0)
        assert set(store.all_step_types()) == {"A", "B"}


# ---------------------------------------------------------------------------
# TimingStore (disk persistence)
# ---------------------------------------------------------------------------

class TestTimingStorePersistence:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        store1 = TimingStore(tmp_path)
        store1.record("Calibrate", 5.0)
        store1.record("Calibrate", 7.0)
        store1.save()

        store2 = TimingStore(tmp_path)
        assert store2.eta("Calibrate") == pytest.approx(6.0)

    def test_save_creates_astroai_dir(self, tmp_path: Path) -> None:
        store = TimingStore(tmp_path)
        store.record("Step", 1.0)
        store.save()
        assert (tmp_path / ".astroai" / "timing.json").exists()

    def test_corrupt_file_handled_gracefully(self, tmp_path: Path) -> None:
        (tmp_path / ".astroai").mkdir(parents=True)
        (tmp_path / ".astroai" / "timing.json").write_text("NOT JSON")
        store = TimingStore(tmp_path)  # must not raise
        assert store.eta("Any") is None

    def test_missing_file_is_ok(self, tmp_path: Path) -> None:
        store = TimingStore(tmp_path)
        assert store.eta("X") is None

    def test_save_without_project_dir_is_noop(self) -> None:
        store = TimingStore()  # no dir
        store.record("S", 1.0)
        store.save()  # must not raise

    def test_json_structure(self, tmp_path: Path) -> None:
        store = TimingStore(tmp_path)
        store.record("Stack", 3.0)
        store.record("Stack", 5.0)
        store.save()
        raw = json.loads((tmp_path / ".astroai" / "timing.json").read_text())
        assert raw["Stack"] == pytest.approx([3.0, 5.0])


# ---------------------------------------------------------------------------
# PipelineTimer
# ---------------------------------------------------------------------------

class TestPipelineTimer:
    def test_start_and_finish_step(self) -> None:
        timer = PipelineTimer()
        t = timer.start_step("Load")
        assert timer.active is t
        timer.finish_step(t)
        assert timer.active is None
        assert t.is_finished

    def test_timings_accumulate(self) -> None:
        timer = PipelineTimer()
        for name in ("A", "B", "C"):
            t = timer.start_step(name)
            timer.finish_step(t)
        assert len(timer.timings) == 3
        assert [e.step_type for e in timer.timings] == ["A", "B", "C"]

    def test_eta_for_known_step(self) -> None:
        store = TimingStore()
        store.record("Register", 12.0)
        timer = PipelineTimer(store=store)
        assert timer.eta_for("Register") == pytest.approx(12.0)

    def test_eta_for_unknown_step(self) -> None:
        timer = PipelineTimer()
        assert timer.eta_for("Nonexistent") is None

    def test_remaining_eta_all_known(self) -> None:
        store = TimingStore()
        store.record("A", 10.0)
        store.record("B", 20.0)
        timer = PipelineTimer(store=store)
        assert timer.remaining_eta(["A", "B"]) == pytest.approx(30.0)

    def test_remaining_eta_unknown_returns_none(self) -> None:
        store = TimingStore()
        store.record("A", 5.0)
        timer = PipelineTimer(store=store)
        assert timer.remaining_eta(["A", "Unknown"]) is None

    def test_persist_calls_store_save(self, tmp_path: Path) -> None:
        store = TimingStore(tmp_path)
        timer = PipelineTimer(store=store)
        t = timer.start_step("Export")
        timer.finish_step(t)
        timer.persist()
        assert (tmp_path / ".astroai" / "timing.json").exists()

    def test_finished_step_updates_eta_history(self) -> None:
        store = TimingStore()
        timer = PipelineTimer(store=store)
        # simulate two runs
        for _ in range(2):
            t = timer.start_step("Process")
            time.sleep(0.01)
            timer.finish_step(t)
        eta = timer.eta_for("Process")
        assert eta is not None and eta > 0

    def test_timings_returns_copy(self) -> None:
        timer = PipelineTimer()
        copy = timer.timings
        copy.append(StepTiming(step_type="fake"))
        assert len(timer.timings) == 0  # original unmodified

    def test_active_is_latest_started(self) -> None:
        timer = PipelineTimer()
        t1 = timer.start_step("S1")
        timer.finish_step(t1)
        t2 = timer.start_step("S2")
        assert timer.active is t2
