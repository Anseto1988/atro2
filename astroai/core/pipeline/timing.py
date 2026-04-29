"""Pipeline step wall-time measurement and rolling-average ETA.

Usage:
    store = TimingStore(Path("/my/project"))
    timer = PipelineTimer(store)
    t = timer.start_step("AIAlignment")
    ...do work...
    timer.finish_step(t)
    timer.persist()
    eta = timer.eta_for("AIAlignment")  # seconds or None
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "StepTiming",
    "TimingStore",
    "PipelineTimer",
]

_DEFAULT_WINDOW = 5  # rolling-average window size


@dataclass
class StepTiming:
    """Tracks start/end wall-clock time for one step execution."""

    step_type: str
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None

    @property
    def elapsed(self) -> float:
        """Elapsed seconds; uses current time if still running."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.monotonic() - self.start_time

    @property
    def is_finished(self) -> bool:
        return self.end_time is not None

    def finish(self) -> None:
        self.end_time = time.monotonic()


class TimingStore:
    """Loads and persists per-step duration history from `.astroai/timing.json`."""

    def __init__(self, project_dir: Path | None = None) -> None:
        self._dir = project_dir
        self._history: dict[str, list[float]] = {}
        if project_dir is not None:
            self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self) -> Path | None:
        if self._dir is None:
            return None
        return self._dir / ".astroai" / "timing.json"

    def _load(self) -> None:
        p = self._path()
        if p is None or not p.exists():
            return
        try:
            data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
            self._history = {
                k: [float(v) for v in vals]
                for k, vals in data.items()
                if isinstance(vals, list)
            }
        except Exception:
            self._history = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist history to disk (no-op when no project_dir)."""
        p = self._path()
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._history, indent=2), encoding="utf-8")

    def record(self, step_type: str, duration: float, window: int = _DEFAULT_WINDOW) -> None:
        """Append a duration sample, keeping only the last *window* values."""
        bucket = self._history.setdefault(step_type, [])
        bucket.append(duration)
        if len(bucket) > window:
            del bucket[: len(bucket) - window]

    def eta(self, step_type: str) -> float | None:
        """Rolling-average ETA in seconds, or None when no history exists."""
        bucket = self._history.get(step_type)
        if not bucket:
            return None
        return sum(bucket) / len(bucket)

    def all_step_types(self) -> list[str]:
        return list(self._history.keys())

    def history(self, step_type: str) -> list[float]:
        return list(self._history.get(step_type, []))


class PipelineTimer:
    """Orchestrates per-step timing across a single pipeline run."""

    def __init__(self, store: TimingStore | None = None) -> None:
        self._store: TimingStore = store if store is not None else TimingStore()
        self._timings: list[StepTiming] = []
        self._active: StepTiming | None = None

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def start_step(self, step_type: str) -> StepTiming:
        t = StepTiming(step_type=step_type)
        self._timings.append(t)
        self._active = t
        return t

    def finish_step(self, timing: StepTiming) -> None:
        """Mark step done and record its duration in the store."""
        timing.finish()
        self._active = None
        self._store.record(timing.step_type, timing.elapsed)

    # ------------------------------------------------------------------
    # ETA helpers
    # ------------------------------------------------------------------

    def eta_for(self, step_type: str) -> float | None:
        """Return rolling-average ETA for *step_type*, or None."""
        return self._store.eta(step_type)

    def remaining_eta(self, pending_step_types: list[str]) -> float | None:
        """Sum of ETAs for all pending step types; None if any ETA is unknown."""
        total = 0.0
        for st in pending_step_types:
            e = self._store.eta(st)
            if e is None:
                return None
            total += e
        return total

    # ------------------------------------------------------------------
    # Persistence & introspection
    # ------------------------------------------------------------------

    def persist(self) -> None:
        self._store.save()

    @property
    def timings(self) -> list[StepTiming]:
        return list(self._timings)

    @property
    def active(self) -> StepTiming | None:
        return self._active
