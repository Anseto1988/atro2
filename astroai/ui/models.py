"""Shared UI data models for pipeline state and signals."""
from __future__ import annotations

from enum import Enum
from typing import Any

from PySide6.QtCore import QObject, Signal


class StepState(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    ERROR = "error"
    DISABLED = "disabled"


class PipelineStep:
    __slots__ = ("key", "label", "state", "progress", "optional")

    def __init__(self, key: str, label: str, *, optional: bool = False) -> None:
        self.key = key
        self.label = label
        self.state: StepState = StepState.PENDING
        self.progress: float = 0.0
        self.optional = optional


class PipelineModel(QObject):
    """Observable pipeline state shared between backend workers and UI."""

    step_changed = Signal(str, str)  # (step_key, new_state_value)
    progress_changed = Signal(str, float)  # (step_key, 0.0-1.0)
    pipeline_reset = Signal()
    starless_config_changed = Signal()

    DEFAULT_STEPS = [
        ("calibrate", "Kalibrierung", False),
        ("register", "Registrierung", False),
        ("stack", "Stacking", False),
        ("stretch", "Stretching", False),
        ("denoise", "Entrauschen", False),
        ("starless", "Starless", True),
        ("export", "Export", False),
    ]

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._steps: list[PipelineStep] = [
            PipelineStep(k, lbl, optional=opt) for k, lbl, opt in self.DEFAULT_STEPS
        ]
        self._starless_enabled: bool = False
        self._starless_strength: float = 1.0
        self._starless_format: str = "xisf"
        self._save_star_mask: bool = True
        self._update_starless_step_state()

    # -- starless config properties --

    @property
    def starless_enabled(self) -> bool:
        return self._starless_enabled

    @starless_enabled.setter
    def starless_enabled(self, value: bool) -> None:
        if self._starless_enabled == value:
            return
        self._starless_enabled = value
        self._update_starless_step_state()
        self.starless_config_changed.emit()

    @property
    def starless_strength(self) -> float:
        return self._starless_strength

    @starless_strength.setter
    def starless_strength(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._starless_strength == value:
            return
        self._starless_strength = value
        self.starless_config_changed.emit()

    @property
    def starless_format(self) -> str:
        return self._starless_format

    @starless_format.setter
    def starless_format(self, value: str) -> None:
        if self._starless_format == value:
            return
        self._starless_format = value
        self.starless_config_changed.emit()

    @property
    def save_star_mask(self) -> bool:
        return self._save_star_mask

    @save_star_mask.setter
    def save_star_mask(self, value: bool) -> None:
        if self._save_star_mask == value:
            return
        self._save_star_mask = value
        self.starless_config_changed.emit()

    def _update_starless_step_state(self) -> None:
        step = self.step_by_key("starless")
        if step is None:
            return
        if not self._starless_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("starless", StepState.DISABLED.value)
        elif self._starless_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("starless", StepState.PENDING.value)

    # -- export config bridge --

    def export_config(self) -> dict[str, Any]:
        """Return ExportStep-compatible parameters derived from UI state."""
        return {
            "fmt_value": self._starless_format,
            "export_starless": self._starless_enabled,
            "export_star_mask": self._starless_enabled and self._save_star_mask,
        }

    # -- step access --

    @property
    def steps(self) -> list[PipelineStep]:
        return list(self._steps)

    def step_by_key(self, key: str) -> PipelineStep | None:
        return next((s for s in self._steps if s.key == key), None)

    def active_step(self) -> PipelineStep | None:
        return next((s for s in self._steps if s.state is StepState.ACTIVE), None)

    def set_step_state(self, key: str, state: StepState) -> None:
        step = self.step_by_key(key)
        if step is None:
            return
        step.state = state
        self.step_changed.emit(key, state.value)

    def set_step_progress(self, key: str, value: float) -> None:
        step = self.step_by_key(key)
        if step is None:
            return
        step.progress = max(0.0, min(1.0, value))
        self.progress_changed.emit(key, step.progress)

    def reset(self) -> None:
        for step in self._steps:
            if step.optional and not self._starless_enabled:
                step.state = StepState.DISABLED
            else:
                step.state = StepState.PENDING
            step.progress = 0.0
        self.pipeline_reset.emit()

    def advance_to(self, key: str) -> None:
        found = False
        for step in self._steps:
            if step.state is StepState.DISABLED:
                continue
            if step.key == key:
                step.state = StepState.ACTIVE
                step.progress = 0.0
                found = True
            elif not found:
                step.state = StepState.DONE
                step.progress = 1.0
        if found:
            self.step_changed.emit(key, StepState.ACTIVE.value)
