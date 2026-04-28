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
    deconvolution_config_changed = Signal()
    channel_combine_config_changed = Signal()
    drizzle_config_changed = Signal()
    mosaic_config_changed = Signal()
    color_calibration_config_changed = Signal()

    DEFAULT_STEPS = [
        ("calibrate", "Kalibrierung", False),
        ("register", "Registrierung", False),
        ("stack", "Stacking", False),
        ("drizzle", "Drizzle", True),
        ("mosaic", "Mosaic", True),
        ("channel_combine", "Kanal-Kombination", True),
        ("stretch", "Stretching", False),
        ("color_calibration", "Farbkalibrierung", True),
        ("denoise", "Entrauschen", False),
        ("deconvolution", "Deconvolution", True),
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
        self._deconvolution_enabled: bool = False
        self._deconvolution_iterations: int = 10
        self._deconvolution_psf_sigma: float = 1.0
        self._channel_combine_enabled: bool = False
        self._channel_combine_mode: str = "lrgb"
        self._channel_combine_palette: str = "SHO"
        self._drizzle_enabled: bool = False
        self._drizzle_drop_size: float = 0.7
        self._drizzle_scale: float = 1.0
        self._drizzle_pixfrac: float = 1.0
        self._mosaic_enabled: bool = False
        self._mosaic_blend_mode: str = "average"
        self._mosaic_gradient_correct: bool = True
        self._mosaic_output_scale: float = 1.0
        self._mosaic_panels: list[str] = []
        self._color_calibration_enabled: bool = False
        self._color_calibration_catalog: str = "gaia_dr3"
        self._color_calibration_sample_radius: int = 8
        self._update_starless_step_state()
        self._update_deconvolution_step_state()
        self._update_drizzle_step_state()
        self._update_mosaic_step_state()
        self._update_channel_combine_step_state()
        self._update_color_calibration_step_state()

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

    # -- deconvolution config properties -------------------------------------

    @property
    def deconvolution_enabled(self) -> bool:
        return self._deconvolution_enabled

    @deconvolution_enabled.setter
    def deconvolution_enabled(self, value: bool) -> None:
        if self._deconvolution_enabled == value:
            return
        self._deconvolution_enabled = value
        self._update_deconvolution_step_state()
        self.deconvolution_config_changed.emit()

    @property
    def deconvolution_iterations(self) -> int:
        return self._deconvolution_iterations

    @deconvolution_iterations.setter
    def deconvolution_iterations(self, value: int) -> None:
        value = max(1, min(100, value))
        if self._deconvolution_iterations == value:
            return
        self._deconvolution_iterations = value
        self.deconvolution_config_changed.emit()

    @property
    def deconvolution_psf_sigma(self) -> float:
        return self._deconvolution_psf_sigma

    @deconvolution_psf_sigma.setter
    def deconvolution_psf_sigma(self, value: float) -> None:
        value = max(0.1, min(10.0, value))
        if self._deconvolution_psf_sigma == value:
            return
        self._deconvolution_psf_sigma = value
        self.deconvolution_config_changed.emit()

    def _update_deconvolution_step_state(self) -> None:
        step = self.step_by_key("deconvolution")
        if step is None:
            return
        if not self._deconvolution_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("deconvolution", StepState.DISABLED.value)
        elif self._deconvolution_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("deconvolution", StepState.PENDING.value)

    # -- channel combine config properties -----------------------------------

    @property
    def channel_combine_enabled(self) -> bool:
        return self._channel_combine_enabled

    @channel_combine_enabled.setter
    def channel_combine_enabled(self, value: bool) -> None:
        if self._channel_combine_enabled == value:
            return
        self._channel_combine_enabled = value
        self._update_channel_combine_step_state()
        self.channel_combine_config_changed.emit()

    @property
    def channel_combine_mode(self) -> str:
        return self._channel_combine_mode

    @channel_combine_mode.setter
    def channel_combine_mode(self, value: str) -> None:
        if self._channel_combine_mode == value:
            return
        self._channel_combine_mode = value
        self.channel_combine_config_changed.emit()

    @property
    def channel_combine_palette(self) -> str:
        return self._channel_combine_palette

    @channel_combine_palette.setter
    def channel_combine_palette(self, value: str) -> None:
        if self._channel_combine_palette == value:
            return
        self._channel_combine_palette = value
        self.channel_combine_config_changed.emit()

    def _update_channel_combine_step_state(self) -> None:
        step = self.step_by_key("channel_combine")
        if step is None:
            return
        if not self._channel_combine_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("channel_combine", StepState.DISABLED.value)
        elif self._channel_combine_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("channel_combine", StepState.PENDING.value)

    # -- drizzle config properties -----------------------------------------

    @property
    def drizzle_enabled(self) -> bool:
        return self._drizzle_enabled

    @drizzle_enabled.setter
    def drizzle_enabled(self, value: bool) -> None:
        if self._drizzle_enabled == value:
            return
        self._drizzle_enabled = value
        self._update_drizzle_step_state()
        self.drizzle_config_changed.emit()

    @property
    def drizzle_drop_size(self) -> float:
        return self._drizzle_drop_size

    @drizzle_drop_size.setter
    def drizzle_drop_size(self, value: float) -> None:
        if self._drizzle_drop_size == value:
            return
        self._drizzle_drop_size = value
        self.drizzle_config_changed.emit()

    @property
    def drizzle_scale(self) -> float:
        return self._drizzle_scale

    @drizzle_scale.setter
    def drizzle_scale(self, value: float) -> None:
        value = max(0.5, min(3.0, value))
        if self._drizzle_scale == value:
            return
        self._drizzle_scale = value
        self.drizzle_config_changed.emit()

    @property
    def drizzle_pixfrac(self) -> float:
        return self._drizzle_pixfrac

    @drizzle_pixfrac.setter
    def drizzle_pixfrac(self, value: float) -> None:
        value = max(0.1, min(1.0, value))
        if self._drizzle_pixfrac == value:
            return
        self._drizzle_pixfrac = value
        self.drizzle_config_changed.emit()

    def _update_drizzle_step_state(self) -> None:
        step = self.step_by_key("drizzle")
        if step is None:
            return
        if not self._drizzle_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("drizzle", StepState.DISABLED.value)
        elif self._drizzle_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("drizzle", StepState.PENDING.value)

    # -- mosaic config properties --------------------------------------------

    @property
    def mosaic_enabled(self) -> bool:
        return self._mosaic_enabled

    @mosaic_enabled.setter
    def mosaic_enabled(self, value: bool) -> None:
        if self._mosaic_enabled == value:
            return
        self._mosaic_enabled = value
        self._update_mosaic_step_state()
        self.mosaic_config_changed.emit()

    @property
    def mosaic_blend_mode(self) -> str:
        return self._mosaic_blend_mode

    @mosaic_blend_mode.setter
    def mosaic_blend_mode(self, value: str) -> None:
        if self._mosaic_blend_mode == value:
            return
        self._mosaic_blend_mode = value
        self.mosaic_config_changed.emit()

    @property
    def mosaic_gradient_correct(self) -> bool:
        return self._mosaic_gradient_correct

    @mosaic_gradient_correct.setter
    def mosaic_gradient_correct(self, value: bool) -> None:
        if self._mosaic_gradient_correct == value:
            return
        self._mosaic_gradient_correct = value
        self.mosaic_config_changed.emit()

    @property
    def mosaic_output_scale(self) -> float:
        return self._mosaic_output_scale

    @mosaic_output_scale.setter
    def mosaic_output_scale(self, value: float) -> None:
        value = max(0.25, min(4.0, value))
        if self._mosaic_output_scale == value:
            return
        self._mosaic_output_scale = value
        self.mosaic_config_changed.emit()

    @property
    def mosaic_panels(self) -> list[str]:
        return list(self._mosaic_panels)

    @mosaic_panels.setter
    def mosaic_panels(self, value: list[str]) -> None:
        if self._mosaic_panels == value:
            return
        self._mosaic_panels = list(value)
        self.mosaic_config_changed.emit()

    def add_mosaic_panel(self, path: str) -> None:
        if path not in self._mosaic_panels:
            self._mosaic_panels.append(path)
            self.mosaic_config_changed.emit()

    def remove_mosaic_panel(self, path: str) -> None:
        if path in self._mosaic_panels:
            self._mosaic_panels.remove(path)
            self.mosaic_config_changed.emit()

    def _update_mosaic_step_state(self) -> None:
        step = self.step_by_key("mosaic")
        if step is None:
            return
        if not self._mosaic_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("mosaic", StepState.DISABLED.value)
        elif self._mosaic_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("mosaic", StepState.PENDING.value)

    # -- color calibration config properties --------------------------------

    @property
    def color_calibration_enabled(self) -> bool:
        return self._color_calibration_enabled

    @color_calibration_enabled.setter
    def color_calibration_enabled(self, value: bool) -> None:
        if self._color_calibration_enabled == value:
            return
        self._color_calibration_enabled = value
        self._update_color_calibration_step_state()
        self.color_calibration_config_changed.emit()

    @property
    def color_calibration_catalog(self) -> str:
        return self._color_calibration_catalog

    @color_calibration_catalog.setter
    def color_calibration_catalog(self, value: str) -> None:
        if self._color_calibration_catalog == value:
            return
        self._color_calibration_catalog = value
        self.color_calibration_config_changed.emit()

    @property
    def color_calibration_sample_radius(self) -> int:
        return self._color_calibration_sample_radius

    @color_calibration_sample_radius.setter
    def color_calibration_sample_radius(self, value: int) -> None:
        value = max(3, min(20, value))
        if self._color_calibration_sample_radius == value:
            return
        self._color_calibration_sample_radius = value
        self.color_calibration_config_changed.emit()

    def _update_color_calibration_step_state(self) -> None:
        step = self.step_by_key("color_calibration")
        if step is None:
            return
        if not self._color_calibration_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("color_calibration", StepState.DISABLED.value)
        elif self._color_calibration_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("color_calibration", StepState.PENDING.value)

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
        optional_enabled = {
            "starless": self._starless_enabled,
            "deconvolution": self._deconvolution_enabled,
            "drizzle": self._drizzle_enabled,
            "mosaic": self._mosaic_enabled,
            "channel_combine": self._channel_combine_enabled,
            "color_calibration": self._color_calibration_enabled,
        }
        for step in self._steps:
            if step.optional and not optional_enabled.get(step.key, False):
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
