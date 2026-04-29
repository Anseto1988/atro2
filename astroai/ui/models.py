"""Shared UI data models for pipeline state and signals."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    comet_stack_config_changed = Signal()
    comet_preview_changed = Signal()
    synthetic_flat_config_changed = Signal()
    frame_selection_config_changed = Signal()
    background_removal_config_changed = Signal()
    denoise_config_changed = Signal()
    sharpening_config_changed = Signal()
    saturation_config_changed = Signal()
    white_balance_config_changed = Signal()
    background_neutralization_config_changed = Signal()
    asinh_stretch_config_changed = Signal()
    mtf_stretch_config_changed = Signal()
    clahe_config_changed = Signal()
    star_reduction_config_changed = Signal()
    color_grading_config_changed = Signal()
    stretch_config_changed = Signal()
    star_processing_config_changed = Signal()
    registration_config_changed = Signal()
    stacking_config_changed = Signal()
    annotation_config_changed = Signal()
    export_config_changed = Signal()
    curves_config_changed = Signal()
    histogram_changed = Signal(object)  # ndarray emitted after pipeline finish or file load

    DEFAULT_STEPS = [
        ("calibrate", "Kalibrierung", False),
        ("frame_selection", "Frame-Selektion", True),
        ("synthetic_flat", "Synth. Flat", True),
        ("register", "Registrierung", False),
        ("stack", "Stacking", False),
        ("comet_stacking", "Komet-Stacking", True),
        ("drizzle", "Drizzle", True),
        ("mosaic", "Mosaic", True),
        ("channel_combine", "Kanal-Kombination", True),
        ("stretch", "Stretching", False),
        ("curves", "Kurven", True),
        ("background_removal", "Hintergrundentfernung", True),
        ("color_calibration", "Farbkalibrierung", True),
        ("denoise", "Entrauschen", False),
        ("deconvolution", "Deconvolution", True),
        ("sharpening", "Schärfung", True),
        ("saturation", "Selektive Sättigung", True),
        ("bg_neutralization", "Hintergrundneutralisierung", True),
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
        self._synthetic_flat_enabled: bool = False
        self._synthetic_flat_tile_size: int = 64
        self._synthetic_flat_smoothing_sigma: float = 8.0
        self._frame_selection_enabled: bool = False
        self._frame_selection_min_score: float = 0.5
        self._frame_selection_max_rejected_fraction: float = 0.8
        self._curves_enabled: bool = False
        self._curves_rgb_points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self._curves_r_points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self._curves_g_points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self._curves_b_points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self._background_removal_enabled: bool = False
        self._background_removal_tile_size: int = 64
        self._background_removal_method: str = "rbf"
        self._background_removal_preserve_median: bool = True
        self._denoise_strength: float = 1.0
        self._denoise_tile_size: int = 512
        self._denoise_tile_overlap: int = 64
        self._adaptive_denoise_enabled: bool = False
        self._sharpening_enabled: bool = False
        self._sharpening_radius: float = 1.0
        self._sharpening_amount: float = 0.5
        self._sharpening_threshold: float = 0.02
        self._saturation_enabled: bool = False
        self._saturation_global: float = 1.0
        # white balance
        self._white_balance_enabled: bool = False
        self._wb_red: float = 1.0
        self._wb_green: float = 1.0
        self._wb_blue: float = 1.0
        # background neutralization
        self._bg_neutralization_enabled: bool = False
        self._bg_neutralization_sample_mode: str = "auto"
        self._bg_neutralization_target: float = 0.1
        self._bg_neutralization_roi: tuple[int, int, int, int] | None = None
        # arcsinh stretch
        self._asinh_enabled: bool = False
        self._asinh_stretch_factor: float = 1.0
        self._asinh_black_point: float = 0.0
        self._asinh_linked: bool = True
        # MTF stretch
        self._mtf_enabled: bool = False
        self._mtf_midpoint: float = 0.25
        self._mtf_shadows_clipping: float = 0.0
        self._mtf_highlights: float = 1.0
        # CLAHE local contrast enhancement
        self._clahe_enabled: bool = False
        self._clahe_clip_limit: float = 2.0
        self._clahe_tile_size: int = 64
        self._clahe_channel_mode: str = "luminance"
        # star reduction
        self._star_reduction_enabled: bool = False
        self._star_reduction_amount: float = 0.5
        self._star_reduction_radius: int = 2
        self._star_reduction_threshold: float = 0.5
        # color grading
        self._color_grading_enabled: bool = False
        self._cg_shadow_r: float = 0.0
        self._cg_shadow_g: float = 0.0
        self._cg_shadow_b: float = 0.0
        self._cg_midtone_r: float = 0.0
        self._cg_midtone_g: float = 0.0
        self._cg_midtone_b: float = 0.0
        self._cg_highlight_r: float = 0.0
        self._cg_highlight_g: float = 0.0
        self._cg_highlight_b: float = 0.0
        self._saturation_reds: float = 1.0
        self._saturation_oranges: float = 1.0
        self._saturation_yellows: float = 1.0
        self._saturation_greens: float = 1.0
        self._saturation_cyans: float = 1.0
        self._saturation_blues: float = 1.0
        self._saturation_purples: float = 1.0
        self._stretch_target_background: float = 0.25
        self._stretch_shadow_clipping_sigmas: float = -2.8
        self._stretch_linked_channels: bool = True
        self._star_detection_sigma: float = 4.0
        self._star_min_area: int = 3
        self._star_max_area: int = 5000
        self._star_mask_dilation: int = 3
        self._star_reduce_enabled: bool = False
        self._star_reduce_factor: float = 0.5
        self._annotation_show_dso: bool = True
        self._annotation_show_stars: bool = True
        self._annotation_show_boundaries: bool = False
        self._annotation_show_grid: bool = False
        self._output_path: str = ""
        self._output_format: str = "fits"
        self._output_filename: str = "output"
        self._registration_upsample_factor: int = 10
        self._registration_reference_frame_index: int = 0
        self._registration_method: str = "star"
        self._stacking_method: str = "sigma_clip"
        self._stacking_sigma_low: float = 2.5
        self._stacking_sigma_high: float = 2.5
        self._comet_stack_enabled: bool = False
        self._comet_tracking_mode: str = "blend"
        self._comet_blend_factor: float = 0.5
        self._comet_star_stack: NDArray[np.floating[Any]] | None = None
        self._comet_nucleus_stack: NDArray[np.floating[Any]] | None = None
        self._update_starless_step_state()
        self._update_deconvolution_step_state()
        self._update_drizzle_step_state()
        self._update_mosaic_step_state()
        self._update_channel_combine_step_state()
        self._update_color_calibration_step_state()
        self._update_comet_stack_step_state()
        self._update_synthetic_flat_step_state()
        self._update_frame_selection_step_state()
        self._update_background_removal_step_state()
        self._update_curves_step_state()
        self._update_bg_neutralization_step_state()

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

    # -- comet stack config properties -----------------------------------------

    @property
    def comet_stack_enabled(self) -> bool:
        return self._comet_stack_enabled

    @comet_stack_enabled.setter
    def comet_stack_enabled(self, value: bool) -> None:
        if self._comet_stack_enabled == value:
            return
        self._comet_stack_enabled = value
        self._update_comet_stack_step_state()
        self.comet_stack_config_changed.emit()

    @property
    def comet_tracking_mode(self) -> str:
        return self._comet_tracking_mode

    @comet_tracking_mode.setter
    def comet_tracking_mode(self, value: str) -> None:
        if self._comet_tracking_mode == value:
            return
        self._comet_tracking_mode = value
        self.comet_stack_config_changed.emit()
        if self._has_comet_stacks():
            self.comet_preview_changed.emit()

    @property
    def comet_blend_factor(self) -> float:
        return self._comet_blend_factor

    @comet_blend_factor.setter
    def comet_blend_factor(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._comet_blend_factor == value:
            return
        self._comet_blend_factor = value
        self.comet_stack_config_changed.emit()
        if self._comet_tracking_mode == "blend" and self._has_comet_stacks():
            self.comet_preview_changed.emit()

    def _has_comet_stacks(self) -> bool:
        return self._comet_star_stack is not None and self._comet_nucleus_stack is not None

    def set_comet_stacks(
        self,
        star_stack: NDArray[np.floating[Any]],
        nucleus_stack: NDArray[np.floating[Any]],
    ) -> None:
        self._comet_star_stack = star_stack
        self._comet_nucleus_stack = nucleus_stack
        self.comet_preview_changed.emit()

    def clear_comet_stacks(self) -> None:
        self._comet_star_stack = None
        self._comet_nucleus_stack = None

    @property
    def comet_preview_image(self) -> NDArray[np.floating[Any]] | None:
        if not self._has_comet_stacks():
            return None
        assert self._comet_star_stack is not None
        assert self._comet_nucleus_stack is not None
        mode = self._comet_tracking_mode
        if mode == "stars":
            return self._comet_star_stack
        if mode == "comet":
            return self._comet_nucleus_stack
        f = self._comet_blend_factor
        return ((1.0 - f) * self._comet_star_stack + f * self._comet_nucleus_stack).astype(
            self._comet_star_stack.dtype
        )

    def _update_comet_stack_step_state(self) -> None:
        step = self.step_by_key("comet_stacking")
        if step is None:
            return
        if not self._comet_stack_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("comet_stacking", StepState.DISABLED.value)
        elif self._comet_stack_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("comet_stacking", StepState.PENDING.value)

    # -- synthetic flat config properties -------------------------------------

    @property
    def synthetic_flat_enabled(self) -> bool:
        return self._synthetic_flat_enabled

    @synthetic_flat_enabled.setter
    def synthetic_flat_enabled(self, value: bool) -> None:
        if self._synthetic_flat_enabled == value:
            return
        self._synthetic_flat_enabled = value
        self._update_synthetic_flat_step_state()
        self.synthetic_flat_config_changed.emit()

    @property
    def synthetic_flat_tile_size(self) -> int:
        return self._synthetic_flat_tile_size

    @synthetic_flat_tile_size.setter
    def synthetic_flat_tile_size(self, value: int) -> None:
        value = max(16, min(256, value))
        if self._synthetic_flat_tile_size == value:
            return
        self._synthetic_flat_tile_size = value
        self.synthetic_flat_config_changed.emit()

    @property
    def synthetic_flat_smoothing_sigma(self) -> float:
        return self._synthetic_flat_smoothing_sigma

    @synthetic_flat_smoothing_sigma.setter
    def synthetic_flat_smoothing_sigma(self, value: float) -> None:
        value = max(0.0, min(50.0, value))
        if self._synthetic_flat_smoothing_sigma == value:
            return
        self._synthetic_flat_smoothing_sigma = value
        self.synthetic_flat_config_changed.emit()

    def _update_synthetic_flat_step_state(self) -> None:
        step = self.step_by_key("synthetic_flat")
        if step is None:
            return
        if not self._synthetic_flat_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("synthetic_flat", StepState.DISABLED.value)
        elif self._synthetic_flat_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("synthetic_flat", StepState.PENDING.value)

    # -- frame selection config properties -----------------------------------

    @property
    def frame_selection_enabled(self) -> bool:
        return self._frame_selection_enabled

    @frame_selection_enabled.setter
    def frame_selection_enabled(self, value: bool) -> None:
        if self._frame_selection_enabled == value:
            return
        self._frame_selection_enabled = value
        self._update_frame_selection_step_state()
        self.frame_selection_config_changed.emit()

    @property
    def frame_selection_min_score(self) -> float:
        return self._frame_selection_min_score

    @frame_selection_min_score.setter
    def frame_selection_min_score(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._frame_selection_min_score == value:
            return
        self._frame_selection_min_score = value
        self.frame_selection_config_changed.emit()

    @property
    def frame_selection_max_rejected_fraction(self) -> float:
        return self._frame_selection_max_rejected_fraction

    @frame_selection_max_rejected_fraction.setter
    def frame_selection_max_rejected_fraction(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._frame_selection_max_rejected_fraction == value:
            return
        self._frame_selection_max_rejected_fraction = value
        self.frame_selection_config_changed.emit()

    def _update_frame_selection_step_state(self) -> None:
        step = self.step_by_key("frame_selection")
        if step is None:
            return
        if not self._frame_selection_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("frame_selection", StepState.DISABLED.value)
        elif self._frame_selection_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("frame_selection", StepState.PENDING.value)

    # -- background removal config properties --------------------------------

    @property
    def background_removal_enabled(self) -> bool:
        return self._background_removal_enabled

    @background_removal_enabled.setter
    def background_removal_enabled(self, value: bool) -> None:
        if self._background_removal_enabled == value:
            return
        self._background_removal_enabled = value
        self._update_background_removal_step_state()
        self.background_removal_config_changed.emit()

    @property
    def background_removal_tile_size(self) -> int:
        return self._background_removal_tile_size

    @background_removal_tile_size.setter
    def background_removal_tile_size(self, value: int) -> None:
        value = max(16, min(256, value))
        if self._background_removal_tile_size == value:
            return
        self._background_removal_tile_size = value
        self.background_removal_config_changed.emit()

    @property
    def background_removal_method(self) -> str:
        return self._background_removal_method

    @background_removal_method.setter
    def background_removal_method(self, value: str) -> None:
        if self._background_removal_method == value:
            return
        self._background_removal_method = value
        self.background_removal_config_changed.emit()

    @property
    def background_removal_preserve_median(self) -> bool:
        return self._background_removal_preserve_median

    @background_removal_preserve_median.setter
    def background_removal_preserve_median(self, value: bool) -> None:
        if self._background_removal_preserve_median == value:
            return
        self._background_removal_preserve_median = value
        self.background_removal_config_changed.emit()

    def _update_background_removal_step_state(self) -> None:
        step = self.step_by_key("background_removal")
        if step is None:
            return
        if not self._background_removal_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("background_removal", StepState.DISABLED.value)
        elif self._background_removal_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("background_removal", StepState.PENDING.value)

    # -- denoise config properties -------------------------------------------

    @property
    def denoise_strength(self) -> float:
        return self._denoise_strength

    @denoise_strength.setter
    def denoise_strength(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._denoise_strength == value:
            return
        self._denoise_strength = value
        self.denoise_config_changed.emit()

    @property
    def denoise_tile_size(self) -> int:
        return self._denoise_tile_size

    @denoise_tile_size.setter
    def denoise_tile_size(self, value: int) -> None:
        value = max(64, min(1024, value))
        if self._denoise_tile_size == value:
            return
        self._denoise_tile_size = value
        self.denoise_config_changed.emit()

    @property
    def denoise_tile_overlap(self) -> int:
        return self._denoise_tile_overlap

    @denoise_tile_overlap.setter
    def denoise_tile_overlap(self, value: int) -> None:
        value = max(0, min(256, value))
        if self._denoise_tile_overlap == value:
            return
        self._denoise_tile_overlap = value
        self.denoise_config_changed.emit()

    @property
    def adaptive_denoise_enabled(self) -> bool:
        return self._adaptive_denoise_enabled

    @adaptive_denoise_enabled.setter
    def adaptive_denoise_enabled(self, value: bool) -> None:
        if self._adaptive_denoise_enabled == value:
            return
        self._adaptive_denoise_enabled = value
        self.denoise_config_changed.emit()

    # -- sharpening config properties -----------------------------------------

    @property
    def sharpening_enabled(self) -> bool:
        return self._sharpening_enabled

    @sharpening_enabled.setter
    def sharpening_enabled(self, value: bool) -> None:
        if self._sharpening_enabled == value:
            return
        self._sharpening_enabled = value
        self.sharpening_config_changed.emit()

    @property
    def sharpening_radius(self) -> float:
        return self._sharpening_radius

    @sharpening_radius.setter
    def sharpening_radius(self, value: float) -> None:
        value = max(0.1, min(10.0, value))
        if self._sharpening_radius == value:
            return
        self._sharpening_radius = value
        self.sharpening_config_changed.emit()

    @property
    def sharpening_amount(self) -> float:
        return self._sharpening_amount

    @sharpening_amount.setter
    def sharpening_amount(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._sharpening_amount == value:
            return
        self._sharpening_amount = value
        self.sharpening_config_changed.emit()

    @property
    def sharpening_threshold(self) -> float:
        return self._sharpening_threshold

    @sharpening_threshold.setter
    def sharpening_threshold(self, value: float) -> None:
        value = max(0.0, min(0.5, value))
        if self._sharpening_threshold == value:
            return
        self._sharpening_threshold = value
        self.sharpening_config_changed.emit()

    # -- saturation config properties -----------------------------------------

    @property
    def saturation_enabled(self) -> bool:
        return self._saturation_enabled

    @saturation_enabled.setter
    def saturation_enabled(self, value: bool) -> None:
        if self._saturation_enabled == value:
            return
        self._saturation_enabled = value
        self.saturation_config_changed.emit()

    @property
    def saturation_global(self) -> float:
        return self._saturation_global

    @saturation_global.setter
    def saturation_global(self, value: float) -> None:
        value = max(0.0, min(4.0, value))
        if self._saturation_global == value:
            return
        self._saturation_global = value
        self.saturation_config_changed.emit()

    def _saturation_range_getter(self, attr: str) -> float:
        return getattr(self, attr)

    def _saturation_range_setter(self, attr: str, value: float) -> None:
        value = max(0.0, min(4.0, value))
        if getattr(self, attr) == value:
            return
        setattr(self, attr, value)
        self.saturation_config_changed.emit()

    @property
    def saturation_reds(self) -> float:
        return self._saturation_reds

    @saturation_reds.setter
    def saturation_reds(self, value: float) -> None:
        value = max(0.0, min(4.0, value))
        if self._saturation_reds == value:
            return
        self._saturation_reds = value
        self.saturation_config_changed.emit()

    @property
    def saturation_oranges(self) -> float:
        return self._saturation_oranges

    @saturation_oranges.setter
    def saturation_oranges(self, value: float) -> None:
        value = max(0.0, min(4.0, value))
        if self._saturation_oranges == value:
            return
        self._saturation_oranges = value
        self.saturation_config_changed.emit()

    @property
    def saturation_yellows(self) -> float:
        return self._saturation_yellows

    @saturation_yellows.setter
    def saturation_yellows(self, value: float) -> None:
        value = max(0.0, min(4.0, value))
        if self._saturation_yellows == value:
            return
        self._saturation_yellows = value
        self.saturation_config_changed.emit()

    @property
    def saturation_greens(self) -> float:
        return self._saturation_greens

    @saturation_greens.setter
    def saturation_greens(self, value: float) -> None:
        value = max(0.0, min(4.0, value))
        if self._saturation_greens == value:
            return
        self._saturation_greens = value
        self.saturation_config_changed.emit()

    @property
    def saturation_cyans(self) -> float:
        return self._saturation_cyans

    @saturation_cyans.setter
    def saturation_cyans(self, value: float) -> None:
        value = max(0.0, min(4.0, value))
        if self._saturation_cyans == value:
            return
        self._saturation_cyans = value
        self.saturation_config_changed.emit()

    @property
    def saturation_blues(self) -> float:
        return self._saturation_blues

    @saturation_blues.setter
    def saturation_blues(self, value: float) -> None:
        value = max(0.0, min(4.0, value))
        if self._saturation_blues == value:
            return
        self._saturation_blues = value
        self.saturation_config_changed.emit()

    @property
    def saturation_purples(self) -> float:
        return self._saturation_purples

    @saturation_purples.setter
    def saturation_purples(self, value: float) -> None:
        value = max(0.0, min(4.0, value))
        if self._saturation_purples == value:
            return
        self._saturation_purples = value
        self.saturation_config_changed.emit()

    # -- white balance config properties --------------------------------------

    @property
    def white_balance_enabled(self) -> bool:
        return self._white_balance_enabled

    @white_balance_enabled.setter
    def white_balance_enabled(self, value: bool) -> None:
        if self._white_balance_enabled == value:
            return
        self._white_balance_enabled = value
        self.white_balance_config_changed.emit()

    @property
    def wb_red(self) -> float:
        return self._wb_red

    @wb_red.setter
    def wb_red(self, value: float) -> None:
        value = max(0.1, min(5.0, value))
        if self._wb_red == value:
            return
        self._wb_red = value
        self.white_balance_config_changed.emit()

    @property
    def wb_green(self) -> float:
        return self._wb_green

    @wb_green.setter
    def wb_green(self, value: float) -> None:
        value = max(0.1, min(5.0, value))
        if self._wb_green == value:
            return
        self._wb_green = value
        self.white_balance_config_changed.emit()

    @property
    def wb_blue(self) -> float:
        return self._wb_blue

    @wb_blue.setter
    def wb_blue(self, value: float) -> None:
        value = max(0.1, min(5.0, value))
        if self._wb_blue == value:
            return
        self._wb_blue = value
        self.white_balance_config_changed.emit()

    # -- background neutralization config properties --------------------------

    @property
    def bg_neutralization_enabled(self) -> bool:
        return self._bg_neutralization_enabled

    @bg_neutralization_enabled.setter
    def bg_neutralization_enabled(self, value: bool) -> None:
        if self._bg_neutralization_enabled == value:
            return
        self._bg_neutralization_enabled = value
        self._update_bg_neutralization_step_state()
        self.background_neutralization_config_changed.emit()

    @property
    def bg_neutralization_sample_mode(self) -> str:
        return self._bg_neutralization_sample_mode

    @bg_neutralization_sample_mode.setter
    def bg_neutralization_sample_mode(self, value: str) -> None:
        if value not in ("auto", "manual"):
            value = "auto"
        if self._bg_neutralization_sample_mode == value:
            return
        self._bg_neutralization_sample_mode = value
        self.background_neutralization_config_changed.emit()

    @property
    def bg_neutralization_target(self) -> float:
        return self._bg_neutralization_target

    @bg_neutralization_target.setter
    def bg_neutralization_target(self, value: float) -> None:
        value = max(0.0, min(0.3, value))
        if self._bg_neutralization_target == value:
            return
        self._bg_neutralization_target = value
        self.background_neutralization_config_changed.emit()

    @property
    def bg_neutralization_roi(self) -> tuple[int, int, int, int] | None:
        return self._bg_neutralization_roi

    @bg_neutralization_roi.setter
    def bg_neutralization_roi(self, value: tuple[int, int, int, int] | None) -> None:
        if self._bg_neutralization_roi == value:
            return
        self._bg_neutralization_roi = value
        self.background_neutralization_config_changed.emit()

    def _update_bg_neutralization_step_state(self) -> None:
        step = self.step_by_key("bg_neutralization")
        if step is None:
            return
        if not self._bg_neutralization_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("bg_neutralization", StepState.DISABLED.value)
        elif self._bg_neutralization_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("bg_neutralization", StepState.PENDING.value)

    # -- arcsinh stretch config properties ------------------------------------

    @property
    def asinh_enabled(self) -> bool:
        return self._asinh_enabled

    @asinh_enabled.setter
    def asinh_enabled(self, value: bool) -> None:
        if self._asinh_enabled == value:
            return
        self._asinh_enabled = value
        self.asinh_stretch_config_changed.emit()

    @property
    def asinh_stretch_factor(self) -> float:
        return self._asinh_stretch_factor

    @asinh_stretch_factor.setter
    def asinh_stretch_factor(self, value: float) -> None:
        value = max(0.001, min(1000.0, value))
        if self._asinh_stretch_factor == value:
            return
        self._asinh_stretch_factor = value
        self.asinh_stretch_config_changed.emit()

    @property
    def asinh_black_point(self) -> float:
        return self._asinh_black_point

    @asinh_black_point.setter
    def asinh_black_point(self, value: float) -> None:
        value = max(0.0, min(0.5, value))
        if self._asinh_black_point == value:
            return
        self._asinh_black_point = value
        self.asinh_stretch_config_changed.emit()

    @property
    def asinh_linked(self) -> bool:
        return self._asinh_linked

    @asinh_linked.setter
    def asinh_linked(self, value: bool) -> None:
        if self._asinh_linked == value:
            return
        self._asinh_linked = value
        self.asinh_stretch_config_changed.emit()

    # -- MTF stretch config properties ----------------------------------------

    @property
    def mtf_enabled(self) -> bool:
        return self._mtf_enabled

    @mtf_enabled.setter
    def mtf_enabled(self, value: bool) -> None:
        if self._mtf_enabled == value:
            return
        self._mtf_enabled = value
        self.mtf_stretch_config_changed.emit()

    @property
    def mtf_midpoint(self) -> float:
        return self._mtf_midpoint

    @mtf_midpoint.setter
    def mtf_midpoint(self, value: float) -> None:
        value = max(0.001, min(0.499, value))
        if self._mtf_midpoint == value:
            return
        self._mtf_midpoint = value
        self.mtf_stretch_config_changed.emit()

    @property
    def mtf_shadows_clipping(self) -> float:
        return self._mtf_shadows_clipping

    @mtf_shadows_clipping.setter
    def mtf_shadows_clipping(self, value: float) -> None:
        value = max(0.0, min(0.1, value))
        if self._mtf_shadows_clipping == value:
            return
        self._mtf_shadows_clipping = value
        self.mtf_stretch_config_changed.emit()

    @property
    def mtf_highlights(self) -> float:
        return self._mtf_highlights

    @mtf_highlights.setter
    def mtf_highlights(self, value: float) -> None:
        value = max(0.98, min(1.0, value))
        if self._mtf_highlights == value:
            return
        self._mtf_highlights = value
        self.mtf_stretch_config_changed.emit()

    # -- CLAHE config properties ----------------------------------------------

    @property
    def clahe_enabled(self) -> bool:
        return self._clahe_enabled

    @clahe_enabled.setter
    def clahe_enabled(self, value: bool) -> None:
        if self._clahe_enabled == value:
            return
        self._clahe_enabled = value
        self.clahe_config_changed.emit()

    @property
    def clahe_clip_limit(self) -> float:
        return self._clahe_clip_limit

    @clahe_clip_limit.setter
    def clahe_clip_limit(self, value: float) -> None:
        value = max(1.0, min(10.0, value))
        if self._clahe_clip_limit == value:
            return
        self._clahe_clip_limit = value
        self.clahe_config_changed.emit()

    @property
    def clahe_tile_size(self) -> int:
        return self._clahe_tile_size

    @clahe_tile_size.setter
    def clahe_tile_size(self, value: int) -> None:
        value = max(8, min(512, int(value)))
        if self._clahe_tile_size == value:
            return
        self._clahe_tile_size = value
        self.clahe_config_changed.emit()

    @property
    def clahe_channel_mode(self) -> str:
        return self._clahe_channel_mode

    @clahe_channel_mode.setter
    def clahe_channel_mode(self, value: str) -> None:
        if value not in ("luminance", "each", "grayscale"):
            value = "luminance"
        if self._clahe_channel_mode == value:
            return
        self._clahe_channel_mode = value
        self.clahe_config_changed.emit()

    # -- star reduction config properties ------------------------------------

    @property
    def star_reduction_enabled(self) -> bool:
        return self._star_reduction_enabled

    @star_reduction_enabled.setter
    def star_reduction_enabled(self, value: bool) -> None:
        if self._star_reduction_enabled == value:
            return
        self._star_reduction_enabled = value
        self.star_reduction_config_changed.emit()

    @property
    def star_reduction_amount(self) -> float:
        return self._star_reduction_amount

    @star_reduction_amount.setter
    def star_reduction_amount(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._star_reduction_amount == value:
            return
        self._star_reduction_amount = value
        self.star_reduction_config_changed.emit()

    @property
    def star_reduction_radius(self) -> int:
        return self._star_reduction_radius

    @star_reduction_radius.setter
    def star_reduction_radius(self, value: int) -> None:
        value = max(1, min(10, int(value)))
        if self._star_reduction_radius == value:
            return
        self._star_reduction_radius = value
        self.star_reduction_config_changed.emit()

    @property
    def star_reduction_threshold(self) -> float:
        return self._star_reduction_threshold

    @star_reduction_threshold.setter
    def star_reduction_threshold(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._star_reduction_threshold == value:
            return
        self._star_reduction_threshold = value
        self.star_reduction_config_changed.emit()

    # -- color grading config properties -------------------------------------

    @property
    def color_grading_enabled(self) -> bool:
        return self._color_grading_enabled

    @color_grading_enabled.setter
    def color_grading_enabled(self, value: bool) -> None:
        if self._color_grading_enabled == value:
            return
        self._color_grading_enabled = value
        self.color_grading_config_changed.emit()

    def _cg_shift_getter(self, attr: str) -> float:
        return getattr(self, attr)

    def _cg_shift_setter(self, attr: str, value: float) -> None:
        value = max(-0.5, min(0.5, value))
        if getattr(self, attr) == value:
            return
        setattr(self, attr, value)
        self.color_grading_config_changed.emit()

    @property
    def cg_shadow_r(self) -> float:
        return self._cg_shadow_r

    @cg_shadow_r.setter
    def cg_shadow_r(self, value: float) -> None:
        value = max(-0.5, min(0.5, value))
        if self._cg_shadow_r == value:
            return
        self._cg_shadow_r = value
        self.color_grading_config_changed.emit()

    @property
    def cg_shadow_g(self) -> float:
        return self._cg_shadow_g

    @cg_shadow_g.setter
    def cg_shadow_g(self, value: float) -> None:
        value = max(-0.5, min(0.5, value))
        if self._cg_shadow_g == value:
            return
        self._cg_shadow_g = value
        self.color_grading_config_changed.emit()

    @property
    def cg_shadow_b(self) -> float:
        return self._cg_shadow_b

    @cg_shadow_b.setter
    def cg_shadow_b(self, value: float) -> None:
        value = max(-0.5, min(0.5, value))
        if self._cg_shadow_b == value:
            return
        self._cg_shadow_b = value
        self.color_grading_config_changed.emit()

    @property
    def cg_midtone_r(self) -> float:
        return self._cg_midtone_r

    @cg_midtone_r.setter
    def cg_midtone_r(self, value: float) -> None:
        value = max(-0.5, min(0.5, value))
        if self._cg_midtone_r == value:
            return
        self._cg_midtone_r = value
        self.color_grading_config_changed.emit()

    @property
    def cg_midtone_g(self) -> float:
        return self._cg_midtone_g

    @cg_midtone_g.setter
    def cg_midtone_g(self, value: float) -> None:
        value = max(-0.5, min(0.5, value))
        if self._cg_midtone_g == value:
            return
        self._cg_midtone_g = value
        self.color_grading_config_changed.emit()

    @property
    def cg_midtone_b(self) -> float:
        return self._cg_midtone_b

    @cg_midtone_b.setter
    def cg_midtone_b(self, value: float) -> None:
        value = max(-0.5, min(0.5, value))
        if self._cg_midtone_b == value:
            return
        self._cg_midtone_b = value
        self.color_grading_config_changed.emit()

    @property
    def cg_highlight_r(self) -> float:
        return self._cg_highlight_r

    @cg_highlight_r.setter
    def cg_highlight_r(self, value: float) -> None:
        value = max(-0.5, min(0.5, value))
        if self._cg_highlight_r == value:
            return
        self._cg_highlight_r = value
        self.color_grading_config_changed.emit()

    @property
    def cg_highlight_g(self) -> float:
        return self._cg_highlight_g

    @cg_highlight_g.setter
    def cg_highlight_g(self, value: float) -> None:
        value = max(-0.5, min(0.5, value))
        if self._cg_highlight_g == value:
            return
        self._cg_highlight_g = value
        self.color_grading_config_changed.emit()

    @property
    def cg_highlight_b(self) -> float:
        return self._cg_highlight_b

    @cg_highlight_b.setter
    def cg_highlight_b(self, value: float) -> None:
        value = max(-0.5, min(0.5, value))
        if self._cg_highlight_b == value:
            return
        self._cg_highlight_b = value
        self.color_grading_config_changed.emit()

    # -- stretch config properties --------------------------------------------

    @property
    def stretch_target_background(self) -> float:
        return self._stretch_target_background

    @stretch_target_background.setter
    def stretch_target_background(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._stretch_target_background == value:
            return
        self._stretch_target_background = value
        self.stretch_config_changed.emit()

    @property
    def stretch_shadow_clipping_sigmas(self) -> float:
        return self._stretch_shadow_clipping_sigmas

    @stretch_shadow_clipping_sigmas.setter
    def stretch_shadow_clipping_sigmas(self, value: float) -> None:
        value = max(-10.0, min(0.0, value))
        if self._stretch_shadow_clipping_sigmas == value:
            return
        self._stretch_shadow_clipping_sigmas = value
        self.stretch_config_changed.emit()

    @property
    def stretch_linked_channels(self) -> bool:
        return self._stretch_linked_channels

    @stretch_linked_channels.setter
    def stretch_linked_channels(self, value: bool) -> None:
        if self._stretch_linked_channels == value:
            return
        self._stretch_linked_channels = value
        self.stretch_config_changed.emit()

    # -- curves config properties --------------------------------------------

    @property
    def curves_enabled(self) -> bool:
        return self._curves_enabled

    @curves_enabled.setter
    def curves_enabled(self, value: bool) -> None:
        if self._curves_enabled == value:
            return
        self._curves_enabled = value
        self._update_curves_step_state()
        self.curves_config_changed.emit()

    @property
    def curves_rgb_points(self) -> list[tuple[float, float]]:
        return list(self._curves_rgb_points)

    @curves_rgb_points.setter
    def curves_rgb_points(self, value: list[tuple[float, float]]) -> None:
        self._curves_rgb_points = list(value)
        self.curves_config_changed.emit()

    @property
    def curves_r_points(self) -> list[tuple[float, float]]:
        return list(self._curves_r_points)

    @curves_r_points.setter
    def curves_r_points(self, value: list[tuple[float, float]]) -> None:
        self._curves_r_points = list(value)
        self.curves_config_changed.emit()

    @property
    def curves_g_points(self) -> list[tuple[float, float]]:
        return list(self._curves_g_points)

    @curves_g_points.setter
    def curves_g_points(self, value: list[tuple[float, float]]) -> None:
        self._curves_g_points = list(value)
        self.curves_config_changed.emit()

    @property
    def curves_b_points(self) -> list[tuple[float, float]]:
        return list(self._curves_b_points)

    @curves_b_points.setter
    def curves_b_points(self, value: list[tuple[float, float]]) -> None:
        self._curves_b_points = list(value)
        self.curves_config_changed.emit()

    def _update_curves_step_state(self) -> None:
        step = self.step_by_key("curves")
        if step is None:
            return
        if not self._curves_enabled and step.state is StepState.PENDING:
            step.state = StepState.DISABLED
            self.step_changed.emit("curves", StepState.DISABLED.value)
        elif self._curves_enabled and step.state is StepState.DISABLED:
            step.state = StepState.PENDING
            self.step_changed.emit("curves", StepState.PENDING.value)

    # -- star processing config properties -----------------------------------

    @property
    def star_detection_sigma(self) -> float:
        return self._star_detection_sigma

    @star_detection_sigma.setter
    def star_detection_sigma(self, value: float) -> None:
        value = max(1.0, min(10.0, value))
        if self._star_detection_sigma == value:
            return
        self._star_detection_sigma = value
        self.star_processing_config_changed.emit()

    @property
    def star_min_area(self) -> int:
        return self._star_min_area

    @star_min_area.setter
    def star_min_area(self, value: int) -> None:
        value = max(1, min(500, int(value)))
        if self._star_min_area == value:
            return
        self._star_min_area = value
        self.star_processing_config_changed.emit()

    @property
    def star_max_area(self) -> int:
        return self._star_max_area

    @star_max_area.setter
    def star_max_area(self, value: int) -> None:
        value = max(100, min(50000, int(value)))
        if self._star_max_area == value:
            return
        self._star_max_area = value
        self.star_processing_config_changed.emit()

    @property
    def star_mask_dilation(self) -> int:
        return self._star_mask_dilation

    @star_mask_dilation.setter
    def star_mask_dilation(self, value: int) -> None:
        value = max(0, min(20, int(value)))
        if self._star_mask_dilation == value:
            return
        self._star_mask_dilation = value
        self.star_processing_config_changed.emit()

    @property
    def star_reduce_enabled(self) -> bool:
        return self._star_reduce_enabled

    @star_reduce_enabled.setter
    def star_reduce_enabled(self, value: bool) -> None:
        if self._star_reduce_enabled == value:
            return
        self._star_reduce_enabled = value
        self.star_processing_config_changed.emit()

    @property
    def star_reduce_factor(self) -> float:
        return self._star_reduce_factor

    @star_reduce_factor.setter
    def star_reduce_factor(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if self._star_reduce_factor == value:
            return
        self._star_reduce_factor = value
        self.star_processing_config_changed.emit()

    # -- registration config properties -----------------------------------

    @property
    def registration_upsample_factor(self) -> int:
        return self._registration_upsample_factor

    @registration_upsample_factor.setter
    def registration_upsample_factor(self, value: int) -> None:
        value = max(1, min(100, int(value)))
        if self._registration_upsample_factor == value:
            return
        self._registration_upsample_factor = value
        self.registration_config_changed.emit()

    @property
    def registration_reference_frame_index(self) -> int:
        return self._registration_reference_frame_index

    @registration_reference_frame_index.setter
    def registration_reference_frame_index(self, value: int) -> None:
        value = max(0, int(value))
        if self._registration_reference_frame_index == value:
            return
        self._registration_reference_frame_index = value
        self.registration_config_changed.emit()

    @property
    def registration_method(self) -> str:
        return self._registration_method

    @registration_method.setter
    def registration_method(self, value: str) -> None:
        if value not in ("star", "phase_correlation"):
            value = "star"
        if self._registration_method == value:
            return
        self._registration_method = value
        self.registration_config_changed.emit()

    # -- stacking config properties -----------------------------------

    @property
    def stacking_method(self) -> str:
        return self._stacking_method

    @stacking_method.setter
    def stacking_method(self, value: str) -> None:
        if self._stacking_method == value:
            return
        self._stacking_method = value
        self.stacking_config_changed.emit()

    @property
    def stacking_sigma_low(self) -> float:
        return self._stacking_sigma_low

    @stacking_sigma_low.setter
    def stacking_sigma_low(self, value: float) -> None:
        value = max(0.0, min(10.0, value))
        if self._stacking_sigma_low == value:
            return
        self._stacking_sigma_low = value
        self.stacking_config_changed.emit()

    @property
    def stacking_sigma_high(self) -> float:
        return self._stacking_sigma_high

    @stacking_sigma_high.setter
    def stacking_sigma_high(self, value: float) -> None:
        value = max(0.0, min(10.0, value))
        if self._stacking_sigma_high == value:
            return
        self._stacking_sigma_high = value
        self.stacking_config_changed.emit()

    # -- annotation config properties -----------------------------------

    @property
    def annotation_show_dso(self) -> bool:
        return self._annotation_show_dso

    @annotation_show_dso.setter
    def annotation_show_dso(self, value: bool) -> None:
        value = bool(value)
        if self._annotation_show_dso == value:
            return
        self._annotation_show_dso = value
        self.annotation_config_changed.emit()

    @property
    def annotation_show_stars(self) -> bool:
        return self._annotation_show_stars

    @annotation_show_stars.setter
    def annotation_show_stars(self, value: bool) -> None:
        value = bool(value)
        if self._annotation_show_stars == value:
            return
        self._annotation_show_stars = value
        self.annotation_config_changed.emit()

    @property
    def annotation_show_boundaries(self) -> bool:
        return self._annotation_show_boundaries

    @annotation_show_boundaries.setter
    def annotation_show_boundaries(self, value: bool) -> None:
        value = bool(value)
        if self._annotation_show_boundaries == value:
            return
        self._annotation_show_boundaries = value
        self.annotation_config_changed.emit()

    @property
    def annotation_show_grid(self) -> bool:
        return self._annotation_show_grid

    @annotation_show_grid.setter
    def annotation_show_grid(self, value: bool) -> None:
        value = bool(value)
        if self._annotation_show_grid == value:
            return
        self._annotation_show_grid = value
        self.annotation_config_changed.emit()

    # -- export config properties -------------------------------------------

    @property
    def output_path(self) -> str:
        return self._output_path

    @output_path.setter
    def output_path(self, value: str) -> None:
        if self._output_path == value:
            return
        self._output_path = value
        self.export_config_changed.emit()

    @property
    def output_format(self) -> str:
        return self._output_format

    @output_format.setter
    def output_format(self, value: str) -> None:
        if self._output_format == value:
            return
        self._output_format = value
        self.export_config_changed.emit()

    @property
    def output_filename(self) -> str:
        return self._output_filename

    @output_filename.setter
    def output_filename(self, value: str) -> None:
        if self._output_filename == value:
            return
        self._output_filename = value
        self.export_config_changed.emit()

    # -- export config bridge --

    def export_config(self) -> dict[str, Any]:
        """Return ExportStep-compatible parameters derived from UI state."""
        return {
            "fmt_value": self._output_format,
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
            "comet_stacking": self._comet_stack_enabled,
            "synthetic_flat": self._synthetic_flat_enabled,
            "frame_selection": self._frame_selection_enabled,
            "background_removal": self._background_removal_enabled,
            "curves": self._curves_enabled,
            "bg_neutralization": self._bg_neutralization_enabled,
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

    # -- processing parameter snapshot/restore for undo/redo -------------------

    _SNAPSHOT_ATTRS: tuple[str, ...] = (
        "_stretch_target_background",
        "_stretch_shadow_clipping_sigmas",
        "_stretch_linked_channels",
        "_curves_enabled",
        "_curves_rgb_points",
        "_curves_r_points",
        "_curves_g_points",
        "_curves_b_points",
        "_background_removal_enabled",
        "_background_removal_tile_size",
        "_background_removal_method",
        "_background_removal_preserve_median",
        "_denoise_strength",
        "_denoise_tile_size",
        "_denoise_tile_overlap",
        "_adaptive_denoise_enabled",
        "_sharpening_enabled",
        "_sharpening_radius",
        "_sharpening_amount",
        "_sharpening_threshold",
        "_starless_enabled",
        "_starless_strength",
        "_starless_format",
        "_save_star_mask",
        "_deconvolution_enabled",
        "_deconvolution_iterations",
        "_deconvolution_psf_sigma",
        "_star_detection_sigma",
        "_star_min_area",
        "_star_max_area",
        "_star_mask_dilation",
        "_star_reduce_enabled",
        "_star_reduce_factor",
        "_color_calibration_enabled",
        "_color_calibration_catalog",
        "_color_calibration_sample_radius",
        "_asinh_enabled",
        "_asinh_stretch_factor",
        "_asinh_black_point",
        "_asinh_linked",
        "_mtf_enabled",
        "_mtf_midpoint",
        "_mtf_shadows_clipping",
        "_mtf_highlights",
        "_bg_neutralization_enabled",
        "_bg_neutralization_sample_mode",
        "_bg_neutralization_target",
        "_bg_neutralization_roi",
        "_clahe_enabled",
        "_clahe_clip_limit",
        "_clahe_tile_size",
        "_clahe_channel_mode",
        "_color_grading_enabled",
        "_cg_shadow_r",
        "_cg_shadow_g",
        "_cg_shadow_b",
        "_cg_midtone_r",
        "_cg_midtone_g",
        "_cg_midtone_b",
        "_cg_highlight_r",
        "_cg_highlight_g",
        "_cg_highlight_b",
    )

    def snapshot_processing_params(self) -> dict[str, object]:
        snap: dict[str, object] = {}
        for attr in self._SNAPSHOT_ATTRS:
            val = getattr(self, attr)
            if isinstance(val, list):
                val = [tuple(p) if isinstance(p, (list, tuple)) else p for p in val]
            snap[attr] = val
        return snap

    def restore_processing_params(self, params: dict[str, object]) -> None:
        for attr in self._SNAPSHOT_ATTRS:
            if attr in params:
                val = params[attr]
                if isinstance(val, list):
                    val = [tuple(p) if isinstance(p, (list, tuple)) else p for p in val]
                setattr(self, attr, val)
