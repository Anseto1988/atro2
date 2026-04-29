"""Shadow/Midtone/Highlight color grading — additive per-zone RGB shifts."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["ColorGradingConfig", "ColorGrader", "ColorGradingStep"]

logger = logging.getLogger(__name__)

_SHIFT_FIELDS = (
    "shadow_r", "shadow_g", "shadow_b",
    "midtone_r", "midtone_g", "midtone_b",
    "highlight_r", "highlight_g", "highlight_b",
)


@dataclass(frozen=True)
class ColorGradingConfig:
    """Additive RGB color shifts for shadows, midtones and highlights.

    All shift values must be in [-0.5, 0.5].
    """

    shadow_r: float = 0.0
    shadow_g: float = 0.0
    shadow_b: float = 0.0
    midtone_r: float = 0.0
    midtone_g: float = 0.0
    midtone_b: float = 0.0
    highlight_r: float = 0.0
    highlight_g: float = 0.0
    highlight_b: float = 0.0

    def __post_init__(self) -> None:
        for name in _SHIFT_FIELDS:
            val = getattr(self, name)
            if not (-0.5 <= val <= 0.5):
                raise ValueError(
                    f"ColorGradingConfig.{name} must be in [-0.5, 0.5], got {val!r}"
                )

    def is_identity(self) -> bool:
        """True when all shifts are approximately 0.0 (atol=1e-5)."""
        return all(abs(getattr(self, f)) < 1e-5 for f in _SHIFT_FIELDS)

    def as_dict(self) -> dict[str, float]:
        """Return all fields as a plain dict."""
        return {f: getattr(self, f) for f in _SHIFT_FIELDS}


class ColorGrader:
    """Apply shadow/midtone/highlight color grading to an image."""

    def __init__(self, config: ColorGradingConfig | None = None) -> None:
        self.config = config or ColorGradingConfig()

    def grade(self, image: NDArray) -> NDArray:
        """Return a color-graded copy of *image*.

        Grayscale images (2-D or H×W×1) are returned unchanged.
        """
        arr = np.asarray(image)

        # Grayscale: return unchanged
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            logger.warning("ColorGrader: grayscale image passed, returning unchanged")
            return image  # type: ignore[return-value]

        if self.config.is_identity():
            return image  # type: ignore[return-value]

        cfg = self.config
        out = arr.astype(np.float64, copy=True)

        # Luminance per pixel  (H×W)
        lum = (
            0.2126 * arr[:, :, 0].astype(np.float64)
            + 0.7152 * arr[:, :, 1].astype(np.float64)
            + 0.0722 * arr[:, :, 2].astype(np.float64)
        )

        # Tone masks
        shadow = (1.0 - lum) ** 2
        highlight = lum ** 2
        midtone_raw = 4.0 * lum * (1.0 - lum) ** 1.5
        mt_max = midtone_raw.max()
        midtone = midtone_raw / mt_max if mt_max > 0 else midtone_raw

        # Apply additive shifts per channel
        # Red
        out[:, :, 0] += (
            shadow * cfg.shadow_r
            + midtone * cfg.midtone_r
            + highlight * cfg.highlight_r
        )
        # Green
        out[:, :, 1] += (
            shadow * cfg.shadow_g
            + midtone * cfg.midtone_g
            + highlight * cfg.highlight_g
        )
        # Blue
        out[:, :, 2] += (
            shadow * cfg.shadow_b
            + midtone * cfg.midtone_b
            + highlight * cfg.highlight_b
        )

        np.clip(out, 0.0, 1.0, out=out)
        return out.astype(arr.dtype)


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

from astroai.core.pipeline.base import (  # noqa: E402
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)


class ColorGradingStep(PipelineStep):
    """Apply shadow/midtone/highlight color grading as a pipeline step."""

    def __init__(self, config: ColorGradingConfig | None = None) -> None:
        self._grader = ColorGrader(config)

    @property
    def name(self) -> str:
        return "Farbabstufung"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        image = context.result if context.result is not None else (
            context.images[0] if context.images else None
        )
        if image is None:
            logger.warning("ColorGradingStep: no image in context, skipping")
            return context

        progress(PipelineProgress(stage=self.stage, current=0, total=1, message="Farbabstufung anwenden…"))
        context.result = self._grader.grade(image)
        progress(PipelineProgress(stage=self.stage, current=1, total=1, message="Farbabstufung abgeschlossen"))
        return context
