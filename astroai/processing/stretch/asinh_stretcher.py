"""Arcsinh (sinh^-1) stretch for astrophotography — preserves star photometry."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["AsinHConfig", "AsinHStep", "AsinHStretcher"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsinHConfig:
    """Configuration for arcsinh stretch.

    The formula is: output = arcsinh(β × x) / arcsinh(β)
    where β is the stretch_factor.
    """

    stretch_factor: float = 1.0
    """β parameter [0.001, 1000.0]. Must be > 0."""

    black_point: float = 0.0
    """Black-point subtraction [0.0, 0.5]. Applied before stretch."""

    linked_channels: bool = True
    """True = stretch all RGB channels with the same parameters."""

    def __post_init__(self) -> None:
        if self.stretch_factor <= 0:
            raise ValueError(
                f"stretch_factor must be > 0, got {self.stretch_factor!r}"
            )
        if self.black_point < 0:
            raise ValueError(
                f"black_point must be >= 0, got {self.black_point!r}"
            )
        if self.black_point > 0.5:
            raise ValueError(
                f"black_point must be <= 0.5, got {self.black_point!r}"
            )

    def is_identity(self) -> bool:
        """True when stretch_factor ≈ 1.0 AND black_point ≈ 0.0 (atol=1e-6)."""
        return (
            abs(self.stretch_factor - 1.0) < 1e-6
            and abs(self.black_point) < 1e-6
        )

    def as_dict(self) -> dict[str, object]:
        """Return all fields as a plain dict."""
        return {
            "stretch_factor": self.stretch_factor,
            "black_point": self.black_point,
            "linked_channels": self.linked_channels,
        }


def _apply_asinh(x: NDArray, beta: float) -> NDArray:
    """Apply arcsinh stretch formula to a single channel or array."""
    if beta < 1e-9:
        # Near-zero beta: arcsinh(β·x)/arcsinh(β) → x (linear passthrough)
        return x
    return np.arcsinh(beta * x) / np.arcsinh(beta)


class AsinHStretcher:
    """Apply arcsinh stretch to astro images."""

    def __init__(self, config: AsinHConfig | None = None) -> None:
        self.config = config or AsinHConfig()

    def stretch(self, image: NDArray) -> NDArray:
        """Stretch a linear image using the arcsinh formula.

        Parameters
        ----------
        image:
            Float array of shape (H, W) or (H, W, 3) with values in [0, 1].

        Returns
        -------
        NDArray
            Stretched image, same shape and dtype as input.
        """
        original_dtype = image.dtype
        img = image.astype(np.float64)

        cfg = self.config
        beta = cfg.stretch_factor

        # Subtract black point and clip to [0, 1]
        if cfg.black_point > 0:
            img = np.clip(img - cfg.black_point, 0.0, 1.0)

        if img.ndim == 2:
            result = np.clip(_apply_asinh(img, beta), 0.0, 1.0)
            return result.astype(original_dtype)

        # RGB image (H, W, 3)
        if cfg.linked_channels:
            result = np.clip(_apply_asinh(img, beta), 0.0, 1.0)
        else:
            result = np.empty_like(img)
            for c in range(img.shape[2]):
                result[..., c] = np.clip(_apply_asinh(img[..., c], beta), 0.0, 1.0)

        return result.astype(original_dtype)


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


class AsinHStep(PipelineStep):
    """Apply arcsinh stretch as a pipeline step."""

    def __init__(self, config: AsinHConfig | None = None) -> None:
        self._stretcher = AsinHStretcher(config)

    @property
    def name(self) -> str:
        return "Arcsinh Stretch"

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
            logger.warning("AsinHStep: no image in context, skipping")
            return context

        progress(PipelineProgress(
            stage=self.stage,
            current=0,
            total=1,
            message="Arcsinh-Stretch anwenden…",
        ))
        context.result = self._stretcher.stretch(image)
        progress(PipelineProgress(
            stage=self.stage,
            current=1,
            total=1,
            message="Arcsinh-Stretch abgeschlossen",
        ))
        return context
