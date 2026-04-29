"""Star reduction via morphological minimum filter (shrinks stars without removing them)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, minimum_filter

__all__ = ["StarReductionConfig", "StarReducer", "StarReductionStep"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StarReductionConfig:
    """Configuration for morphological star size reduction."""

    amount: float = 0.5
    """Blend strength of the reduction effect [0.0, 1.0]."""

    radius: int = 2
    """Kernel radius for the minimum filter; kernel size = 2*radius+1. [1, 10]"""

    threshold: float = 0.5
    """Brightness threshold for star detection [0.0, 1.0].
    Pixels above this value are considered stars."""

    def __post_init__(self) -> None:
        if not (0.0 <= self.amount <= 1.0):
            raise ValueError(f"amount must be in [0.0, 1.0], got {self.amount!r}")
        if not (1 <= self.radius <= 10):
            raise ValueError(f"radius must be in [1, 10], got {self.radius!r}")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {self.threshold!r}")

    def is_identity(self) -> bool:
        """True when amount ≈ 0.0 (atol=1e-4) — no visible effect."""
        return abs(self.amount) < 1e-4

    def as_dict(self) -> dict[str, float | int]:
        """Return all fields as a plain dict."""
        return {
            "amount": self.amount,
            "radius": self.radius,
            "threshold": self.threshold,
        }


class StarReducer:
    """Apply star reduction to an astronomical image."""

    def __init__(self, config: StarReductionConfig | None = None) -> None:
        self.config = config or StarReductionConfig()

    def reduce(self, image: NDArray) -> NDArray:
        """Return a copy of *image* with stars shrunk according to config.

        Supports H×W (grayscale) and H×W×3 (RGB) float images in [0, 1].
        """
        arr = np.asarray(image, dtype=np.float64)
        orig_dtype = image.dtype

        if self.config.is_identity():
            return image  # type: ignore[return-value]

        is_gray = arr.ndim == 2
        if is_gray:
            arr = arr[:, :, np.newaxis]  # treat as H×W×1

        kernel_size = 2 * self.config.radius + 1

        # Build soft star mask from first channel (or luminance-like)
        # Use max across channels so stars on any channel are captured
        lum = arr.max(axis=2)

        # Hard mask: pixels above threshold
        hard_mask = (lum > self.config.threshold).astype(np.float64)

        # Smooth the hard mask to get soft transitions
        soft_mask = gaussian_filter(hard_mask, sigma=self.config.radius * 0.5)
        soft_mask = np.clip(soft_mask, 0.0, 1.0)

        # Apply minimum filter per channel to erode (shrink) bright regions
        filtered = np.empty_like(arr)
        for c in range(arr.shape[2]):
            filtered[:, :, c] = minimum_filter(arr[:, :, c], size=kernel_size)

        # Blend: where star mask is active, replace original with filtered
        alpha = soft_mask[:, :, np.newaxis] * self.config.amount
        out = arr * (1.0 - alpha) + filtered * alpha

        out = np.clip(out, 0.0, 1.0)

        if is_gray:
            out = out[:, :, 0]

        return out.astype(orig_dtype)


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


class StarReductionStep(PipelineStep):
    """Reduce star sizes as a pipeline step."""

    def __init__(self, config: StarReductionConfig | None = None) -> None:
        self._reducer = StarReducer(config)

    @property
    def name(self) -> str:
        return "Sternreduktion"

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
            logger.warning("StarReductionStep: no image in context, skipping")
            return context

        progress(PipelineProgress(stage=self.stage, current=0, total=1, message="Sternreduktion anwenden…"))
        context.result = self._reducer.reduce(image)
        progress(PipelineProgress(stage=self.stage, current=1, total=1, message="Sternreduktion abgeschlossen"))
        return context
