"""Midtone Transfer Function (MTF) / Histogram Transformation stretch for astrophotography."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["MidtoneTransferConfig", "MidtoneTransferFunction", "MTFStep"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MidtoneTransferConfig:
    """Configuration for MTF histogram transformation.

    Formula: f(x) = ((m-1)*x) / ((2*m-1)*x - m)
    where m = midpoint. Edge cases: f(0)=0, f(1)=1.
    """

    midpoint: float = 0.25
    """Midtone balance m ∈ [0.001, 0.499]. Controls the S-curve shape."""

    shadows_clipping: float = 0.0
    """Shadow clipping threshold ∈ [0.0, 0.1]. Pixels below are mapped to 0."""

    highlights: float = 1.0
    """Highlights clipping threshold ∈ [0.98, 1.0]. Pixels above are mapped to 1."""

    def __post_init__(self) -> None:
        if not (0.001 <= self.midpoint <= 0.499):
            raise ValueError(
                f"midpoint must be in [0.001, 0.499], got {self.midpoint!r}"
            )
        if not (0.0 <= self.shadows_clipping <= 0.1):
            raise ValueError(
                f"shadows_clipping must be in [0.0, 0.1], got {self.shadows_clipping!r}"
            )
        if not (0.98 <= self.highlights <= 1.0):
            raise ValueError(
                f"highlights must be in [0.98, 1.0], got {self.highlights!r}"
            )

    def is_identity(self) -> bool:
        """True when transformation is near-linear (midpoint≥0.499, no clipping)."""
        return (
            abs(self.midpoint - 0.5) < 2e-3
            and abs(self.shadows_clipping) < 1e-6
            and abs(self.highlights - 1.0) < 1e-6
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "midpoint": self.midpoint,
            "shadows_clipping": self.shadows_clipping,
            "highlights": self.highlights,
        }


def _apply_mtf_array(data: NDArray[Any], m: float) -> NDArray[Any]:
    """Vectorised MTF: f(x) = ((m-1)*x) / ((2*m-1)*x - m).

    Exact edge cases: x=0 → 0, x=1 → 1.
    """
    numerator = (m - 1.0) * data
    denominator = (2.0 * m - 1.0) * data - m
    safe_denom = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    result = numerator / safe_denom
    result = np.where(data <= 0.0, 0.0, result)
    result = np.where(data >= 1.0, 1.0, result)
    return np.clip(result, 0.0, 1.0)


class MidtoneTransferFunction:
    """Apply MTF histogram transformation to astrophotography images."""

    def __init__(self, config: MidtoneTransferConfig | None = None) -> None:
        self.config = config or MidtoneTransferConfig()

    def apply(self, image: NDArray[Any]) -> NDArray[Any]:
        """Stretch *image* using the MTF curve.

        Parameters
        ----------
        image:
            Float array (H, W) or (H, W, 3), values in [0, 1].

        Returns
        -------
        NDArray
            Stretched image, same shape and dtype.
        """
        original_dtype = image.dtype
        img = np.asarray(image, dtype=np.float64)
        cfg = self.config

        # Normalize to [0, 1] within [shadows_clipping, highlights]
        sc = cfg.shadows_clipping
        hl = cfg.highlights
        rng = hl - sc
        if rng < 1e-10:
            return np.zeros_like(image)

        img = np.clip((img - sc) / rng, 0.0, 1.0)
        result = _apply_mtf_array(img, cfg.midpoint)
        return result.astype(original_dtype)

    @staticmethod
    def compute_midpoint_from_background(
        background_median: float,
        target_background: float = 0.25,
    ) -> float:
        """Compute optimal midpoint m so that MTF(background_median) ≈ target_background.

        Uses the inverse MTF (PixInsight Auto-STF) formula:
            m = (target*(bg-1)) / ((2*target-1)*bg - target)

        Parameters
        ----------
        background_median:
            Median pixel value of the sky background, normalised to [0, 1].
        target_background:
            Desired output background level (default 0.25).

        Returns
        -------
        float
            Optimal midpoint clamped to [0.001, 0.499].
        """
        bg = float(np.clip(background_median, 1e-6, 1.0 - 1e-6))
        if bg >= target_background:
            return 0.25  # already bright enough — return default
        m = (target_background * (bg - 1.0)) / (
            (2.0 * target_background - 1.0) * bg - target_background
        )
        return float(np.clip(m, 0.001, 0.499))

    @staticmethod
    def estimate_background(image: NDArray[Any]) -> float:
        """Estimate background as median of pixel values below the global median.

        Parameters
        ----------
        image:
            Float array (H, W) or (H, W, 3).

        Returns
        -------
        float
            Background median estimate in [0, 1].
        """
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)
        flat = arr.ravel()
        median = float(np.median(flat))
        bg_pixels = flat[flat < median]
        if len(bg_pixels) == 0:
            return median
        return float(np.median(bg_pixels))


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


class MTFStep(PipelineStep):
    """Apply Midtone Transfer Function stretch as a pipeline step."""

    def __init__(self, config: MidtoneTransferConfig | None = None) -> None:
        self._mtf = MidtoneTransferFunction(config)

    @property
    def name(self) -> str:
        return "MTF-Stretch"

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
            logger.warning("MTFStep: no image in context, skipping")
            return context

        progress(PipelineProgress(
            stage=self.stage,
            current=0,
            total=1,
            message="MTF-Stretch anwenden…",
        ))
        context.result = self._mtf.apply(image)
        progress(PipelineProgress(
            stage=self.stage,
            current=1,
            total=1,
            message="MTF-Stretch abgeschlossen",
        ))
        return context
