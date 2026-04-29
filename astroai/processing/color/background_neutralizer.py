"""Background Neutralization: removes color casts from sky background."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BackgroundNeutralizationConfig",
    "BackgroundNeutralizer",
    "BackgroundNeutralizationStep",
    "SampleMode",
]

logger = logging.getLogger(__name__)


class SampleMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


@dataclass(frozen=True)
class BackgroundNeutralizationConfig:
    """Configuration for background color neutralization.

    Auto mode samples the darkest 2% pixels per channel to estimate the
    sky background; manual mode uses a user-specified ROI rectangle.
    """

    sample_mode: SampleMode = SampleMode.AUTO
    target_background: float = 0.1
    """Target sky-background level ∈ [0.0, 0.3]."""

    roi: tuple[int, int, int, int] | None = None
    """(row_start, row_end, col_start, col_end) for manual mode. None → auto."""

    sample_percentile: float = 2.0
    """Percentile used in AUTO mode to identify background pixels (default 2%)."""

    def __post_init__(self) -> None:
        if not (0.0 <= self.target_background <= 0.3):
            raise ValueError(
                f"target_background must be in [0.0, 0.3], got {self.target_background!r}"
            )
        if not (0.1 <= self.sample_percentile <= 20.0):
            raise ValueError(
                f"sample_percentile must be in [0.1, 20.0], got {self.sample_percentile!r}"
            )
        if self.sample_mode is SampleMode.MANUAL and self.roi is not None:
            r0, r1, c0, c1 = self.roi
            if r0 >= r1 or c0 >= c1:
                raise ValueError(
                    f"roi must satisfy row_start < row_end and col_start < col_end, got {self.roi!r}"
                )

    def is_identity(self) -> bool:
        """True when target_background == 0.0 (no shift applied)."""
        return abs(self.target_background) < 1e-9

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_mode": self.sample_mode.value,
            "target_background": self.target_background,
            "roi": self.roi,
            "sample_percentile": self.sample_percentile,
        }


class BackgroundNeutralizer:
    """Estimate and neutralize sky-background color cast per channel."""

    def __init__(self, config: BackgroundNeutralizationConfig | None = None) -> None:
        self.config = config or BackgroundNeutralizationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, image: NDArray[Any]) -> NDArray[Any]:
        """Return background-neutralized copy of *image*.

        Parameters
        ----------
        image:
            Float array (H, W) grayscale or (H, W, 3) RGB, values in [0, 1].

        Returns
        -------
        NDArray
            Corrected image, same shape and dtype, clipped to [0, 1].
        """
        arr = np.asarray(image, dtype=np.float64)

        if arr.ndim == 2:
            return self._apply_single_channel(arr, image)

        if arr.ndim == 3 and arr.shape[2] == 3:
            return self._apply_rgb(arr, image)

        logger.warning("BackgroundNeutralizer: unsupported shape %s, returning unchanged", arr.shape)
        return image  # type: ignore[return-value]

    def estimate_background(self, image: NDArray[Any]) -> NDArray[Any]:
        """Return per-channel background estimate as a 1-D array of length C (or 1 for grayscale).

        Values are in [0, 1].
        """
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 2:
            return np.array([self._channel_background(arr)])
        if arr.ndim == 3 and arr.shape[2] == 3:
            return np.array([self._channel_background(arr[:, :, c]) for c in range(3)])
        return np.array([0.0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _channel_background(self, channel: NDArray[Any]) -> float:
        """Estimate background level for a single 2-D channel."""
        cfg = self.config
        if cfg.sample_mode is SampleMode.MANUAL and cfg.roi is not None:
            r0, r1, c0, c1 = cfg.roi
            # Clamp to array bounds
            h, w = channel.shape
            r0, r1 = max(0, r0), min(h, r1)
            c0, c1 = max(0, c0), min(w, c1)
            region = channel[r0:r1, c0:c1]
            if region.size == 0:
                region = channel
            return float(np.median(region))

        # AUTO: percentile of darkest pixels
        flat = channel.ravel()
        threshold = float(np.percentile(flat, cfg.sample_percentile))
        bg_pixels = flat[flat <= threshold]
        if bg_pixels.size == 0:
            return float(np.percentile(flat, cfg.sample_percentile))
        return float(np.median(bg_pixels))

    def _apply_single_channel(
        self, arr: NDArray[Any], original: NDArray[Any]
    ) -> NDArray[Any]:
        bg = self._channel_background(arr)
        shift = self.config.target_background - bg
        result = np.clip(arr + shift, 0.0, 1.0)
        return result.astype(original.dtype)

    def _apply_rgb(
        self, arr: NDArray[Any], original: NDArray[Any]
    ) -> NDArray[Any]:
        out = arr.copy()
        for c in range(3):
            bg = self._channel_background(arr[:, :, c])
            shift = self.config.target_background - bg
            out[:, :, c] = np.clip(arr[:, :, c] + shift, 0.0, 1.0)
        return out.astype(original.dtype)


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


class BackgroundNeutralizationStep(PipelineStep):
    """Apply background color neutralization as a pipeline step.

    Sits after WhiteBalanceStep and before AsinHStep in the processing pipeline.
    """

    def __init__(self, config: BackgroundNeutralizationConfig | None = None) -> None:
        self._neutralizer = BackgroundNeutralizer(config)

    @property
    def name(self) -> str:
        return "Hintergrundneutralisierung"

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
            logger.warning("BackgroundNeutralizationStep: no image in context, skipping")
            return context

        progress(PipelineProgress(
            stage=self.stage, current=0, total=2,
            message="Hintergrund analysieren…",
        ))

        bg_estimate = self._neutralizer.estimate_background(image)
        context.metadata["background_neutralization_estimate"] = bg_estimate

        progress(PipelineProgress(
            stage=self.stage, current=1, total=2,
            message="Hintergrundfarbstich entfernen…",
        ))

        context.result = self._neutralizer.apply(image)

        progress(PipelineProgress(
            stage=self.stage, current=2, total=2,
            message="Hintergrundneutralisierung abgeschlossen",
        ))

        return context
