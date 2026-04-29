"""Manual per-channel white balance (R/G/B multiplier) adjustment."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["WhiteBalanceAdjustment", "WhiteBalanceConfig", "WhiteBalanceStep"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhiteBalanceConfig:
    """Per-channel RGB multipliers for white balance correction."""

    red_factor: float = 1.0
    green_factor: float = 1.0
    blue_factor: float = 1.0

    def __post_init__(self) -> None:
        for name, val in (
            ("red_factor", self.red_factor),
            ("green_factor", self.green_factor),
            ("blue_factor", self.blue_factor),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val!r}")

    def is_identity(self) -> bool:
        """True when all factors are approximately 1.0 (atol=1e-4)."""
        return all(abs(v - 1.0) < 1e-4 for v in (self.red_factor, self.green_factor, self.blue_factor))

    def as_dict(self) -> dict[str, float]:
        """Return all fields as a plain dict."""
        return {
            "red_factor": self.red_factor,
            "green_factor": self.green_factor,
            "blue_factor": self.blue_factor,
        }


class WhiteBalanceAdjustment:
    """Apply per-channel RGB multiplier white balance."""

    def __init__(self, config: WhiteBalanceConfig | None = None) -> None:
        self.config = config or WhiteBalanceConfig()

    def apply(self, image: NDArray) -> NDArray:
        """Return a white-balance-adjusted copy of *image*."""
        arr = np.asarray(image)

        # Grayscale: return unchanged
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            logger.warning("WhiteBalanceAdjustment: grayscale image passed, returning unchanged")
            return image  # type: ignore[return-value]

        if self.config.is_identity():
            return image  # type: ignore[return-value]

        out = arr.copy()
        out[:, :, 0] = np.clip(arr[:, :, 0] * self.config.red_factor, 0.0, 1.0)
        out[:, :, 1] = np.clip(arr[:, :, 1] * self.config.green_factor, 0.0, 1.0)
        out[:, :, 2] = np.clip(arr[:, :, 2] * self.config.blue_factor, 0.0, 1.0)
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


class WhiteBalanceStep(PipelineStep):
    """Apply per-channel white balance as a pipeline step."""

    def __init__(self, config: WhiteBalanceConfig | None = None) -> None:
        self._adj = WhiteBalanceAdjustment(config)

    @property
    def name(self) -> str:
        return "Weißabgleich"

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
            logger.warning("WhiteBalanceStep: no image in context, skipping")
            return context

        progress(PipelineProgress(stage=self.stage, current=0, total=1, message="Weißabgleich anpassen…"))
        context.result = self._adj.apply(image)
        progress(PipelineProgress(stage=self.stage, current=1, total=1, message="Weißabgleich abgeschlossen"))
        return context
