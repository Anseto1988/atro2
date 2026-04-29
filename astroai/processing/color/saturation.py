"""Selective HSV-based color saturation adjustment for astrophotography."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["SaturationAdjustment", "SaturationConfig", "SaturationStep"]

# Hue ranges as (center, half-width) in normalised [0, 1] hue space.
# Reds wrap around 0, so they are handled separately.
_RANGES: dict[str, tuple[float, float]] = {
    "reds":    (0.000, 0.083),   # 0° ± 30°  (wraps)
    "oranges": (0.083, 0.042),   # 30° ± 15°
    "yellows": (0.167, 0.042),   # 60° ± 15°
    "greens":  (0.333, 0.125),   # 120° ± 45°
    "cyans":   (0.500, 0.042),   # 180° ± 15°
    "blues":   (0.667, 0.083),   # 240° ± 30°
    "purples": (0.833, 0.083),   # 300° ± 30°
}


@dataclass
class SaturationConfig:
    """Per-range saturation multipliers and a global multiplier.

    Each per-range value is a multiplier [0, 4]; 1.0 means no change.
    """

    global_saturation: float = 1.0
    reds: float = 1.0
    oranges: float = 1.0
    yellows: float = 1.0
    greens: float = 1.0
    cyans: float = 1.0
    blues: float = 1.0
    purples: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in ("reds", "oranges", "yellows", "greens", "cyans", "blues", "purples")}

    def is_identity(self) -> bool:
        return all(v == 1.0 for v in (self.global_saturation, *self.as_dict().values()))


class SaturationAdjustment:
    """Apply selective saturation adjustments via HSV colour space.

    Uses a smooth triangular weighting so adjacent hue ranges blend naturally.
    """

    def __init__(self, config: SaturationConfig | None = None) -> None:
        self.config = config or SaturationConfig()

    def apply(self, image: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return a saturation-adjusted copy of *image* (H × W × 3, float [0,1])."""
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got shape {img.shape}")
        if self.config.is_identity():
            return image.copy()  # type: ignore[return-value]

        hsv = self._rgb_to_hsv(img)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        multiplier = self._compute_multiplier(h)
        s_new = np.clip(s * multiplier * self.config.global_saturation, 0.0, 1.0)

        hsv_out = np.stack([h, s_new, v], axis=2)
        return self._hsv_to_rgb(hsv_out)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_multiplier(self, h: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute a per-pixel saturation multiplier using weighted sum of ranges."""
        weights_total = np.zeros_like(h)
        weighted_mult = np.zeros_like(h)

        per_range = self.config.as_dict()
        for name, (center, hw) in _RANGES.items():
            w = self._triangular_weight(h, center, hw)
            weights_total += w
            weighted_mult += w * per_range[name]

        # Pixels with no significant range weight (achromatic or edge) → 1.0
        return np.where(weights_total > 1e-8, weighted_mult / weights_total, 1.0)

    @staticmethod
    def _triangular_weight(
        h: NDArray[np.float64], center: float, half_width: float
    ) -> NDArray[np.float64]:
        """Triangular hat kernel with circular distance on [0, 1] hue."""
        # Circular distance
        d = np.abs(h - center)
        d = np.minimum(d, 1.0 - d)
        return np.maximum(0.0, 1.0 - d / half_width)

    @staticmethod
    def _rgb_to_hsv(img: NDArray[np.float64]) -> NDArray[np.float64]:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        diff = maxc - minc

        v = maxc
        s = np.where(maxc > 0, diff / maxc, 0.0)

        safe = diff > 0
        rc = np.where(safe, (maxc - r) / diff, 0.0)
        gc = np.where(safe, (maxc - g) / diff, 0.0)
        bc = np.where(safe, (maxc - b) / diff, 0.0)

        h = np.where(
            r == maxc, bc - gc,
            np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc),
        )
        h = (h / 6.0) % 1.0

        return np.stack([h, s, v], axis=2)

    @staticmethod
    def _hsv_to_rgb(hsv: NDArray[np.float64]) -> NDArray[np.float64]:
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        i = (h * 6.0).astype(int) % 6
        f = h * 6.0 - np.floor(h * 6.0)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
        g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
        b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])

        return np.clip(np.stack([r, g, b], axis=2), 0.0, 1.0)


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

import logging  # noqa: E402

logger = logging.getLogger(__name__)


class SaturationStep(PipelineStep):
    """Apply selective HSV saturation adjustment as a pipeline step."""

    def __init__(self, config: SaturationConfig | None = None) -> None:
        self._adj = SaturationAdjustment(config)

    @property
    def name(self) -> str:
        return "Selektive Sättigung"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        if context.result is not None:
            progress(PipelineProgress(stage=self.stage, current=0, total=1, message="Sättigung anpassen…"))
            context.result = self._adj.apply(context.result)
            progress(PipelineProgress(stage=self.stage, current=1, total=1, message="Sättigung abgeschlossen"))
        return context
