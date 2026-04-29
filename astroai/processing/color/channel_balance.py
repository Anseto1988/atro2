"""Per-channel R/G/B(/L) background-level balance via addition offsets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["ChannelBalanceConfig", "ChannelBalancer"]

logger = logging.getLogger(__name__)

_DEFAULT_PERCENTILE = 5.0


@dataclass(frozen=True)
class ChannelBalanceConfig:
    """Per-channel addition offsets for sky-background level correction."""

    r_offset: float = 0.0
    g_offset: float = 0.0
    b_offset: float = 0.0
    l_offset: float = 0.0  # for grayscale images
    sample_percentile: float = _DEFAULT_PERCENTILE

    def __post_init__(self) -> None:
        if not (0.1 <= self.sample_percentile <= 25.0):
            raise ValueError(
                f"sample_percentile must be in [0.1, 25.0], got {self.sample_percentile!r}"
            )

    def is_identity(self) -> bool:
        return all(
            abs(v) < 1e-6
            for v in (self.r_offset, self.g_offset, self.b_offset, self.l_offset)
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "r_offset": self.r_offset,
            "g_offset": self.g_offset,
            "b_offset": self.b_offset,
            "l_offset": self.l_offset,
            "sample_percentile": self.sample_percentile,
        }


def _bg_median(channel: NDArray[Any], percentile: float) -> float:
    """Return median of the darkest *percentile*% pixels."""
    flat = channel.ravel().astype(np.float64)
    threshold = float(np.percentile(flat, percentile))
    dark = flat[flat <= threshold]
    return float(np.median(dark)) if dark.size > 0 else float(threshold)


class ChannelBalancer:
    """Apply per-channel addition-offset background balance (numpy-only)."""

    def __init__(self, config: ChannelBalanceConfig | None = None) -> None:
        self.config = config or ChannelBalanceConfig()

    def apply(self, image: NDArray[Any]) -> NDArray[Any]:
        """Return offset-corrected copy of *image*, clipped to [0, 1]."""
        arr = np.asarray(image, dtype=np.float64)

        if arr.ndim == 2:
            out = np.clip(arr + self.config.l_offset, 0.0, 1.0)
            return out.astype(image.dtype)

        if arr.ndim == 3 and arr.shape[2] >= 3:
            out = arr.copy()
            for c, off in enumerate(
                (self.config.r_offset, self.config.g_offset, self.config.b_offset)
            ):
                out[:, :, c] = np.clip(arr[:, :, c] + off, 0.0, 1.0)
            return out.astype(image.dtype)

        logger.warning("ChannelBalancer: unsupported shape %s, returning unchanged", arr.shape)
        return image  # type: ignore[return-value]

    def auto_sample(self, image: NDArray[Any]) -> ChannelBalanceConfig:
        """Compute balancing offsets from the darkest-pixel background estimate.

        Equalises all channels to the channel with the lowest background level
        (adds to the brighter channels so no data is clipped).
        """
        arr = np.asarray(image, dtype=np.float64)
        pct = self.config.sample_percentile

        if arr.ndim == 2:
            bg = _bg_median(arr, pct)
            return ChannelBalanceConfig(l_offset=-bg, sample_percentile=pct)

        if arr.ndim == 3 and arr.shape[2] >= 3:
            bgs = [_bg_median(arr[:, :, c], pct) for c in range(3)]
            min_bg = min(bgs)
            offsets = [min_bg - b for b in bgs]  # negative: pulls brighter channels down
            return ChannelBalanceConfig(
                r_offset=offsets[0],
                g_offset=offsets[1],
                b_offset=offsets[2],
                sample_percentile=pct,
            )

        return ChannelBalanceConfig(sample_percentile=pct)
