"""LRGB channel combiner for astrophotography compositing."""
from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["ChannelCombiner"]

_EPS = np.finfo(np.float32).tiny


class ChannelCombiner:
    """Combines separate L, R, G, B grayscale frames into an LRGB colour image."""

    def combine_lrgb(
        self,
        L: NDArray[np.floating[Any]] | None,
        R: NDArray[np.floating[Any]] | None,
        G: NDArray[np.floating[Any]] | None,
        B: NDArray[np.floating[Any]] | None,
    ) -> NDArray[np.float32]:
        """Return an (H, W, 3) float32 image in [0, 1].

        Missing colour channels fall back to zero; a missing L returns the RGB
        luminance unmodified.  At least one channel must be provided.
        """
        channels = [L, R, G, B]
        provided = [c for c in channels if c is not None]
        if not provided:
            raise ValueError("At least one channel must be provided.")

        h, w = provided[0].shape[:2]

        def _prep(c: NDArray[np.floating[Any]] | None) -> NDArray[np.float32]:
            if c is None:
                return np.zeros((h, w), dtype=np.float32)
            arr = np.asarray(c, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            return np.clip(arr, 0.0, 1.0)

        r_f = _prep(R)
        g_f = _prep(G)
        b_f = _prep(B)

        rgb = np.stack([r_f, g_f, b_f], axis=-1)  # (H, W, 3)

        if L is None:
            return rgb

        lum_f = _prep(L)
        # Luminance of the colour channels
        lum_rgb = 0.299 * r_f + 0.587 * g_f + 0.114 * b_f
        # Scale colour channels so their luminance matches L
        scale = lum_f / (lum_rgb + _EPS)
        result = np.clip(rgb * scale[..., np.newaxis], 0.0, 1.0)
        return cast(NDArray[np.float32], result.astype(np.float32))
