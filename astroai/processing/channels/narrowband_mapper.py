"""Narrowband palette mapper (SHO / HOO / NHO)."""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["NarrowbandMapper", "NarrowbandPalette"]


class NarrowbandPalette(Enum):
    SHO = "sho"   # Hubble: R=SII, G=Ha, B=OIII
    HOO = "hoo"   # R=Ha, G=OIII, B=OIII
    NHO = "nho"   # R=NII(≈Ha), G=Ha, B=OIII


class NarrowbandMapper:
    """Maps narrowband emission channels to an RGB composite."""

    def map(
        self,
        Ha: NDArray[np.floating[Any]] | None,
        OIII: NDArray[np.floating[Any]] | None,
        SII: NDArray[np.floating[Any]] | None,
        palette: NarrowbandPalette = NarrowbandPalette.SHO,
    ) -> NDArray[np.float32]:
        """Return an (H, W, 3) float32 image in [0, 1]."""
        provided = [c for c in (Ha, OIII, SII) if c is not None]
        if not provided:
            raise ValueError("At least one narrowband channel must be provided.")

        h, w = provided[0].shape[:2]

        def _prep(c: NDArray[np.floating[Any]] | None) -> NDArray[np.float32]:
            if c is None:
                return np.zeros((h, w), dtype=np.float32)
            arr = np.asarray(c, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            return np.clip(arr, 0.0, 1.0)

        ha = _prep(Ha)
        oiii = _prep(OIII)
        sii = _prep(SII)

        if palette is NarrowbandPalette.SHO:
            r, g, b = sii, ha, oiii
        elif palette is NarrowbandPalette.HOO:
            r, g, b = ha, oiii, oiii
        elif palette is NarrowbandPalette.NHO:
            r, g, b = ha, ha, oiii  # NII ≈ Ha when NII not provided
        else:
            raise ValueError(f"Unknown palette: {palette!r}")

        return np.stack([r, g, b], axis=-1).astype(np.float32)
