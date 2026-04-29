"""CLAHE — Contrast Limited Adaptive Histogram Equalization (pure numpy)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["CLAHEConfig", "CLAHEEnhancer", "CLAHEStep"]

logger = logging.getLogger(__name__)

_VALID_CHANNEL_MODES = ("luminance", "each", "grayscale")


@dataclass(frozen=True)
class CLAHEConfig:
    """Configuration for CLAHE local contrast enhancement."""

    clip_limit: float = 2.0
    """Histogram clip limit [1.0, 10.0]. Higher = less clipping = stronger effect."""

    tile_size: int = 64
    """Tile size in pixels [8, 512]. Each tile is processed independently."""

    n_bins: int = 256
    """Number of histogram bins [64, 1024]."""

    channel_mode: str = "luminance"
    """Processing mode: 'luminance', 'each', or 'grayscale'."""

    def __post_init__(self) -> None:
        if self.clip_limit < 1.0:
            raise ValueError(
                f"clip_limit must be >= 1.0, got {self.clip_limit!r}"
            )
        if self.tile_size <= 0:
            raise ValueError(
                f"tile_size must be > 0, got {self.tile_size!r}"
            )
        if self.n_bins < 64 or self.n_bins > 1024:
            raise ValueError(
                f"n_bins must be in [64, 1024], got {self.n_bins!r}"
            )
        if self.channel_mode not in _VALID_CHANNEL_MODES:
            raise ValueError(
                f"channel_mode must be one of {_VALID_CHANNEL_MODES}, "
                f"got {self.channel_mode!r}"
            )

    def is_identity(self) -> bool:
        """Always False — no simple identity check for CLAHE."""
        return False

    def as_dict(self) -> dict:
        return {
            "clip_limit": self.clip_limit,
            "tile_size": self.tile_size,
            "n_bins": self.n_bins,
            "channel_mode": self.channel_mode,
        }


class CLAHEEnhancer:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""

    def __init__(self, config: CLAHEConfig | None = None) -> None:
        self.config = config or CLAHEConfig()

    def enhance(self, image: NDArray) -> NDArray:
        """Return a contrast-enhanced copy of *image*.

        Supports float arrays (H×W or H×W×3) with values in [0, 1].
        """
        arr = np.asarray(image, dtype=np.float64)
        original_dtype = image.dtype

        if arr.ndim == 2:
            # Grayscale
            out = self._clahe_channel(arr)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mode = self.config.channel_mode
            if mode == "each":
                out = np.stack([
                    self._clahe_channel(arr[:, :, c])
                    for c in range(3)
                ], axis=2)
            elif mode == "grayscale":
                # treat as grayscale using luminance weights, return RGB
                lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
                lum_enhanced = self._clahe_channel(lum)
                # scale each channel by correction factor
                with np.errstate(divide="ignore", invalid="ignore"):
                    factor = np.where(lum > 1e-8, lum_enhanced / lum, 1.0)
                out = np.clip(arr * factor[:, :, np.newaxis], 0.0, 1.0)
            else:
                # luminance mode (default): apply CLAHE to luminance, scale RGB
                lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
                lum_enhanced = self._clahe_channel(lum)
                with np.errstate(divide="ignore", invalid="ignore"):
                    factor = np.where(lum > 1e-8, lum_enhanced / lum, 1.0)
                out = np.clip(arr * factor[:, :, np.newaxis], 0.0, 1.0)
        else:
            logger.warning(
                "CLAHEEnhancer: unexpected image shape %s, returning unchanged", arr.shape
            )
            return image

        out = np.clip(out, 0.0, 1.0)
        return out.astype(original_dtype)

    def _clahe_channel(self, channel: np.ndarray) -> np.ndarray:
        """Apply CLAHE to a single float [0,1] channel.

        Returns a float array of same shape with enhanced contrast.
        """
        H, W = channel.shape
        n_bins = self.config.n_bins
        tile_size = self.config.tile_size

        # Number of tiles in each direction (at least 1)
        tile_h = max(1, H // tile_size)
        tile_w = max(1, W // tile_size)

        # Quantize to bins
        img_int = (channel * (n_bins - 1)).astype(np.int32)
        img_int = np.clip(img_int, 0, n_bins - 1)

        # Build per-tile lookup tables (LUT)
        # luts[ty, tx] = array of shape (n_bins,) mapping bin → [0, 1]
        luts = np.zeros((tile_h, tile_w, n_bins), dtype=np.float64)

        for ty in range(tile_h):
            for tx in range(tile_w):
                # Tile boundaries
                r0 = ty * tile_size
                r1 = min(r0 + tile_size, H)
                c0 = tx * tile_size
                c1 = min(c0 + tile_size, W)

                tile_pixels = img_int[r0:r1, c0:c1]
                n_pixels = tile_pixels.size

                # Compute histogram
                hist, _ = np.histogram(tile_pixels, bins=n_bins, range=(0, n_bins - 1))

                # Clip histogram at clip_limit (as absolute pixel count)
                # clip_limit is relative to average bin occupancy × clip_limit_factor
                avg_bin_count = n_pixels / n_bins
                abs_clip = max(1, int(round(avg_bin_count * self.config.clip_limit)))

                # Clip excess and redistribute
                excess = np.sum(np.maximum(hist - abs_clip, 0))
                hist = np.minimum(hist, abs_clip)
                # Redistribute excess uniformly
                redistribute_per_bin = excess // n_bins
                remainder = excess - redistribute_per_bin * n_bins
                hist = hist + redistribute_per_bin
                # Add remainder to lower bins
                hist[:remainder] += 1

                # Build cumulative distribution function
                cdf = np.cumsum(hist)
                # Normalize to [0, 1]
                cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0
                if n_pixels - cdf_min > 0:
                    lut = (cdf - cdf_min) / (n_pixels - cdf_min)
                else:
                    lut = np.linspace(0.0, 1.0, n_bins)
                luts[ty, tx] = np.clip(lut, 0.0, 1.0)

        # Bilinear interpolation between tile LUTs
        # For each pixel, find the four surrounding tile centers and interpolate
        result = np.zeros_like(channel, dtype=np.float64)

        # Compute tile centers in pixel coordinates
        # Center of tile (ty, tx) is at row: ty*tile_size + tile_size/2 - 0.5
        def tile_center_row(ty: int) -> float:
            r0 = ty * tile_size
            r1 = min(r0 + tile_size, H)
            return (r0 + r1 - 1) / 2.0

        def tile_center_col(tx: int) -> float:
            c0 = tx * tile_size
            c1 = min(c0 + tile_size, W)
            return (c0 + c1 - 1) / 2.0

        tile_centers_r = np.array([tile_center_row(ty) for ty in range(tile_h)], dtype=np.float64)
        tile_centers_c = np.array([tile_center_col(tx) for tx in range(tile_w)], dtype=np.float64)

        # Create coordinate grids
        rows = np.arange(H, dtype=np.float64)
        cols = np.arange(W, dtype=np.float64)

        # For each row, find the two bounding tile rows
        # For each col, find the two bounding tile cols
        # Then bilinearly interpolate

        # Get the pixel values as bin indices for LUT lookup
        pix_bins = img_int  # shape (H, W)

        if tile_h == 1 and tile_w == 1:
            # Only one tile: just apply that LUT
            result = luts[0, 0][pix_bins]
        elif tile_h == 1:
            # Only one row of tiles: linear interpolation in column direction
            for col in range(W):
                c = float(col)
                tx0, tx1, tc = _find_surrounding(c, tile_centers_c, tile_w)
                lut_vals_0 = luts[0, tx0][pix_bins[:, col]]
                lut_vals_1 = luts[0, tx1][pix_bins[:, col]]
                result[:, col] = lut_vals_0 * (1.0 - tc) + lut_vals_1 * tc
        elif tile_w == 1:
            # Only one column of tiles: linear interpolation in row direction
            for row in range(H):
                r = float(row)
                ty0, ty1, tr = _find_surrounding(r, tile_centers_r, tile_h)
                lut_vals_0 = luts[ty0, 0][pix_bins[row, :]]
                lut_vals_1 = luts[ty1, 0][pix_bins[row, :]]
                result[row, :] = lut_vals_0 * (1.0 - tr) + lut_vals_1 * tr
        else:
            # Full bilinear interpolation
            for row in range(H):
                r = float(row)
                ty0, ty1, tr = _find_surrounding(r, tile_centers_r, tile_h)
                for col in range(W):
                    c = float(col)
                    tx0, tx1, tc = _find_surrounding(c, tile_centers_c, tile_w)
                    b = int(pix_bins[row, col])
                    v00 = luts[ty0, tx0][b]
                    v01 = luts[ty0, tx1][b]
                    v10 = luts[ty1, tx0][b]
                    v11 = luts[ty1, tx1][b]
                    result[row, col] = (
                        v00 * (1 - tr) * (1 - tc)
                        + v01 * (1 - tr) * tc
                        + v10 * tr * (1 - tc)
                        + v11 * tr * tc
                    )

        return result


def _find_surrounding(
    pos: float,
    centers: np.ndarray,
    n: int,
) -> tuple[int, int, float]:
    """Find the two surrounding tile indices and interpolation weight.

    Returns (idx0, idx1, t) where t is the weight for idx1 [0, 1].
    """
    if n == 1:
        return 0, 0, 0.0

    # Find where pos sits relative to centers
    idx = np.searchsorted(centers, pos) - 1
    idx0 = int(np.clip(idx, 0, n - 2))
    idx1 = idx0 + 1

    span = centers[idx1] - centers[idx0]
    if span <= 0:
        t = 0.0
    else:
        t = float(np.clip((pos - centers[idx0]) / span, 0.0, 1.0))

    return idx0, idx1, t


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


class CLAHEStep(PipelineStep):
    """Apply CLAHE local contrast enhancement as a pipeline step."""

    def __init__(self, config: CLAHEConfig | None = None) -> None:
        self._enhancer = CLAHEEnhancer(config)

    @property
    def name(self) -> str:
        return "Lokale Kontrastverbesserung"

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
            logger.warning("CLAHEStep: no image in context, skipping")
            return context

        progress(PipelineProgress(
            stage=self.stage, current=0, total=1,
            message="Lokale Kontrastverbesserung anpassen…",
        ))
        context.result = self._enhancer.enhance(image)
        progress(PipelineProgress(
            stage=self.stage, current=1, total=1,
            message="Lokale Kontrastverbesserung abgeschlossen",
        ))
        return context
