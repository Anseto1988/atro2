"""Drizzle integration engine for sub-pixel super-resolution stacking."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroai.astrometry.catalog import WcsSolution

__all__ = ["DrizzleEngine"]

logger = logging.getLogger(__name__)

_VALID_DROP_SIZES = (0.5, 0.7, 1.0)


class DrizzleEngine:
    """WCS-based drizzle algorithm for sub-pixel image combination.

    Implements a simplified HST/DrizzlePac-style variable-pixel linear
    reconstruction. Each input pixel is projected onto an output grid
    using WCS transformations, with its footprint shrunk by ``pixfrac``.
    """

    def __init__(
        self,
        drop_size: float = 0.7,
        pixfrac: float = 1.0,
        scale: float = 1.0,
    ) -> None:
        if drop_size not in _VALID_DROP_SIZES:
            raise ValueError(
                f"drop_size must be one of {_VALID_DROP_SIZES}, got {drop_size}"
            )
        if pixfrac <= 0.0 or pixfrac > 1.0:
            raise ValueError(f"pixfrac must be in (0, 1], got {pixfrac}")
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")

        self._drop_size = drop_size
        self._pixfrac = pixfrac
        self._scale = scale

    @property
    def drop_size(self) -> float:
        return self._drop_size

    @property
    def pixfrac(self) -> float:
        return self._pixfrac

    @property
    def scale(self) -> float:
        return self._scale

    def drizzle(
        self,
        frames: list[NDArray[np.floating[Any]]],
        wcs_solutions: list[WcsSolution],
        output_shape: tuple[int, int],
    ) -> NDArray[np.float32]:
        """Drizzle-combine frames onto a common output grid.

        Args:
            frames: Input frames (H, W) or (H, W, C).
            wcs_solutions: Per-frame WCS solutions for coordinate mapping.
            output_shape: (height, width) of the output grid.

        Returns:
            Drizzled output image as float32.
        """
        if not frames:
            raise ValueError("No frames provided")
        if len(frames) != len(wcs_solutions):
            raise ValueError(
                f"Frame count ({len(frames)}) != WCS count ({len(wcs_solutions)})"
            )

        out_h, out_w = output_shape
        is_color = frames[0].ndim == 3
        n_channels = frames[0].shape[2] if is_color else 1

        output = np.zeros((out_h, out_w, n_channels), dtype=np.float64)
        weight_map = np.zeros((out_h, out_w, n_channels), dtype=np.float64)

        ref_wcs = wcs_solutions[0]

        for frame_idx, (frame, wcs) in enumerate(zip(frames, wcs_solutions)):
            logger.debug("Drizzling frame %d/%d", frame_idx + 1, len(frames))
            self._drizzle_single(
                frame, wcs, ref_wcs, output, weight_map, out_h, out_w, is_color
            )

        mask = weight_map > 0
        output[mask] /= weight_map[mask]

        result = output.astype(np.float32)
        if not is_color:
            result = result[:, :, 0]
        return result

    def _drizzle_single(
        self,
        frame: NDArray[np.floating[Any]],
        wcs_in: WcsSolution,
        wcs_ref: WcsSolution,
        output: NDArray[np.float64],
        weight_map: NDArray[np.float64],
        out_h: int,
        out_w: int,
        is_color: bool,
    ) -> None:
        """Project a single frame onto the output grid."""
        in_h, in_w = frame.shape[0], frame.shape[1]

        transform = self._compute_affine(wcs_in, wcs_ref)

        effective_drop = self._drop_size * self._pixfrac
        half_drop = effective_drop / 2.0

        y_in, x_in = np.mgrid[0:in_h, 0:in_w]
        y_in_flat = y_in.ravel().astype(np.float64)
        x_in_flat = x_in.ravel().astype(np.float64)

        x_out = transform[0, 0] * x_in_flat + transform[0, 1] * y_in_flat + transform[0, 2]
        y_out = transform[1, 0] * x_in_flat + transform[1, 1] * y_in_flat + transform[1, 2]

        x_out /= self._scale
        y_out /= self._scale

        x_min = np.floor(x_out - half_drop).astype(np.intp)
        x_max = np.floor(x_out + half_drop).astype(np.intp)
        y_min = np.floor(y_out - half_drop).astype(np.intp)
        y_max = np.floor(y_out + half_drop).astype(np.intp)

        valid = (x_min >= 0) & (y_min >= 0) & (x_max < out_w) & (y_max < out_h)
        indices = np.where(valid)[0]

        frame_flat: np.ndarray = (
            frame.reshape(-1, frame.shape[2]) if is_color else frame.ravel()
        )

        for idx in indices:
            ox_min, ox_max = x_min[idx], x_max[idx]
            oy_min, oy_max = y_min[idx], y_max[idx]

            cx, cy = x_out[idx], y_out[idx]

            for oy in range(oy_min, oy_max + 1):
                for ox in range(ox_min, ox_max + 1):
                    overlap = self._pixel_overlap(
                        cx, cy, half_drop, float(ox), float(oy)
                    )
                    if overlap <= 0.0:
                        continue
                    if is_color:
                        output[oy, ox, :] += frame_flat[idx] * overlap
                        weight_map[oy, ox, :] += overlap
                    else:
                        output[oy, ox, 0] += frame_flat[idx] * overlap
                        weight_map[oy, ox, 0] += overlap

    def _compute_affine(
        self, wcs_in: WcsSolution, wcs_ref: WcsSolution
    ) -> NDArray[np.float64]:
        """Compute affine transformation from input frame to reference frame coords."""
        cd_in = np.array(
            [[wcs_in.cd_matrix[0], wcs_in.cd_matrix[1]],
             [wcs_in.cd_matrix[2], wcs_in.cd_matrix[3]]],
            dtype=np.float64,
        )
        cd_ref = np.array(
            [[wcs_ref.cd_matrix[0], wcs_ref.cd_matrix[1]],
             [wcs_ref.cd_matrix[2], wcs_ref.cd_matrix[3]]],
            dtype=np.float64,
        )

        cd_ref_inv = np.linalg.inv(cd_ref)
        rotation = cd_ref_inv @ cd_in

        crpix_in = np.array([wcs_in.crpix1 - 1.0, wcs_in.crpix2 - 1.0])
        crpix_ref = np.array([wcs_ref.crpix1 - 1.0, wcs_ref.crpix2 - 1.0])

        translation = crpix_ref - rotation @ crpix_in

        affine = np.zeros((2, 3), dtype=np.float64)
        affine[:, :2] = rotation
        affine[:, 2] = translation
        return affine

    @staticmethod
    def _pixel_overlap(
        cx: float, cy: float, half_drop: float, px: float, py: float
    ) -> float:
        """Compute overlap area between a drop and an output pixel."""
        drop_x0 = cx - half_drop
        drop_x1 = cx + half_drop
        drop_y0 = cy - half_drop
        drop_y1 = cy + half_drop

        pix_x0 = px
        pix_x1 = px + 1.0
        pix_y0 = py
        pix_y1 = py + 1.0

        ox = max(0.0, min(drop_x1, pix_x1) - max(drop_x0, pix_x0))
        oy = max(0.0, min(drop_y1, pix_y1) - max(drop_y0, pix_y0))
        return ox * oy
