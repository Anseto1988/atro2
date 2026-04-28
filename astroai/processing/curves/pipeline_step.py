"""Tone curve adjustment pipeline step using CubicSpline interpolation."""
from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)

__all__ = ["CurvesStep"]

logger = logging.getLogger(__name__)

_IDENTITY: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
_LUT_SIZE = 65536


def _is_identity(points: list[tuple[float, float]]) -> bool:
    """Return True if points represent the identity (diagonal) curve."""
    if len(points) == 2:
        return points[0] == (0.0, 0.0) and points[1] == (1.0, 1.0)
    return False


def _build_lut(points: list[tuple[float, float]]) -> NDArray[np.float64]:
    """Build a 65536-entry LUT from control points via CubicSpline."""
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    # Deduplicate x values to avoid CubicSpline errors
    _, unique_idx = np.unique(xs, return_index=True)
    xs, ys = xs[unique_idx], ys[unique_idx]
    if len(xs) < 2:
        return np.arange(_LUT_SIZE, dtype=np.float64) / (_LUT_SIZE - 1)
    cs = CubicSpline(xs, ys, bc_type="not-a-knot")
    t = np.linspace(0.0, 1.0, _LUT_SIZE)
    return cast(NDArray[np.float64], np.clip(cs(t), 0.0, 1.0))


def _apply_lut(
    channel: NDArray[np.floating[Any]],
    lut: NDArray[np.float64],
) -> NDArray[np.floating[Any]]:
    """Apply a precomputed LUT to a single-channel array."""
    indices = np.clip(
        (channel * (_LUT_SIZE - 1)).astype(np.int32), 0, _LUT_SIZE - 1
    )
    return lut[indices].astype(channel.dtype)


class CurvesStep(PipelineStep):
    """Apply tone curves to the stacked result image."""

    def __init__(
        self,
        rgb_points: list[tuple[float, float]] | None = None,
        r_points: list[tuple[float, float]] | None = None,
        g_points: list[tuple[float, float]] | None = None,
        b_points: list[tuple[float, float]] | None = None,
    ) -> None:
        self._rgb_points = rgb_points or list(_IDENTITY)
        self._r_points = r_points or list(_IDENTITY)
        self._g_points = g_points or list(_IDENTITY)
        self._b_points = b_points or list(_IDENTITY)

    @property
    def name(self) -> str:
        return "Kurven"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def _apply_curves(
        self, image: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Apply all configured curves to a single image array."""
        # Apply RGB (all-channel) curve
        if not _is_identity(self._rgb_points):
            lut = _build_lut(self._rgb_points)
            if image.ndim == 2:
                image = _apply_lut(image, lut)
            else:
                for c in range(image.shape[2]):
                    image[..., c] = _apply_lut(image[..., c], lut)

        # Apply per-channel curves for 3-channel images
        if image.ndim == 3 and image.shape[2] >= 3:
            for ch_idx, pts in enumerate(
                [self._r_points, self._g_points, self._b_points]
            ):
                if not _is_identity(pts):
                    lut = _build_lut(pts)
                    image[..., ch_idx] = _apply_lut(image[..., ch_idx], lut)

        return image

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        if context.result is None:
            return context
        progress(PipelineProgress(
            stage=self.stage, current=0, total=1, message="Kurven anwenden…",
        ))
        logger.debug("CurvesStep: applying tone curves")
        context.result = self._apply_curves(context.result)
        progress(PipelineProgress(
            stage=self.stage, current=1, total=1, message="Kurven abgeschlossen",
        ))
        return context
