"""Drizzle pipeline step for integration into the processing pipeline."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroai.astrometry.catalog import WcsSolution
from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.engine.drizzle.engine import DrizzleEngine

__all__ = ["DrizzleStep"]

logger = logging.getLogger(__name__)

_WCS_METADATA_KEYS = ("wcs_solutions", "wcs_solution", "wcs")


class DrizzleStep(PipelineStep):
    """Pipeline step that applies drizzle integration to input frames."""

    def __init__(
        self,
        drop_size: float = 0.7,
        pixfrac: float = 1.0,
        scale: float = 1.0,
    ) -> None:
        self._engine = DrizzleEngine(
            drop_size=drop_size, pixfrac=pixfrac, scale=scale
        )

    @property
    def name(self) -> str:
        return "Drizzle"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.DRIZZLE

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        if not context.images:
            logger.warning("DrizzleStep: no images in context, skipping")
            return context

        progress(PipelineProgress(
            stage=PipelineStage.DRIZZLE,
            current=0,
            total=len(context.images),
            message="Preparing drizzle integration",
        ))

        wcs_solutions = self._resolve_wcs(context)
        output_shape = self._compute_output_shape(context.images[0], self._engine.scale)

        progress(PipelineProgress(
            stage=PipelineStage.DRIZZLE,
            current=1,
            total=len(context.images),
            message=f"Drizzling {len(context.images)} frames",
        ))

        context.result = self._engine.drizzle(
            context.images, wcs_solutions, output_shape
        )

        progress(PipelineProgress(
            stage=PipelineStage.DRIZZLE,
            current=len(context.images),
            total=len(context.images),
            message="Drizzle complete",
        ))

        return context

    def _resolve_wcs(self, context: PipelineContext) -> list[WcsSolution]:
        """Extract WCS solutions from context metadata or generate identity fallback."""
        for key in _WCS_METADATA_KEYS:
            value = context.metadata.get(key)
            if value is None:
                continue
            if isinstance(value, list) and len(value) == len(context.images):
                return value
            if isinstance(value, WcsSolution):
                return [value] * len(context.images)

        logger.info("No WCS found in metadata, using identity transform fallback")
        return self._identity_wcs_list(context.images)

    def _identity_wcs_list(
        self, images: list[NDArray[np.floating[Any]]]
    ) -> list[WcsSolution]:
        """Generate identity WCS solutions (no reprojection)."""
        h, w = images[0].shape[0], images[0].shape[1]
        identity = WcsSolution(
            ra_center=0.0,
            dec_center=0.0,
            pixel_scale_arcsec=1.0,
            rotation_deg=0.0,
            fov_width_deg=w / 3600.0,
            fov_height_deg=h / 3600.0,
            cd_matrix=(1.0 / 3600.0, 0.0, 0.0, 1.0 / 3600.0),
            crpix1=w / 2.0,
            crpix2=h / 2.0,
        )
        return [identity] * len(images)

    @staticmethod
    def _compute_output_shape(
        reference_frame: NDArray[np.floating[Any]], scale: float
    ) -> tuple[int, int]:
        """Compute output shape based on reference frame and scale factor."""
        h, w = reference_frame.shape[0], reference_frame.shape[1]
        return (int(h * scale), int(w * scale))
