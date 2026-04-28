"""Pipeline step for astrometric plate-solving (ASTAP)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from astroai.astrometry.catalog import WcsSolution
from astroai.astrometry.solver import AstapSolver, SolverError
from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)

__all__ = ["AstrometryStep"]

logger = logging.getLogger(__name__)

_METADATA_KEY = "wcs_solution"


class AstrometryStep(PipelineStep):
    """Plate-solve the stacked result and store the WCS in the pipeline context.

    The WCS solution is stored under ``context.metadata["wcs_solution"]`` as a
    :class:`~astroai.astrometry.catalog.WcsSolution` instance.  If the solver
    fails and *fail_silently* is True the step is skipped without raising.

    Args:
        executable: Path to the ASTAP binary.  *None* searches PATH.
        search_radius_deg: ASTAP search radius.
        fov_deg: Field-of-view hint (0 = auto).
        fail_silently: If True, solver failures are logged but do not abort
                       the pipeline.
    """

    def __init__(
        self,
        executable: str | Path | None = None,
        search_radius_deg: float = 30.0,
        fov_deg: float = 0.0,
        fail_silently: bool = True,
    ) -> None:
        self._solver = AstapSolver(
            executable=executable,
            search_radius_deg=search_radius_deg,
            fov_deg=fov_deg,
        )
        self._fail_silently = fail_silently

    @property
    def name(self) -> str:
        return "Plate Solving (ASTAP)"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.ASTROMETRY

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        progress(PipelineProgress(
            stage=self.stage,
            current=0,
            total=1,
            message="Running plate solver…",
        ))

        image = context.result
        if image is None and context.images:
            image = context.images[0]

        if image is None:
            logger.warning("AstrometryStep: no image in context, skipping")
            return context

        fits_header: dict[str, Any] = {}
        if context.metadata.get("pixel_size_um") and context.metadata.get("focal_length_mm"):
            scale = (
                206.265
                * float(context.metadata["pixel_size_um"])
                / float(context.metadata["focal_length_mm"])
            )
            fits_header["SCALE"] = scale

        try:
            solution: WcsSolution = self._solver.solve_array(
                image, fits_header=fits_header or None
            )
            context.metadata[_METADATA_KEY] = solution
            logger.info(
                "Plate solve: RA=%.4f Dec=%.4f scale=%.3f\"/px rot=%.2f°",
                solution.ra_center,
                solution.dec_center,
                solution.pixel_scale_arcsec,
                solution.rotation_deg,
            )
        except SolverError as exc:
            if self._fail_silently:
                logger.warning("Plate solve failed (skipped): %s", exc)
            else:
                raise

        progress(PipelineProgress(
            stage=self.stage,
            current=1,
            total=1,
            message="Plate solving complete",
        ))
        return context
