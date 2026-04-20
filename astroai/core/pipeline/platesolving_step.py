"""Pipeline step for plate solving using the engine's PlateSolver."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    _noop_callback,
)
from astroai.engine.platesolving.solver import PlateSolver, SolveError, SolveResult
from astroai.engine.platesolving.wcs_writer import WCSWriter

__all__ = ["PlateSolvingStep"]

logger = logging.getLogger(__name__)

_METADATA_KEY_SOLVE_RESULT = "solve_result"
_METADATA_KEY_WCS = "wcs"


class PlateSolvingStep(PipelineStep):
    """Plate-solve the stacked result and write WCS headers into the output FITS.

    Stores the :class:`~astroai.engine.platesolving.solver.SolveResult` under
    ``context.metadata["solve_result"]`` and the astropy WCS under
    ``context.metadata["wcs"]``.

    Args:
        astap_path: Explicit path to ASTAP binary; None = auto-detect.
        astrometry_api_key: Optional astrometry.net API key for fallback.
        search_radius_deg: Initial search radius for ASTAP.
        max_retries: Number of retry attempts with expanded radius.
        timeout_s: Subprocess timeout in seconds.
        write_wcs_to_fits: If True and an output FITS path is in metadata, write WCS.
        fail_silently: Log solver failures instead of aborting the pipeline.
    """

    def __init__(
        self,
        astap_path: str | Path | None = None,
        astrometry_api_key: str | None = None,
        search_radius_deg: float = 10.0,
        max_retries: int = 3,
        timeout_s: float = 120.0,
        write_wcs_to_fits: bool = True,
        fail_silently: bool = True,
    ) -> None:
        self._solver = PlateSolver(
            astap_path=Path(astap_path) if astap_path else None,
            astrometry_api_key=astrometry_api_key,
            search_radius_deg=search_radius_deg,
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        self._wcs_writer = WCSWriter()
        self._write_wcs = write_wcs_to_fits
        self._fail_silently = fail_silently

    @property
    def name(self) -> str:
        return "Plate Solving"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.ASTROMETRY

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = _noop_callback,
    ) -> PipelineContext:
        progress(PipelineProgress(
            stage=self.stage,
            current=0,
            total=2,
            message="Plate solving: preparing image…",
        ))

        image = context.result
        if image is None and context.images:
            image = context.images[0]

        if image is None:
            logger.warning("PlateSolvingStep: no image in context, skipping")
            return context

        try:
            result = self._solve_image(image, context.metadata)
            context.metadata[_METADATA_KEY_SOLVE_RESULT] = result
            context.metadata[_METADATA_KEY_WCS] = result.wcs

            logger.info(
                "Plate solve OK: RA=%.4f Dec=%.4f (%.2fs, %s)",
                result.ra_center,
                result.dec_center,
                result.solve_time_s,
                result.solver_used,
            )

            progress(PipelineProgress(
                stage=self.stage,
                current=1,
                total=2,
                message="Writing WCS headers…",
            ))

            if self._write_wcs:
                self._write_wcs_to_output(result, context.metadata)

        except SolveError as exc:
            if self._fail_silently:
                logger.warning("Plate solve failed (skipped): %s", exc)
            else:
                raise

        progress(PipelineProgress(
            stage=self.stage,
            current=2,
            total=2,
            message="Plate solving complete",
        ))
        return context

    def _solve_image(
        self,
        image: NDArray[np.floating[Any]],
        metadata: dict[str, Any],
    ) -> SolveResult:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]

        header = fits.Header()
        if metadata.get("pixel_size_um") and metadata.get("focal_length_mm"):
            scale = (
                206.265
                * float(metadata["pixel_size_um"])
                / float(metadata["focal_length_mm"])
            )
            header["SCALE"] = scale

        ra_hint = metadata.get("ra_hint")
        dec_hint = metadata.get("dec_hint")

        hdu = fits.PrimaryHDU(data=arr, header=header)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            hdu.writeto(tmp_path, overwrite=True)
            return self._solver.solve(
                tmp_path,
                ra_hint=float(ra_hint) if ra_hint is not None else None,
                dec_hint=float(dec_hint) if dec_hint is not None else None,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
            tmp_path.with_suffix(".wcs").unlink(missing_ok=True)
            tmp_path.with_suffix(".ini").unlink(missing_ok=True)

    def _write_wcs_to_output(
        self, result: SolveResult, metadata: dict[str, Any]
    ) -> None:
        output_path = metadata.get("export_path") or metadata.get("output_fits_path")
        if output_path is None:
            return
        path = Path(output_path)
        if path.exists() and path.suffix.lower() in (".fits", ".fit"):
            self._wcs_writer.write_wcs_to_fits(path, result.wcs)
            logger.info("WCS written to %s", path)
