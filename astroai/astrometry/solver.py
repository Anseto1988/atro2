"""ASTAP plate-solving adapter (delegates to engine.platesolving.PlateSolver)."""

from __future__ import annotations

import tempfile
import logging
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from astroai.astrometry.catalog import WcsSolution
from astroai.engine.platesolving.solver import PlateSolver, SolveError as _SolveError

__all__ = ["AstapSolver", "SolverError"]

logger = logging.getLogger(__name__)

# Re-export so callers need only import from this module.
SolverError = _SolveError


class AstapSolver:
    """Thin adapter over :class:`~astroai.engine.platesolving.PlateSolver`.

    Returns a :class:`~astroai.astrometry.catalog.WcsSolution` dataclass rather
    than the full astropy :class:`~astropy.wcs.WCS` object, which is sufficient
    for the pipeline step and UI overlay.

    Args:
        executable: Path to ASTAP binary; *None* → auto-detect via PATH / env.
        search_radius_deg: Initial ASTAP search radius.
        fov_deg: Field-of-view hint (0 = ASTAP auto-detect).
        timeout: Subprocess timeout in seconds.
    """

    def __init__(
        self,
        executable: str | Path | None = None,
        search_radius_deg: float = 30.0,
        fov_deg: float = 0.0,
        timeout: int = 120,
    ) -> None:
        import math
        self._solver = PlateSolver(
            astap_path=Path(executable) if executable else None,
            search_radius_deg=search_radius_deg,
            timeout_s=float(timeout),
        )
        self._fov = fov_deg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, fits_path: Path | str) -> WcsSolution:
        """Plate-solve a FITS file and return a :class:`WcsSolution`.

        Raises:
            SolverError: If ASTAP cannot find a solution.
            FileNotFoundError: If *fits_path* does not exist.
        """
        fits_path = Path(fits_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        result = self._solver.solve(fits_path)
        return self._result_to_solution(result, fits_path)

    def solve_array(
        self,
        image: NDArray[np.floating[Any]],
        fits_header: dict[str, Any] | None = None,
    ) -> WcsSolution:
        """Solve an in-memory numpy image (writes a temp FITS file)."""
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]

        hdr = fits.Header(fits_header or {})
        hdu = fits.PrimaryHDU(data=arr, header=hdr)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            tmp = Path(f.name)
        try:
            hdu.writeto(tmp, overwrite=True)
            return self.solve(tmp)
        finally:
            tmp.unlink(missing_ok=True)
            tmp.with_suffix(".wcs").unlink(missing_ok=True)
            tmp.with_suffix(".ini").unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_solution(result: Any, fits_path: Path) -> WcsSolution:
        """Convert a PlateSolver SolveResult to a WcsSolution dataclass."""
        import math

        wcs = result.wcs
        # Extract CD matrix from astropy WCS
        if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
            cd = wcs.wcs.cd
            cd1_1, cd1_2 = float(cd[0, 0]), float(cd[0, 1])
            cd2_1, cd2_2 = float(cd[1, 0]), float(cd[1, 1])
        elif hasattr(wcs.wcs, "pc") and wcs.wcs.pc is not None:
            pc = wcs.wcs.pc
            cdelt = wcs.wcs.cdelt
            cd1_1, cd1_2 = float(pc[0, 0] * cdelt[0]), float(pc[0, 1] * cdelt[0])
            cd2_1, cd2_2 = float(pc[1, 0] * cdelt[1]), float(pc[1, 1] * cdelt[1])
        else:
            cdelt = wcs.wcs.cdelt
            cd1_1, cd1_2 = float(cdelt[0]), 0.0
            cd2_1, cd2_2 = 0.0, float(cdelt[1])

        pixel_scale_deg = (abs(cd1_1 * cd2_2 - cd1_2 * cd2_1)) ** 0.5
        pixel_scale_arcsec = pixel_scale_deg * 3600.0
        rotation = math.degrees(math.atan2(cd1_2, cd1_1))

        crpix = wcs.wcs.crpix
        crpix1 = float(crpix[0]) if len(crpix) > 0 else 1.0
        crpix2 = float(crpix[1]) if len(crpix) > 1 else 1.0

        return WcsSolution(
            ra_center=result.ra_center,
            dec_center=result.dec_center,
            pixel_scale_arcsec=pixel_scale_arcsec,
            rotation_deg=rotation,
            fov_width_deg=result.field_width_deg,
            fov_height_deg=result.field_height_deg,
            cd_matrix=(cd1_1, cd1_2, cd2_1, cd2_2),
            crpix1=crpix1,
            crpix2=crpix2,
        )

    # Keep backward-compat: expose _parse_wcs for unit tests that mock it.
    @staticmethod
    def _parse_wcs(wcs_path: Path) -> WcsSolution:
        """Read an ASTAP .wcs sidecar and return a WcsSolution (used by tests)."""
        import math
        from astropy.io import fits as _fits

        with _fits.open(wcs_path) as hdul:
            hdr = hdul[0].header  # type: ignore[index]

        def _get(key: str, default: float = 0.0) -> float:
            val = hdr.get(key, default)
            return float(val) if val is not None else default

        ra = _get("CRVAL1")
        dec = _get("CRVAL2")
        crpix1 = _get("CRPIX1", 1.0)
        crpix2 = _get("CRPIX2", 1.0)
        cd1_1 = _get("CD1_1")
        cd1_2 = _get("CD1_2")
        cd2_1 = _get("CD2_1")
        cd2_2 = _get("CD2_2")

        pixel_scale_deg = (abs(cd1_1 * cd2_2 - cd1_2 * cd2_1)) ** 0.5
        pixel_scale_arcsec = pixel_scale_deg * 3600.0
        rotation = math.degrees(math.atan2(cd1_2, cd1_1))
        naxis1 = float(hdr.get("NAXIS1", 0))
        naxis2 = float(hdr.get("NAXIS2", 0))

        return WcsSolution(
            ra_center=ra,
            dec_center=dec,
            pixel_scale_arcsec=pixel_scale_arcsec,
            rotation_deg=rotation,
            fov_width_deg=naxis1 * pixel_scale_deg,
            fov_height_deg=naxis2 * pixel_scale_deg,
            cd_matrix=(cd1_1, cd1_2, cd2_1, cd2_2),
            crpix1=crpix1,
            crpix2=crpix2,
        )
