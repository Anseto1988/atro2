"""Plate solving engine with ASTAP primary and astrometry.net fallback."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.io import fits
from astropy.wcs import WCS

__all__ = ["PlateSolver", "SolveResult", "SolveError"]


class SolveError(Exception):
    pass


@dataclass(frozen=True)
class SolveResult:
    wcs: WCS
    ra_center: float
    dec_center: float
    field_width_deg: float
    field_height_deg: float
    rotation_deg: float
    solve_time_s: float
    solver_used: str


def _detect_astap_binary() -> Path:
    from astroai.engine.platesolving.astap_binary import get_astap_path
    return get_astap_path()


def _try_coord_float(value: object) -> float | None:
    """Safely parse a FITS coordinate value to float; returns None for sexagesimal strings."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _extract_coordinates_from_header(header: fits.Header) -> tuple[float, float] | None:
    ra = _try_coord_float(header.get("RA") or header.get("OBJCTRA") or header.get("CRVAL1"))
    dec = _try_coord_float(header.get("DEC") or header.get("OBJCTDEC") or header.get("CRVAL2"))
    if ra is not None and dec is not None:
        return ra, dec
    return None


class PlateSolver:
    def __init__(
        self,
        astap_path: Path | None = None,
        astrometry_api_key: str | None = None,
        max_retries: int = 3,
        search_radius_deg: float = 10.0,
        timeout_s: float = 120.0,
    ) -> None:
        self._astap_path = astap_path or _detect_astap_binary()
        self._astrometry_api_key = astrometry_api_key
        self._max_retries = max_retries
        self._search_radius_deg = search_radius_deg
        self._timeout_s = timeout_s

    @property
    def astap_path(self) -> Path:
        return self._astap_path

    def solve(
        self,
        fits_path: Path,
        ra_hint: float | None = None,
        dec_hint: float | None = None,
    ) -> SolveResult:
        if ra_hint is None or dec_hint is None:
            coords = self._extract_coords_from_fits(fits_path)
            if coords:
                ra_hint, dec_hint = coords

        try:
            return self._solve_astap(fits_path, ra_hint, dec_hint)
        except SolveError:
            if self._astrometry_api_key:
                return self._solve_astrometry_net(fits_path, ra_hint, dec_hint)
            raise

    def _extract_coords_from_fits(self, fits_path: Path) -> tuple[float, float] | None:
        with fits.open(str(fits_path)) as hdul:
            return _extract_coordinates_from_header(hdul[0].header)

    def _solve_astap(
        self,
        fits_path: Path,
        ra_hint: float | None,
        dec_hint: float | None,
    ) -> SolveResult:
        radius = self._search_radius_deg
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            start = time.monotonic()
            try:
                wcs = self._run_astap_subprocess(fits_path, ra_hint, dec_hint, radius)
                elapsed = time.monotonic() - start
                ra_c, dec_c = wcs.wcs.crval
                pixel_scale = abs(wcs.wcs.cdelt[0]) if wcs.wcs.cdelt[0] != 0 else 0.000277
                naxis1 = wcs.pixel_shape[0] if wcs.pixel_shape else 1000
                naxis2 = wcs.pixel_shape[1] if wcs.pixel_shape else 1000
                fw = naxis1 * pixel_scale
                fh = naxis2 * pixel_scale
                rot = 0.0
                if hasattr(wcs.wcs, "pc") and wcs.wcs.pc is not None:
                    import math
                    rot = math.degrees(math.atan2(wcs.wcs.pc[0, 1], wcs.wcs.pc[0, 0]))

                return SolveResult(
                    wcs=wcs,
                    ra_center=ra_c,
                    dec_center=dec_c,
                    field_width_deg=fw,
                    field_height_deg=fh,
                    rotation_deg=rot,
                    solve_time_s=elapsed,
                    solver_used="astap",
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, SolveError) as e:
                last_error = e
                radius *= 1.5

        raise SolveError(
            f"ASTAP solve failed after {self._max_retries} attempts: {last_error}"
        )

    def _run_astap_subprocess(
        self,
        fits_path: Path,
        ra_hint: float | None,
        dec_hint: float | None,
        radius: float,
    ) -> WCS:
        cmd = [str(self._astap_path), "-f", str(fits_path), "-r", str(radius)]
        if ra_hint is not None:
            cmd.extend(["-ra", str(ra_hint)])
        if dec_hint is not None:
            cmd.extend(["-spd", str(dec_hint + 90)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self._timeout_s,
        )
        if result.returncode != 0:
            raise SolveError(f"ASTAP failed: {result.stderr}")

        wcs_path = fits_path.with_suffix(".wcs")
        if not wcs_path.exists():
            raise SolveError("ASTAP did not produce a WCS file")

        with fits.open(str(wcs_path)) as hdul:
            wcs = WCS(hdul[0].header)
        return wcs

    def _solve_astrometry_net(
        self,
        fits_path: Path,
        ra_hint: float | None,
        dec_hint: float | None,
    ) -> SolveResult:
        import httpx

        start = time.monotonic()
        with open(fits_path, "rb") as f:
            files = {"file": (fits_path.name, f, "application/fits")}
            data: dict[str, Any] = {"apikey": self._astrometry_api_key}
            if ra_hint is not None:
                data["center_ra"] = ra_hint
            if dec_hint is not None:
                data["center_dec"] = dec_hint
            data["radius"] = self._search_radius_deg

            resp = httpx.post(
                "http://nova.astrometry.net/api/upload",
                files=files,
                data=data,
                timeout=self._timeout_s,
            )
            resp.raise_for_status()

        result_data = resp.json()
        if "status" in result_data and result_data["status"] == "error":
            raise SolveError(f"astrometry.net failed: {result_data}")

        header = fits.Header()
        for k, v in result_data.get("wcs_header", {}).items():
            header[k] = v
        wcs = WCS(header)
        elapsed = time.monotonic() - start

        ra_c, dec_c = wcs.wcs.crval
        return SolveResult(
            wcs=wcs,
            ra_center=ra_c,
            dec_center=dec_c,
            field_width_deg=result_data.get("field_width", 0.0),
            field_height_deg=result_data.get("field_height", 0.0),
            rotation_deg=result_data.get("orientation", 0.0),
            solve_time_s=elapsed,
            solver_used="astrometry.net",
        )
