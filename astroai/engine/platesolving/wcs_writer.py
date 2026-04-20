"""Write WCS (World Coordinate System) headers into FITS files."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS


__all__ = ["WCSWriter"]


class WCSWriter:
    def write_wcs_to_fits(self, fits_path: Path, wcs: WCS) -> Path:
        with fits.open(str(fits_path), mode="update") as hdul:
            header = hdul[0].header
            wcs_header = wcs.to_header()
            for key, value in wcs_header.items():
                header[key] = value
            hdul.flush()
        return fits_path

    def read_wcs_from_fits(self, fits_path: Path) -> WCS | None:
        with fits.open(str(fits_path)) as hdul:
            header = hdul[0].header
            if "CTYPE1" not in header:
                return None
            return WCS(header)

    def create_wcs(
        self,
        crval: tuple[float, float],
        crpix: tuple[float, float],
        cdelt: tuple[float, float],
        naxis: tuple[int, int],
        ctype: tuple[str, str] = ("RA---TAN", "DEC--TAN"),
        rotation_deg: float = 0.0,
    ) -> WCS:
        import math

        wcs = WCS(naxis=2)
        wcs.wcs.crval = list(crval)
        wcs.wcs.crpix = list(crpix)
        wcs.wcs.ctype = list(ctype)

        cos_r = math.cos(math.radians(rotation_deg))
        sin_r = math.sin(math.radians(rotation_deg))
        wcs.wcs.cd = [
            [cdelt[0] * cos_r, -cdelt[1] * sin_r],
            [cdelt[0] * sin_r, cdelt[1] * cos_r],
        ]
        wcs.pixel_shape = naxis
        return wcs
