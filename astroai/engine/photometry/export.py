from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from astropy.io import fits

from astroai.engine.photometry.models import PhotometryResult

_COLUMNS = ["star_id", "ra", "dec", "instr_mag", "cal_mag", "catalog_mag", "residual"]


class PhotometryExporter:
    def to_csv(self, result: PhotometryResult, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_COLUMNS)
            for s in result.stars:
                writer.writerow([
                    s.star_id, s.ra, s.dec,
                    s.instr_mag, s.cal_mag, s.catalog_mag, s.residual,
                ])
        return path

    def to_fits(self, result: PhotometryResult, path: str | Path) -> Path:
        path = Path(path)
        n = len(result.stars)

        col_defs = [
            fits.Column(name="star_id", format="J", array=np.array([s.star_id for s in result.stars], dtype=np.int32)),
            fits.Column(name="ra", format="D", array=np.array([s.ra for s in result.stars])),
            fits.Column(name="dec", format="D", array=np.array([s.dec for s in result.stars])),
            fits.Column(name="instr_mag", format="D", array=np.array([s.instr_mag for s in result.stars])),
            fits.Column(name="cal_mag", format="D", array=np.array([s.cal_mag for s in result.stars])),
            fits.Column(name="catalog_mag", format="D", array=np.array([s.catalog_mag for s in result.stars])),
            fits.Column(name="residual", format="D", array=np.array([s.residual for s in result.stars])),
        ]

        hdu = fits.BinTableHDU.from_columns(col_defs)
        hdu.header["EXTNAME"] = "PHOTOMETRY"
        hdu.header["NSTARS"] = (n, "Number of measured stars")
        hdu.header["RSQUARED"] = (result.r_squared, "Calibration R-squared")
        hdu.writeto(path, overwrite=True)
        return path
