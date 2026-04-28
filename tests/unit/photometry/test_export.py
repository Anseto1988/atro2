from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from astroai.engine.photometry.export import PhotometryExporter
from astroai.engine.photometry.models import PhotometryResult, StarMeasurement


def _sample_result() -> PhotometryResult:
    stars = [
        StarMeasurement(
            star_id=i, ra=180.0 + i * 0.01, dec=45.0 + i * 0.01,
            x_pixel=100.0 + i, y_pixel=200.0 + i,
            instr_mag=-12.0 + i * 0.5,
            catalog_mag=10.0 + i * 0.5,
            cal_mag=10.1 + i * 0.5,
            residual=0.1,
        )
        for i in range(4)
    ]
    return PhotometryResult(stars=stars, r_squared=0.97, n_matched=4)


class TestPhotometryExporterCSV:
    def test_csv_creates_file(self, tmp_path: Path) -> None:
        result = _sample_result()
        exporter = PhotometryExporter()
        out = exporter.to_csv(result, tmp_path / "photometry.csv")
        assert out.exists()

    def test_csv_columns(self, tmp_path: Path) -> None:
        result = _sample_result()
        exporter = PhotometryExporter()
        out = exporter.to_csv(result, tmp_path / "photometry.csv")

        with out.open() as f:
            reader = csv.reader(f)
            header = next(reader)

        expected = ["star_id", "ra", "dec", "instr_mag", "cal_mag", "catalog_mag", "residual"]
        assert header == expected

    def test_csv_row_count(self, tmp_path: Path) -> None:
        result = _sample_result()
        exporter = PhotometryExporter()
        out = exporter.to_csv(result, tmp_path / "photometry.csv")

        with out.open() as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 5  # header + 4 data rows

    def test_csv_empty_result(self, tmp_path: Path) -> None:
        result = PhotometryResult()
        exporter = PhotometryExporter()
        out = exporter.to_csv(result, tmp_path / "empty.csv")

        with out.open() as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 1  # header only


class TestPhotometryExporterFITS:
    def test_fits_creates_file(self, tmp_path: Path) -> None:
        result = _sample_result()
        exporter = PhotometryExporter()
        out = exporter.to_fits(result, tmp_path / "photometry.fits")
        assert out.exists()

    def test_fits_bintable_columns(self, tmp_path: Path) -> None:
        result = _sample_result()
        exporter = PhotometryExporter()
        out = exporter.to_fits(result, tmp_path / "photometry.fits")

        with fits.open(out) as hdul:
            table = hdul[1]
            col_names = [c.name for c in table.columns]

        expected = ["star_id", "ra", "dec", "instr_mag", "cal_mag", "catalog_mag", "residual"]
        assert col_names == expected

    def test_fits_header_values(self, tmp_path: Path) -> None:
        result = _sample_result()
        exporter = PhotometryExporter()
        out = exporter.to_fits(result, tmp_path / "photometry.fits")

        with fits.open(out) as hdul:
            hdr = hdul[1].header

        assert hdr["EXTNAME"] == "PHOTOMETRY"
        assert hdr["NSTARS"] == 4
        assert abs(hdr["RSQUARED"] - 0.97) < 1e-6

    def test_fits_row_count(self, tmp_path: Path) -> None:
        result = _sample_result()
        exporter = PhotometryExporter()
        out = exporter.to_fits(result, tmp_path / "photometry.fits")

        with fits.open(out) as hdul:
            data = hdul[1].data

        assert len(data) == 4

    def test_fits_data_values(self, tmp_path: Path) -> None:
        result = _sample_result()
        exporter = PhotometryExporter()
        out = exporter.to_fits(result, tmp_path / "photometry.fits")

        with fits.open(out) as hdul:
            data = hdul[1].data

        assert data["star_id"][0] == 0
        assert abs(data["ra"][0] - 180.0) < 1e-6
