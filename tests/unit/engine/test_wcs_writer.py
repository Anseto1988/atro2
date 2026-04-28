"""Unit tests for WCSWriter — FITS WCS read/write/create."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from astroai.engine.platesolving.wcs_writer import WCSWriter


def _make_fits(tmp_path: Path, shape: tuple[int, int] = (64, 64)) -> Path:
    p = tmp_path / "test.fits"
    fits.writeto(str(p), np.zeros(shape, dtype=np.float32), overwrite=True)
    return p


def _make_wcs(ra: float = 180.0, dec: float = 45.0, scale: float = 0.000277) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [32.0, 32.0]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-scale, scale]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (64, 64)
    return w


class TestWCSWriterRoundtrip:
    def test_write_then_read_returns_wcs(self, tmp_path: Path) -> None:
        p = _make_fits(tmp_path)
        writer = WCSWriter()
        wcs = _make_wcs()
        writer.write_wcs_to_fits(p, wcs)
        result = writer.read_wcs_from_fits(p)
        assert result is not None

    def test_write_returns_same_path(self, tmp_path: Path) -> None:
        p = _make_fits(tmp_path)
        writer = WCSWriter()
        returned = writer.write_wcs_to_fits(p, _make_wcs())
        assert returned == p

    def test_crval_preserved(self, tmp_path: Path) -> None:
        p = _make_fits(tmp_path)
        writer = WCSWriter()
        writer.write_wcs_to_fits(p, _make_wcs(ra=10.5, dec=-33.2))
        result = writer.read_wcs_from_fits(p)
        assert result is not None
        assert result.wcs.crval[0] == pytest.approx(10.5, abs=0.001)
        assert result.wcs.crval[1] == pytest.approx(-33.2, abs=0.001)

    def test_ctype_preserved(self, tmp_path: Path) -> None:
        p = _make_fits(tmp_path)
        writer = WCSWriter()
        writer.write_wcs_to_fits(p, _make_wcs())
        result = writer.read_wcs_from_fits(p)
        assert result is not None
        assert "RA" in result.wcs.ctype[0]
        assert "DEC" in result.wcs.ctype[1]

    def test_read_no_wcs_returns_none(self, tmp_path: Path) -> None:
        p = _make_fits(tmp_path)
        writer = WCSWriter()
        result = writer.read_wcs_from_fits(p)
        assert result is None


class TestWCSWriterCreate:
    def test_create_wcs_identity_rotation(self) -> None:
        writer = WCSWriter()
        wcs = writer.create_wcs(
            crval=(180.0, 45.0),
            crpix=(512.0, 512.0),
            cdelt=(0.001, 0.001),
            naxis=(1024, 1024),
            rotation_deg=0.0,
        )
        assert wcs.wcs.crval[0] == pytest.approx(180.0)
        assert wcs.wcs.crval[1] == pytest.approx(45.0)

    def test_create_wcs_rotation_matrix(self) -> None:
        writer = WCSWriter()
        wcs = writer.create_wcs(
            crval=(0.0, 0.0),
            crpix=(256.0, 256.0),
            cdelt=(0.001, 0.001),
            naxis=(512, 512),
            rotation_deg=90.0,
        )
        cd = wcs.wcs.cd
        # At 90°: cos=0, sin=1 → off-diagonal dominates
        assert abs(cd[0][0]) == pytest.approx(0.0, abs=1e-10)
        assert abs(cd[1][1]) == pytest.approx(0.0, abs=1e-10)

    def test_create_wcs_zero_rotation_cd_diagonal(self) -> None:
        writer = WCSWriter()
        wcs = writer.create_wcs(
            crval=(0.0, 0.0),
            crpix=(256.0, 256.0),
            cdelt=(0.002, 0.002),
            naxis=(512, 512),
            rotation_deg=0.0,
        )
        cd = wcs.wcs.cd
        # At 0°: cos=1, sin=0 → diagonal terms equal cdelt
        assert cd[0][0] == pytest.approx(0.002, abs=1e-12)
        assert cd[1][1] == pytest.approx(0.002, abs=1e-12)
        assert abs(cd[0][1]) == pytest.approx(0.0, abs=1e-12)
        assert abs(cd[1][0]) == pytest.approx(0.0, abs=1e-12)

    def test_create_wcs_custom_ctype(self) -> None:
        writer = WCSWriter()
        wcs = writer.create_wcs(
            crval=(0.0, 0.0),
            crpix=(128.0, 128.0),
            cdelt=(0.001, 0.001),
            naxis=(256, 256),
            ctype=("RA---TAN", "DEC--TAN"),
        )
        assert wcs.wcs.ctype[0] == "RA---TAN"
        assert wcs.wcs.ctype[1] == "DEC--TAN"

    def test_write_created_wcs_roundtrip(self, tmp_path: Path) -> None:
        writer = WCSWriter()
        wcs = writer.create_wcs(
            crval=(83.8, -5.4),
            crpix=(512.0, 512.0),
            cdelt=(0.000277, 0.000277),
            naxis=(1024, 1024),
        )
        p = _make_fits(tmp_path, shape=(1024, 1024))
        writer.write_wcs_to_fits(p, wcs)
        recovered = writer.read_wcs_from_fits(p)
        assert recovered is not None
        assert recovered.wcs.crval[0] == pytest.approx(83.8, abs=0.001)
