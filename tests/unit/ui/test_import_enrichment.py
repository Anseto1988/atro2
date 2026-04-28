"""Tests for _enrich_fits_entry FITS metadata enrichment."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from astroai.project.project_file import FrameEntry
from astroai.ui.main.app import _enrich_fits_entry


def _make_hdul(header_dict: dict) -> MagicMock:
    mock_hdu = MagicMock()
    mock_hdr = MagicMock()
    mock_hdr.get = lambda k, d=None: header_dict.get(k, d)
    mock_hdu.header = mock_hdr
    mock_hdul = MagicMock()
    mock_hdul.__enter__ = lambda s: mock_hdul
    mock_hdul.__exit__ = MagicMock(return_value=False)
    mock_hdul.__getitem__ = lambda s, i: mock_hdu
    return mock_hdul


class TestEnrichFitsEntry:
    def test_exposure_populated_from_exptime(self) -> None:
        entry = FrameEntry(path="/data/frame.fits")
        with patch("astropy.io.fits.open", return_value=_make_hdul({"EXPTIME": 300.0})):
            _enrich_fits_entry(entry)
        assert entry.exposure == pytest.approx(300.0)

    def test_exposure_populated_from_exposure_alias(self) -> None:
        entry = FrameEntry(path="/data/frame.fits")
        with patch("astropy.io.fits.open", return_value=_make_hdul({"EXPOSURE": 120.0})):
            _enrich_fits_entry(entry)
        assert entry.exposure == pytest.approx(120.0)

    def test_gain_iso_populated(self) -> None:
        entry = FrameEntry(path="/data/frame.fits")
        with patch("astropy.io.fits.open", return_value=_make_hdul({"GAIN": 139})):
            _enrich_fits_entry(entry)
        assert entry.gain_iso == 139

    def test_temperature_populated(self) -> None:
        entry = FrameEntry(path="/data/frame.fit")
        with patch("astropy.io.fits.open", return_value=_make_hdul({"CCD-TEMP": -15.0})):
            _enrich_fits_entry(entry)
        assert entry.temperature == pytest.approx(-15.0)

    def test_temperature_alias_ccd_temp(self) -> None:
        entry = FrameEntry(path="/data/frame.fts")
        with patch("astropy.io.fits.open", return_value=_make_hdul({"CCD_TEMP": -20.0})):
            _enrich_fits_entry(entry)
        assert entry.temperature == pytest.approx(-20.0)

    def test_non_fits_file_is_noop(self) -> None:
        entry = FrameEntry(path="/data/image.tiff")
        _enrich_fits_entry(entry)  # must not touch astropy
        assert entry.exposure is None

    def test_missing_keys_leaves_fields_none(self) -> None:
        entry = FrameEntry(path="/data/frame.fits")
        with patch("astropy.io.fits.open", return_value=_make_hdul({})):
            _enrich_fits_entry(entry)
        assert entry.exposure is None
        assert entry.gain_iso is None
        assert entry.temperature is None

    def test_io_error_is_silenced(self) -> None:
        entry = FrameEntry(path="/nonexistent/frame.fits")
        with patch("astropy.io.fits.open", side_effect=OSError("not found")):
            _enrich_fits_entry(entry)  # must not raise
        assert entry.exposure is None

    def test_non_frameentry_is_noop(self) -> None:
        _enrich_fits_entry(object())  # must not raise

    def test_all_fields_set_together(self) -> None:
        entry = FrameEntry(path="/data/frame.fits")
        with patch(
            "astropy.io.fits.open",
            return_value=_make_hdul({"EXPTIME": 600.0, "GAIN": 76, "CCD-TEMP": -10.0}),
        ):
            _enrich_fits_entry(entry)
        assert entry.exposure == pytest.approx(600.0)
        assert entry.gain_iso == 76
        assert entry.temperature == pytest.approx(-10.0)
