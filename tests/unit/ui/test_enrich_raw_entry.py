"""Unit tests for _enrich_raw_entry in app.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astroai.project.project_file import FrameEntry


def _enrich_raw():
    from astroai.ui.main.app import _enrich_raw_entry
    return _enrich_raw_entry


class TestEnrichRawEntrySkips:
    def test_non_frame_entry_is_no_op(self) -> None:
        fn = _enrich_raw()
        fn("not a FrameEntry")  # must not raise

    def test_fits_file_is_skipped(self) -> None:
        fn = _enrich_raw()
        entry = FrameEntry(path="/data/frame.fits")
        fn(entry)
        assert entry.exposure is None

    def test_tiff_file_is_skipped(self) -> None:
        fn = _enrich_raw()
        entry = FrameEntry(path="/data/frame.tiff")
        fn(entry)
        assert entry.exposure is None

    def test_png_file_is_skipped(self) -> None:
        fn = _enrich_raw()
        entry = FrameEntry(path="/data/frame.png")
        fn(entry)
        assert entry.exposure is None


class TestEnrichRawEntryPopulates:
    def _make_meta(self, exposure=None, gain_iso=None, temperature=None):
        m = MagicMock()
        m.exposure = exposure
        m.gain_iso = gain_iso
        m.temperature = temperature
        return m

    def test_cr2_exposure_populated(self) -> None:
        fn = _enrich_raw()
        entry = FrameEntry(path="/data/img.cr2")
        meta = self._make_meta(exposure=0.01)
        with patch("astroai.core.io.raw_io.read_raw_metadata", return_value=meta):
            fn(entry)
        assert entry.exposure == pytest.approx(0.01)

    def test_nef_gain_iso_populated(self) -> None:
        fn = _enrich_raw()
        entry = FrameEntry(path="/data/img.nef")
        meta = self._make_meta(gain_iso=3200)
        with patch("astroai.core.io.raw_io.read_raw_metadata", return_value=meta):
            fn(entry)
        assert entry.gain_iso == 3200

    def test_arw_all_fields_populated(self) -> None:
        fn = _enrich_raw()
        entry = FrameEntry(path="/data/img.arw")
        meta = self._make_meta(exposure=0.5, gain_iso=800, temperature=-10.5)
        with patch("astroai.core.io.raw_io.read_raw_metadata", return_value=meta):
            fn(entry)
        assert entry.exposure == pytest.approx(0.5)
        assert entry.gain_iso == 800
        assert entry.temperature == pytest.approx(-10.5)

    def test_none_fields_not_overwritten(self) -> None:
        fn = _enrich_raw()
        entry = FrameEntry(path="/data/img.cr2", exposure=1.0)
        # metadata returns None for exposure → existing value must stay
        meta = self._make_meta(exposure=None, gain_iso=400)
        with patch("astroai.core.io.raw_io.read_raw_metadata", return_value=meta):
            fn(entry)
        assert entry.exposure == pytest.approx(1.0)
        assert entry.gain_iso == 400

    def test_exception_in_read_raw_metadata_is_silenced(self) -> None:
        fn = _enrich_raw()
        entry = FrameEntry(path="/data/img.cr2")
        with patch("astroai.core.io.raw_io.read_raw_metadata", side_effect=OSError("fail")):
            fn(entry)  # must not raise
        assert entry.exposure is None
