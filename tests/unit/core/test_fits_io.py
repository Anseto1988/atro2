from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from astroai.core.io.fits_io import ImageMetadata, read_fits, write_fits


@pytest.fixture()
def sample_fits(tmp_path: Path) -> Path:
    data = np.random.default_rng(42).random((100, 200), dtype=np.float32)
    header = fits.Header()
    header["EXPTIME"] = 120.0
    header["GAIN"] = 139
    header["CCD-TEMP"] = -10.5
    header["DATE-OBS"] = "2024-07-15T23:30:00"
    hdu = fits.PrimaryHDU(data=data, header=header)
    path = tmp_path / "test.fits"
    hdu.writeto(str(path))
    return path


class TestReadFits:
    def test_returns_float32_array(self, sample_fits: Path) -> None:
        data, _meta = read_fits(sample_fits)
        assert data.dtype == np.float32

    def test_shape_matches(self, sample_fits: Path) -> None:
        data, meta = read_fits(sample_fits)
        assert data.shape == (100, 200)
        assert meta.height == 100
        assert meta.width == 200

    def test_extracts_exposure(self, sample_fits: Path) -> None:
        _data, meta = read_fits(sample_fits)
        assert meta.exposure == pytest.approx(120.0)

    def test_extracts_gain(self, sample_fits: Path) -> None:
        _data, meta = read_fits(sample_fits)
        assert meta.gain_iso == 139

    def test_extracts_temperature(self, sample_fits: Path) -> None:
        _data, meta = read_fits(sample_fits)
        assert meta.temperature == pytest.approx(-10.5)

    def test_extracts_date(self, sample_fits: Path) -> None:
        _data, meta = read_fits(sample_fits)
        assert meta.date_obs == "2024-07-15T23:30:00"


class TestWriteFits:
    def test_roundtrip(self, tmp_path: Path) -> None:
        original = np.random.default_rng(7).random((50, 80), dtype=np.float32)
        meta = ImageMetadata(
            exposure=30.0, gain_iso=800, temperature=-15.0, date_obs="2024-08-01T01:00:00"
        )
        path = write_fits(tmp_path / "out.fits", original, meta)
        loaded, loaded_meta = read_fits(path)
        np.testing.assert_array_almost_equal(loaded, original, decimal=5)
        assert loaded_meta.exposure == pytest.approx(30.0)
        assert loaded_meta.gain_iso == 800

    def test_writes_without_metadata(self, tmp_path: Path) -> None:
        data = np.ones((10, 10), dtype=np.float32)
        path = write_fits(tmp_path / "bare.fits", data)
        loaded, meta = read_fits(path)
        assert loaded.shape == (10, 10)
        assert meta.exposure is None


class TestCoreIoInit:
    def test_read_raw_lazy_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """core.io.__getattr__ provides read_raw lazily (lines 21-22)."""
        import sys
        from unittest.mock import MagicMock

        mock_rawpy = MagicMock()
        monkeypatch.setitem(sys.modules, "rawpy", mock_rawpy)
        # Force reload of raw_io so the mock takes effect
        monkeypatch.delitem(sys.modules, "astroai.core.io.raw_io", raising=False)

        import astroai.core.io as io_pkg
        fn = io_pkg.read_raw
        assert callable(fn)

    def test_raw_extensions_lazy_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """core.io.__getattr__ provides RAW_EXTENSIONS lazily (lines 24-25)."""
        import sys
        from unittest.mock import MagicMock

        mock_rawpy = MagicMock()
        monkeypatch.setitem(sys.modules, "rawpy", mock_rawpy)
        monkeypatch.delitem(sys.modules, "astroai.core.io.raw_io", raising=False)

        import astroai.core.io as io_pkg
        exts = io_pkg.RAW_EXTENSIONS
        assert ".cr2" in exts

    def test_unknown_attr_raises(self) -> None:
        """core.io.__getattr__ raises AttributeError for unknown names."""
        import astroai.core.io as io_pkg
        with pytest.raises(AttributeError):
            _ = io_pkg.nonexistent_attribute  # type: ignore[attr-defined]
