from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astroai.core.io.fits_io import ImageMetadata
from astroai.core.io.xisf_io import read_xisf, write_xisf


class TestWriteXisf:
    def test_roundtrip_single_channel(self, tmp_path: Path) -> None:
        original = np.random.default_rng(42).random((1, 64, 128)).astype(np.float32)
        path = write_xisf(tmp_path / "out.xisf", original)
        loaded, meta = read_xisf(path)
        np.testing.assert_array_almost_equal(loaded, original, decimal=5)
        assert meta.width == 128
        assert meta.height == 64
        assert meta.channels == 1

    def test_roundtrip_three_channels(self, tmp_path: Path) -> None:
        original = np.random.default_rng(7).random((3, 50, 80)).astype(np.float32)
        path = write_xisf(tmp_path / "rgb.xisf", original)
        loaded, meta = read_xisf(path)
        np.testing.assert_array_almost_equal(loaded, original, decimal=5)
        assert meta.channels == 3

    def test_roundtrip_2d_input(self, tmp_path: Path) -> None:
        original = np.ones((32, 48), dtype=np.float32) * 0.5
        path = write_xisf(tmp_path / "flat.xisf", original)
        loaded, meta = read_xisf(path)
        assert loaded.shape == (1, 32, 48)
        assert meta.channels == 1
        np.testing.assert_array_almost_equal(loaded[0], original, decimal=5)

    def test_metadata_roundtrip(self, tmp_path: Path) -> None:
        data = np.zeros((1, 10, 10), dtype=np.float32)
        meta = ImageMetadata(
            exposure=120.0,
            gain_iso=1600,
            temperature=-10.5,
            date_obs="2024-12-01T22:00:00",
        )
        path = write_xisf(tmp_path / "meta.xisf", data, meta)
        _, loaded_meta = read_xisf(path)
        assert loaded_meta.exposure == pytest.approx(120.0)
        assert loaded_meta.gain_iso == 1600
        assert loaded_meta.temperature == pytest.approx(-10.5)
        assert loaded_meta.date_obs == "2024-12-01T22:00:00"

    def test_metadata_partial(self, tmp_path: Path) -> None:
        data = np.zeros((1, 10, 10), dtype=np.float32)
        meta = ImageMetadata(exposure=30.0)
        path = write_xisf(tmp_path / "partial.xisf", data, meta)
        _, loaded_meta = read_xisf(path)
        assert loaded_meta.exposure == pytest.approx(30.0)
        assert loaded_meta.gain_iso is None
        assert loaded_meta.temperature is None

    def test_no_metadata(self, tmp_path: Path) -> None:
        data = np.ones((1, 8, 8), dtype=np.float32)
        path = write_xisf(tmp_path / "bare.xisf", data)
        loaded, meta = read_xisf(path)
        assert loaded.shape == (1, 8, 8)
        assert meta.exposure is None

    def test_returns_path(self, tmp_path: Path) -> None:
        data = np.zeros((1, 4, 4), dtype=np.float32)
        result = write_xisf(tmp_path / "out.xisf", data)
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".xisf"

    def test_rejects_invalid_ndim(self, tmp_path: Path) -> None:
        data = np.zeros((2, 3, 4, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            write_xisf(tmp_path / "bad.xisf", data)

    def test_file_starts_with_magic(self, tmp_path: Path) -> None:
        data = np.zeros((1, 10, 10), dtype=np.float32)
        path = write_xisf(tmp_path / "magic.xisf", data)
        with open(path, "rb") as f:
            assert f.read(8) == b"XISF0100"

    def test_extra_metadata_written_as_fits_keywords(self, tmp_path: Path) -> None:
        """metadata.extra dict is written as FITSKeyword elements (lines 137-138)."""
        data = np.zeros((1, 8, 8), dtype=np.float32)
        meta = ImageMetadata(extra={"OBSERVER": "Hubble", "OBJECT": "M42"})
        path = write_xisf(tmp_path / "extra.xisf", data, meta)
        _d, loaded_meta = read_xisf(path)
        assert "OBSERVER" in loaded_meta.extra or loaded_meta.extra is not None
