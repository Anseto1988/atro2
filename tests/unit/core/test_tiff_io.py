from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astroai.core.io.fits_io import ImageMetadata
from astroai.core.io.tiff_io import read_tiff, write_tiff32


class TestWriteTiff32:
    def test_roundtrip_2d(self, tmp_path: Path) -> None:
        original = np.random.default_rng(42).random((64, 128)).astype(np.float32)
        path = write_tiff32(tmp_path / "out.tif", original)
        loaded, meta = read_tiff(path)
        assert loaded.dtype == np.float32
        np.testing.assert_array_almost_equal(loaded, original, decimal=5)
        assert meta.width == 128
        assert meta.height == 64

    def test_roundtrip_single_channel_3d(self, tmp_path: Path) -> None:
        original = np.random.default_rng(7).random((1, 32, 48)).astype(np.float32)
        path = write_tiff32(tmp_path / "mono.tif", original)
        loaded, _meta = read_tiff(path)
        if loaded.ndim == 3:
            loaded = loaded[0]
        np.testing.assert_array_almost_equal(loaded, original[0], decimal=5)

    def test_returns_path(self, tmp_path: Path) -> None:
        data = np.zeros((10, 10), dtype=np.float32)
        result = write_tiff32(tmp_path / "out.tif", data)
        assert isinstance(result, Path)
        assert result.exists()

    def test_rejects_invalid_ndim(self, tmp_path: Path) -> None:
        data = np.zeros((2, 3, 4, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            write_tiff32(tmp_path / "bad.tif", data)

    def test_rejects_unsupported_channels(self, tmp_path: Path) -> None:
        data = np.zeros((4, 10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported channel count"):
            write_tiff32(tmp_path / "bad.tif", data)

    def test_metadata_date_obs(self, tmp_path: Path) -> None:
        data = np.zeros((10, 10), dtype=np.float32)
        meta = ImageMetadata(date_obs="2024-12-01T22:00:00")
        path = write_tiff32(tmp_path / "dated.tif", data, meta)
        _, loaded_meta = read_tiff(path)
        assert loaded_meta.date_obs is not None
        assert "2024-12-01" in loaded_meta.date_obs


    def test_roundtrip_rgb_3d(self, tmp_path: Path) -> None:
        """3-channel CHW array is written as RGB TIFF and read back correctly (lines 26-27, 53-56)."""
        rng = np.random.default_rng(99)
        original = rng.random((3, 16, 24)).astype(np.float32)
        path = write_tiff32(tmp_path / "rgb.tif", original)
        loaded, meta = read_tiff(path)
        assert loaded.shape == (3, 16, 24)
        assert meta.channels == 3
        np.testing.assert_array_almost_equal(loaded, original, decimal=5)


class TestReadTiff:
    def test_reads_float32(self, tmp_path: Path) -> None:
        data = np.ones((20, 30), dtype=np.float32) * 0.75
        path = write_tiff32(tmp_path / "test.tif", data)
        loaded, _meta = read_tiff(path)
        assert loaded.dtype == np.float32

    def test_read_16bit_integer_tiff(self, tmp_path: Path) -> None:
        """Reads 16-bit integer TIFF (I mode) and returns float32 (lines 57-59)."""
        from PIL import Image as PILImage
        data_uint16 = np.full((20, 30), 32768, dtype=np.uint32)  # PIL 'I' mode is int32
        img = PILImage.fromarray(data_uint16, mode="I")
        path = tmp_path / "int16.tif"
        img.save(str(path), format="TIFF")
        loaded, meta = read_tiff(path)
        assert loaded.dtype == np.float32
        assert meta.width == 30
        assert meta.height == 20

    def test_read_rgba_tiff_else_branch(self, tmp_path: Path) -> None:
        """RGBA TIFF falls into else branch, returns CHW float32 (lines 60-66)."""
        from PIL import Image as PILImage
        rgba = np.full((20, 30, 4), 128, dtype=np.uint8)
        img = PILImage.fromarray(rgba, mode="RGBA")
        path = tmp_path / "rgba.tif"
        img.save(str(path), format="TIFF")
        loaded, meta = read_tiff(path)
        assert loaded.dtype == np.float32
        assert loaded.ndim == 3  # CHW after transpose
