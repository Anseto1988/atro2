from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _get_raw_io(monkeypatch: pytest.MonkeyPatch):
    mock_rawpy = MagicMock()
    monkeypatch.setitem(sys.modules, "rawpy", mock_rawpy)

    mod_key = "astroai.core.io.raw_io"
    if mod_key in sys.modules:
        monkeypatch.delitem(sys.modules, mod_key)

    import importlib
    from astroai.core.io import raw_io as mod

    monkeypatch.setattr(mod, "rawpy", mock_rawpy)
    return mod, mock_rawpy


def _setup_raw_context(mock_rawpy: MagicMock, h: int, w: int, c: int, fill: int = 0):
    fake_rgb = np.full((h, w, c), fill, dtype=np.uint16)

    sizes = MagicMock()
    sizes.height = h
    sizes.width = w

    raw_obj = MagicMock()
    raw_obj.sizes = sizes
    raw_obj.num_colors = c
    raw_obj.postprocess.return_value = fake_rgb

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=raw_obj)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_rawpy.imread.return_value = ctx

    return raw_obj


class TestRawExtensions:
    def test_contains_common_formats(self, monkeypatch) -> None:
        mod, _ = _get_raw_io(monkeypatch)
        for ext in (".cr2", ".nef", ".arw", ".dng"):
            assert ext in mod.RAW_EXTENSIONS

    def test_is_frozen(self, monkeypatch) -> None:
        mod, _ = _get_raw_io(monkeypatch)
        assert isinstance(mod.RAW_EXTENSIONS, frozenset)


class TestReadRaw:
    def test_returns_float32_normalized(self, monkeypatch) -> None:
        mod, mock_rawpy = _get_raw_io(monkeypatch)
        _setup_raw_context(mock_rawpy, 100, 200, 3, fill=32768)

        data, meta = mod.read_raw(Path("/fake/image.cr2"))

        assert data.dtype == np.float32
        assert data.shape == (100, 200, 3)
        np.testing.assert_allclose(data[0, 0, 0], 32768.0 / 65535.0, rtol=1e-5)

    def test_metadata_dimensions(self, monkeypatch) -> None:
        mod, mock_rawpy = _get_raw_io(monkeypatch)
        _setup_raw_context(mock_rawpy, 480, 640, 3)

        _data, meta = mod.read_raw(Path("/fake/image.nef"))

        assert meta.width == 640
        assert meta.height == 480
        assert meta.channels == 3

    def test_calls_postprocess_correctly(self, monkeypatch) -> None:
        mod, mock_rawpy = _get_raw_io(monkeypatch)
        raw_obj = _setup_raw_context(mock_rawpy, 10, 10, 3)

        mod.read_raw(Path("/fake/test.arw"))

        raw_obj.postprocess.assert_called_once_with(
            output_bps=16,
            use_camera_wb=True,
            no_auto_bright=True,
        )


class TestExtractRawMetadataExif:
    def test_exif_all_fields_extracted(self, monkeypatch) -> None:
        """PIL EXIF ExposureTime/ISOSpeedRatings/DateTimeOriginal extracted (lines 31-40)."""
        mod, mock_rawpy = _get_raw_io(monkeypatch)
        _setup_raw_context(mock_rawpy, 100, 200, 3)

        # PIL TAGS maps int → str; tag IDs from TIFF/EXIF spec
        # ExposureTime=33434, ISOSpeedRatings=34855, DateTimeOriginal=36867
        from PIL.ExifTags import TAGS
        name_to_id = {v: k for k, v in TAGS.items()}
        exif_id_exp = name_to_id.get("ExposureTime", 33434)
        exif_id_iso = name_to_id.get("ISOSpeedRatings", 34855)
        exif_id_date = name_to_id.get("DateTimeOriginal", 36867)

        fake_exif = {exif_id_exp: 0.005, exif_id_iso: 1600, exif_id_date: "2024:01:01 12:00:00"}
        mock_img = MagicMock()
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.getexif.return_value = fake_exif

        with patch("PIL.Image.open", return_value=mock_img):
            _data, meta = mod.read_raw(Path("/fake/image.cr2"))

        assert meta.exposure == pytest.approx(0.005)
        assert meta.gain_iso == 1600
        assert meta.date_obs == "2024:01:01 12:00:00"

    def test_empty_exif_leaves_fields_none(self, monkeypatch) -> None:
        """Empty EXIF dict leaves exposure/gain/date as None (line 32 falsy branch)."""
        mod, mock_rawpy = _get_raw_io(monkeypatch)
        _setup_raw_context(mock_rawpy, 50, 50, 3)

        mock_img = MagicMock()
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_img.getexif.return_value = {}  # falsy

        with patch("PIL.Image.open", return_value=mock_img):
            _data, meta = mod.read_raw(Path("/fake/image.cr2"))

        assert meta.exposure is None
        assert meta.gain_iso is None
        assert meta.date_obs is None
