"""Tests for the calibration frame scanner."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits as astrofits

from astroai.core.calibration.matcher import CalibrationLibrary
from astroai.core.calibration.scanner import (
    ScannedFrame,
    _classify_imagetyp,
    build_calibration_library,
    partition_by_type,
    scan_directory,
)


def _write_fits(path: Path, imagetyp: str = "", exposure: float = 120.0) -> None:
    hdu = astrofits.PrimaryHDU(data=np.zeros((32, 32), dtype=np.float32))
    if imagetyp:
        hdu.header["IMAGETYP"] = imagetyp
    hdu.header["EXPTIME"] = exposure
    hdu.writeto(str(path), overwrite=True)


class TestClassifyImagetyp:
    def test_dark(self) -> None:
        assert _classify_imagetyp("Dark Frame") == "dark"
        assert _classify_imagetyp("dark") == "dark"

    def test_flat(self) -> None:
        assert _classify_imagetyp("Flat Field") == "flat"
        assert _classify_imagetyp("flat") == "flat"
        assert _classify_imagetyp("FlatFrame") == "flat"

    def test_bias(self) -> None:
        assert _classify_imagetyp("Bias") == "bias"
        assert _classify_imagetyp("offset") == "bias"

    def test_light(self) -> None:
        assert _classify_imagetyp("Light Frame") == "light"
        assert _classify_imagetyp("science") == "light"

    def test_unknown(self) -> None:
        assert _classify_imagetyp("master_dark_something") == "unknown"
        assert _classify_imagetyp("") == "unknown"

    def test_case_insensitive(self) -> None:
        assert _classify_imagetyp("DARK FRAME") == "dark"
        assert _classify_imagetyp("  Flat  ") == "flat"


class TestScanDirectory:
    def test_nonexistent_directory_returns_empty(self, tmp_path: Path) -> None:
        result = scan_directory(tmp_path / "nope")
        assert result == []

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        assert scan_directory(tmp_path) == []

    def test_non_fits_files_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "readme.txt").write_text("hello")
        assert scan_directory(tmp_path) == []

    def test_classifies_dark_frame(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "dark.fits", imagetyp="Dark Frame")
        results = scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0].frame_type == "dark"
        assert results[0].path == tmp_path / "dark.fits"

    def test_classifies_flat_frame(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "flat.fits", imagetyp="Flat Field")
        results = scan_directory(tmp_path)
        assert results[0].frame_type == "flat"

    def test_classifies_bias_frame(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "bias.fits", imagetyp="Bias")
        results = scan_directory(tmp_path)
        assert results[0].frame_type == "bias"

    def test_classifies_light_frame(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "light.fits", imagetyp="Light Frame")
        results = scan_directory(tmp_path)
        assert results[0].frame_type == "light"

    def test_no_imagetyp_gives_unknown(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "noheader.fits", imagetyp="")
        results = scan_directory(tmp_path)
        assert results[0].frame_type == "unknown"

    def test_corrupt_fits_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "bad.fits").write_bytes(b"not fits")
        _write_fits(tmp_path / "good.fits", imagetyp="Dark Frame")
        results = scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0].frame_type == "dark"

    def test_metadata_populated(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "d.fits", imagetyp="Dark Frame", exposure=300.0)
        results = scan_directory(tmp_path)
        assert results[0].metadata.exposure == pytest.approx(300.0)

    def test_non_recursive_skips_subdirs(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_fits(sub / "d.fits", imagetyp="Dark Frame")
        assert scan_directory(tmp_path, recursive=False) == []

    def test_recursive_finds_nested(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_fits(sub / "d.fits", imagetyp="Dark Frame")
        results = scan_directory(tmp_path, recursive=True)
        assert len(results) == 1

    def test_multiple_types_in_directory(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "d.fits", imagetyp="Dark Frame")
        _write_fits(tmp_path / "f.fits", imagetyp="Flat Field")
        _write_fits(tmp_path / "b.fits", imagetyp="Bias")
        results = scan_directory(tmp_path)
        types = {r.frame_type for r in results}
        assert types == {"dark", "flat", "bias"}


class TestPartitionByType:
    def test_empty_returns_empty_dict(self) -> None:
        assert partition_by_type([]) == {}

    def test_groups_by_type(self, tmp_path: Path) -> None:
        from astroai.core.io.fits_io import ImageMetadata

        frames = [
            ScannedFrame(tmp_path / "d1.fits", "dark", ImageMetadata()),
            ScannedFrame(tmp_path / "d2.fits", "dark", ImageMetadata()),
            ScannedFrame(tmp_path / "f1.fits", "flat", ImageMetadata()),
        ]
        groups = partition_by_type(frames)
        assert len(groups["dark"]) == 2
        assert len(groups["flat"]) == 1
        assert "bias" not in groups


class TestBuildCalibrationLibrary:
    def test_empty_frames_produces_empty_library(self) -> None:
        lib = build_calibration_library([])
        assert isinstance(lib, CalibrationLibrary)
        assert lib.darks == []
        assert lib.flats == []
        assert lib.bias == []

    def test_dark_frames_go_to_darks(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "d.fits", imagetyp="Dark Frame")
        frames = scan_directory(tmp_path)
        lib = build_calibration_library(frames)
        assert isinstance(lib, CalibrationLibrary)
        assert len(lib.darks) == 1
        assert lib.darks[0].path == tmp_path / "d.fits"

    def test_light_frames_excluded(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "l.fits", imagetyp="Light Frame")
        frames = scan_directory(tmp_path)
        lib = build_calibration_library(frames)
        assert lib.darks == []
        assert lib.flats == []
        assert lib.bias == []

    def test_load_data_false_stores_no_array(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "d.fits", imagetyp="Dark Frame")
        frames = scan_directory(tmp_path)
        lib = build_calibration_library(frames, load_data=False)
        assert lib.darks[0].data is None

    def test_load_data_true_stores_array(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "d.fits", imagetyp="Dark Frame")
        frames = scan_directory(tmp_path)
        lib = build_calibration_library(frames, load_data=True)
        assert lib.darks[0].data is not None

    def test_mixed_types_all_classified(self, tmp_path: Path) -> None:
        _write_fits(tmp_path / "d.fits", imagetyp="Dark Frame")
        _write_fits(tmp_path / "f.fits", imagetyp="Flat Field")
        _write_fits(tmp_path / "b.fits", imagetyp="Bias")
        frames = scan_directory(tmp_path)
        lib = build_calibration_library(frames)
        assert len(lib.darks) == 1
        assert len(lib.flats) == 1
        assert len(lib.bias) == 1
