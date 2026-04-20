from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astroai.core.calibration.calibrate import apply_dark, apply_flat, calibrate_frame
from astroai.core.calibration.matcher import (
    CalibrationFrame,
    CalibrationLibrary,
    find_best_dark,
    find_best_flat,
)
from astroai.core.io.fits_io import ImageMetadata


def _meta(
    exposure: float = 120.0,
    gain_iso: int = 800,
    temperature: float = -10.0,
    width: int = 100,
    height: int = 100,
) -> ImageMetadata:
    return ImageMetadata(
        exposure=exposure,
        gain_iso=gain_iso,
        temperature=temperature,
        width=width,
        height=height,
    )


class TestApplyDark:
    def test_subtracts_dark(self) -> None:
        light = np.full((10, 10), 100.0, dtype=np.float32)
        dark = np.full((10, 10), 30.0, dtype=np.float32)
        result = apply_dark(light, dark)
        np.testing.assert_array_almost_equal(result, np.full((10, 10), 70.0))

    def test_clips_to_zero(self) -> None:
        light = np.full((5, 5), 10.0, dtype=np.float32)
        dark = np.full((5, 5), 50.0, dtype=np.float32)
        result = apply_dark(light, dark)
        assert result.min() >= 0.0


class TestApplyFlat:
    def test_divides_by_normalized_flat(self) -> None:
        light = np.full((10, 10), 500.0, dtype=np.float32)
        flat = np.full((10, 10), 1000.0, dtype=np.float32)
        result = apply_flat(light, flat)
        np.testing.assert_array_almost_equal(result, np.full((10, 10), 500.0))

    def test_handles_uneven_flat(self) -> None:
        light = np.full((4, 4), 100.0, dtype=np.float32)
        flat = np.ones((4, 4), dtype=np.float32)
        flat[:2, :] = 0.5
        flat[2:, :] = 1.0
        result = apply_flat(light, flat)
        assert result[:2, :].mean() > result[2:, :].mean()


class TestMatcher:
    def test_find_best_dark_exact_match(self) -> None:
        light_meta = _meta(exposure=120.0, gain_iso=800, temperature=-10.0)
        lib = CalibrationLibrary(
            darks=[
                CalibrationFrame(Path("d1.fits"), _meta(exposure=60.0, gain_iso=800)),
                CalibrationFrame(Path("d2.fits"), _meta(exposure=120.0, gain_iso=800, temperature=-10.0)),
            ],
            flats=[],
            bias=[],
        )
        best = find_best_dark(light_meta, lib)
        assert best is not None
        assert best.path == Path("d2.fits")

    def test_find_best_dark_empty_library(self) -> None:
        assert find_best_dark(_meta(), CalibrationLibrary.empty()) is None

    def test_find_best_flat_prefers_matching_iso(self) -> None:
        light_meta = _meta(gain_iso=1600)
        lib = CalibrationLibrary(
            darks=[],
            flats=[
                CalibrationFrame(Path("f1.fits"), _meta(gain_iso=800)),
                CalibrationFrame(Path("f2.fits"), _meta(gain_iso=1600)),
            ],
            bias=[],
        )
        best = find_best_flat(light_meta, lib)
        assert best is not None
        assert best.path == Path("f2.fits")

    def test_no_match_returns_none_for_bad_dimensions(self) -> None:
        light_meta = _meta(width=4000, height=3000)
        lib = CalibrationLibrary(
            darks=[CalibrationFrame(Path("d.fits"), _meta(width=2000, height=1500))],
            flats=[],
            bias=[],
        )
        assert find_best_dark(light_meta, lib) is None


class TestCalibrateFrame:
    def test_applies_dark_and_flat(self) -> None:
        light = np.full((10, 10), 200.0, dtype=np.float32)
        light_meta = _meta(exposure=120.0, gain_iso=800, temperature=-10.0)
        dark_data = np.full((10, 10), 20.0, dtype=np.float32)
        flat_data = np.full((10, 10), 1000.0, dtype=np.float32)

        lib = CalibrationLibrary(
            darks=[CalibrationFrame(Path("d.fits"), _meta(), data=dark_data)],
            flats=[CalibrationFrame(Path("f.fits"), _meta(), data=flat_data)],
            bias=[],
        )
        result = calibrate_frame(light, light_meta, lib)
        expected_after_dark = 180.0
        np.testing.assert_array_almost_equal(result, np.full((10, 10), expected_after_dark))

    def test_no_calibration_returns_copy(self) -> None:
        light = np.full((5, 5), 100.0, dtype=np.float32)
        result = calibrate_frame(light, _meta(), CalibrationLibrary.empty())
        np.testing.assert_array_equal(result, light)
        assert result is not light
