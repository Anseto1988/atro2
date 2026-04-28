from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroai.core.calibration.calibrate import apply_dark, apply_flat, calibrate_frame
from astroai.core.calibration.matcher import (
    BatchMatchResult,
    CalibrationFrame,
    CalibrationLibrary,
    FrameMatchResult,
    batch_match,
    find_best_dark,
    find_best_flat,
    suggest_calibration_config,
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


class TestMatcherEdgeCases:
    def test_temperature_large_delta_penalises(self) -> None:
        """Temperature delta >= 2.0 deducts score (line 75 coverage)."""
        light = _meta(exposure=120.0, gain_iso=800, temperature=20.0)
        # Dark at -10 → delta=30 → penalty
        cold_dark = CalibrationFrame(Path("cold.fits"), _meta(exposure=120.0, gain_iso=800, temperature=-10.0))
        # Dark with no temperature set → no temperature penalty
        no_temp_dark = CalibrationFrame(
            Path("notemp.fits"),
            ImageMetadata(exposure=120.0, gain_iso=800, temperature=None, width=100, height=100),
        )
        lib = CalibrationLibrary(darks=[cold_dark, no_temp_dark], flats=[], bias=[])
        best = find_best_dark(light, lib)
        assert best is not None
        assert best.path == Path("notemp.fits")

    def test_from_config_empty_config_produces_empty_library(self) -> None:
        """from_config with config that has no paths returns empty CalibrationLibrary."""

        class _Cfg:
            dark_frames: list[str] = []
            flat_frames: list[str] = []
            bias_frames: list[str] = []

        lib = CalibrationLibrary.from_config(_Cfg())
        assert lib.darks == []
        assert lib.flats == []
        assert lib.bias == []

    def test_from_config_skips_nonexistent_files(self, tmp_path: "Path") -> None:
        """from_config silently skips missing FITS paths."""

        class _Cfg:
            dark_frames = [str(tmp_path / "nope.fits")]
            flat_frames: list[str] = []
            bias_frames: list[str] = []

        lib = CalibrationLibrary.from_config(_Cfg())
        assert lib.darks == []


class TestCalibrateFrameLoadData:
    def test_dark_loaded_via_load_data(self) -> None:
        """calibrate_frame calls load_data for dark when data is None (line 56)."""
        light = np.full((8, 8), 100.0, dtype=np.float32)
        dark_data = np.full((8, 8), 20.0, dtype=np.float32)
        light_meta = _meta(exposure=120.0)

        dark_frame = CalibrationFrame(Path("d.fits"), _meta(exposure=120.0), data=None)
        lib = CalibrationLibrary(darks=[dark_frame], flats=[], bias=[])

        def _load(p: Path) -> "np.ndarray":
            return dark_data

        result = calibrate_frame(light, light_meta, lib, load_data=_load)
        np.testing.assert_array_almost_equal(result, np.full((8, 8), 80.0))

    def test_flat_loaded_via_load_data(self) -> None:
        """calibrate_frame calls load_data for flat when data is None (line 64)."""
        light = np.full((8, 8), 200.0, dtype=np.float32)
        flat_data = np.full((8, 8), 1000.0, dtype=np.float32)
        light_meta = _meta()

        flat_frame = CalibrationFrame(Path("f.fits"), _meta(), data=None)
        lib = CalibrationLibrary(darks=[], flats=[flat_frame], bias=[])

        def _load(p: Path) -> "np.ndarray":
            return flat_data

        result = calibrate_frame(light, light_meta, lib, load_data=_load)
        np.testing.assert_array_almost_equal(result, np.full((8, 8), 200.0))


class TestMatcherFromConfig:
    def test_from_config_loads_valid_fits(self, tmp_path: "Path") -> None:
        """from_config reads valid FITS and stores frame (lines 41-45 coverage)."""
        from astropy.io import fits as astrofits
        fits_path = tmp_path / "dark.fits"
        data = np.zeros((64, 64), dtype=np.float32)
        astrofits.PrimaryHDU(data=data).writeto(fits_path)

        class _Cfg:
            dark_frames = [str(fits_path)]
            flat_frames: list[str] = []
            bias_frames: list[str] = []

        lib = CalibrationLibrary.from_config(_Cfg(), load_data=False)
        assert len(lib.darks) == 1
        assert lib.darks[0].path == fits_path
        assert lib.darks[0].data is None  # load_data=False

    def test_from_config_load_data_true_includes_array(self, tmp_path: "Path") -> None:
        """from_config with load_data=True stores array in frame."""
        from astropy.io import fits as astrofits
        fits_path = tmp_path / "dark.fits"
        data = np.full((16, 16), 42.0, dtype=np.float32)
        astrofits.PrimaryHDU(data=data).writeto(fits_path)

        class _Cfg:
            dark_frames = [str(fits_path)]
            flat_frames: list[str] = []
            bias_frames: list[str] = []

        lib = CalibrationLibrary.from_config(_Cfg(), load_data=True)
        assert len(lib.darks) == 1
        assert lib.darks[0].data is not None

    def test_from_config_invalid_fits_skipped(self, tmp_path: "Path") -> None:
        """from_config skips files that fail to read (exception branch lines 43-45)."""
        bad_path = tmp_path / "corrupt.fits"
        bad_path.write_bytes(b"not a fits file at all")

        class _Cfg:
            dark_frames = [str(bad_path)]
            flat_frames: list[str] = []
            bias_frames: list[str] = []

        lib = CalibrationLibrary.from_config(_Cfg())
        assert lib.darks == []


class TestCalibrateFrameGpuPaths:
    def test_gpu_engine_used_when_non_cpu_device(self) -> None:
        """calibrate_frame uses GPU engine when device_type != 'cpu' (lines 45-46)."""
        light = np.full((8, 8), 100.0, dtype=np.float32)
        light_meta = _meta()
        lib = CalibrationLibrary(darks=[], flats=[], bias=[])
        expected = np.full((8, 8), 50.0, dtype=np.float32)

        mock_engine = MagicMock()
        mock_engine.device_type = "cuda"
        mock_engine.calibrate_frame_gpu.return_value = expected

        with patch("astroai.core.calibration.gpu_engine.GPUCalibrationEngine", return_value=mock_engine):
            result = calibrate_frame(light, light_meta, lib, use_gpu=True)

        np.testing.assert_array_equal(result, expected)
        mock_engine.calibrate_frame_gpu.assert_called_once()

    def test_gpu_engine_exception_falls_back_to_cpu(self) -> None:
        """Exception from GPU engine falls back to CPU path (lines 47-48)."""
        light = np.full((8, 8), 100.0, dtype=np.float32)
        light_meta = _meta()
        lib = CalibrationLibrary(darks=[], flats=[], bias=[])

        mock_engine = MagicMock()
        mock_engine.device_type = "cuda"
        mock_engine.calibrate_frame_gpu.side_effect = RuntimeError("GPU OOM")

        with patch("astroai.core.calibration.gpu_engine.GPUCalibrationEngine", return_value=mock_engine):
            result = calibrate_frame(light, light_meta, lib, use_gpu=True)

        np.testing.assert_array_equal(result, light)


class TestBatchMatch:
    def _lib(self) -> CalibrationLibrary:
        return CalibrationLibrary(
            darks=[CalibrationFrame(Path("d120.fits"), _meta(exposure=120.0))],
            flats=[CalibrationFrame(Path("f800.fits"), _meta(gain_iso=800))],
            bias=[],
        )

    def test_empty_lights_returns_empty(self) -> None:
        result = batch_match([], CalibrationLibrary.empty())
        assert result.matches == []

    def test_coverage_zero_when_no_lights(self) -> None:
        assert batch_match([], CalibrationLibrary.empty()).coverage == 0.0

    def test_single_light_matched(self) -> None:
        lights = [(Path("l1.fits"), _meta(exposure=120.0, gain_iso=800))]
        result = batch_match(lights, self._lib())
        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.light_path == Path("l1.fits")
        assert m.dark is not None
        assert m.flat is not None
        assert m.bias is None

    def test_coverage_full_when_all_matched(self) -> None:
        lights = [(Path("l.fits"), _meta(exposure=120.0, gain_iso=800))]
        result = batch_match(lights, self._lib())
        assert result.coverage == 1.0

    def test_coverage_zero_when_empty_library(self) -> None:
        lights = [(Path("l.fits"), _meta())]
        result = batch_match(lights, CalibrationLibrary.empty())
        assert result.coverage == 0.0

    def test_dark_coverage_property(self) -> None:
        lights = [
            (Path("l1.fits"), _meta(exposure=120.0)),
            (Path("l2.fits"), _meta(exposure=999.0)),  # no dark match
        ]
        lib = CalibrationLibrary(
            darks=[CalibrationFrame(Path("d.fits"), _meta(exposure=120.0))],
            flats=[],
            bias=[],
        )
        result = batch_match(lights, lib)
        assert result.dark_coverage == 0.5

    def test_flat_coverage_property(self) -> None:
        lights = [(Path("l.fits"), _meta(gain_iso=800))]
        lib = CalibrationLibrary(
            darks=[],
            flats=[CalibrationFrame(Path("f.fits"), _meta(gain_iso=800))],
            bias=[],
        )
        result = batch_match(lights, lib)
        assert result.flat_coverage == 1.0

    def test_multiple_lights_get_individual_matches(self) -> None:
        lights = [
            (Path("l1.fits"), _meta(exposure=120.0)),
            (Path("l2.fits"), _meta(exposure=120.0)),
        ]
        result = batch_match(lights, self._lib())
        assert len(result.matches) == 2
        assert all(m.dark is not None for m in result.matches)


class TestSuggestCalibrationConfig:
    def test_returns_calibration_config_with_paths(self) -> None:
        from astroai.project.project_file import CalibrationConfig

        dark = CalibrationFrame(Path("d.fits"), _meta())
        flat = CalibrationFrame(Path("f.fits"), _meta())
        result = BatchMatchResult(
            matches=[FrameMatchResult(light_path=Path("l.fits"), dark=dark, flat=flat, bias=None)]
        )
        cfg = suggest_calibration_config(result)
        assert isinstance(cfg, CalibrationConfig)
        assert "d.fits" in cfg.dark_frames[0]
        assert "f.fits" in cfg.flat_frames[0]

    def test_deduplicates_paths(self) -> None:
        dark = CalibrationFrame(Path("d.fits"), _meta())
        result = BatchMatchResult(
            matches=[
                FrameMatchResult(light_path=Path("l1.fits"), dark=dark, flat=None, bias=None),
                FrameMatchResult(light_path=Path("l2.fits"), dark=dark, flat=None, bias=None),
            ]
        )
        cfg = suggest_calibration_config(result)
        assert len(cfg.dark_frames) == 1

    def test_empty_result_gives_empty_config(self) -> None:
        from astroai.project.project_file import CalibrationConfig

        cfg = suggest_calibration_config(BatchMatchResult(matches=[]))
        assert isinstance(cfg, CalibrationConfig)
        assert cfg.dark_frames == []
        assert cfg.flat_frames == []

    def test_none_matches_excluded(self) -> None:
        result = BatchMatchResult(
            matches=[FrameMatchResult(light_path=Path("l.fits"), dark=None, flat=None, bias=None)]
        )
        cfg = suggest_calibration_config(result)
        assert cfg.dark_frames == []
        assert cfg.flat_frames == []


class TestBatchMatchResultEmptyCoverage:
    """Cover lines 140 and 146: dark_coverage/flat_coverage return 0.0 when matches is empty."""

    def test_dark_coverage_empty_matches_returns_zero(self) -> None:
        result = BatchMatchResult(matches=[])
        assert result.dark_coverage == 0.0

    def test_flat_coverage_empty_matches_returns_zero(self) -> None:
        result = BatchMatchResult(matches=[])
        assert result.flat_coverage == 0.0
