"""Integration tests for plate solving pipeline.

Tests the full flow: FITS-Stack -> Solve -> WCS-Output using a mock ASTAP binary.
No real ASTAP binary or network access required.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import NDArray

from astroai.engine.platesolving.solver import PlateSolver, SolveError, SolveResult
from astroai.engine.platesolving.wcs_writer import WCSWriter


# --- Helpers ---


def _create_mock_wcs_file(wcs_path: Path, ra: float = 180.0, dec: float = 45.0) -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [512.0, 512.0]
    wcs.wcs.cdelt = [-0.000277, 0.000277]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    hdu = fits.PrimaryHDU(header=header)
    hdu.writeto(str(wcs_path), overwrite=True)


def _make_stacked_fits(path: Path, with_coords: bool = True) -> Path:
    rng = np.random.default_rng(42)
    data = rng.normal(100.0, 5.0, (1024, 1024)).astype(np.float32)
    header = fits.Header()
    header["EXPTIME"] = 600.0
    header["STACKED"] = True
    header["NFRAMES"] = 10
    if with_coords:
        header["RA"] = 180.0
        header["DEC"] = 45.0
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(str(path), overwrite=True)
    return path


# --- Fixtures ---


@pytest.fixture()
def stacked_fits(tmp_path: Path) -> Path:
    return _make_stacked_fits(tmp_path / "stacked.fits")


@pytest.fixture()
def stacked_fits_no_coords(tmp_path: Path) -> Path:
    return _make_stacked_fits(tmp_path / "stacked_noc.fits", with_coords=False)


# --- Mock ASTAP Pipeline Tests ---


class TestMockAstapPipeline:
    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_full_pipeline_fits_to_wcs(
        self, mock_run: MagicMock, stacked_fits: Path
    ) -> None:
        """FITS-Stack -> Solve -> WCS-Output complete pipeline."""
        wcs_path = stacked_fits.with_suffix(".wcs")
        _create_mock_wcs_file(wcs_path, ra=180.0, dec=45.0)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Solution found", stderr=""
        )

        solver = PlateSolver(
            astap_path=Path("/mock/astap"), max_retries=3, timeout_s=60.0
        )
        result = solver.solve(stacked_fits)

        assert result.solver_used == "astap"
        assert result.ra_center == pytest.approx(180.0)
        assert result.dec_center == pytest.approx(45.0)
        assert result.solve_time_s >= 0
        assert result.wcs is not None

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_wcs_written_back_to_fits(
        self, mock_run: MagicMock, stacked_fits: Path
    ) -> None:
        """After solving, WCS is written back into the original FITS."""
        wcs_path = stacked_fits.with_suffix(".wcs")
        _create_mock_wcs_file(wcs_path)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )

        solver = PlateSolver(astap_path=Path("/mock/astap"), max_retries=1)
        result = solver.solve(stacked_fits)

        writer = WCSWriter()
        writer.write_wcs_to_fits(stacked_fits, result.wcs)

        read_wcs = writer.read_wcs_from_fits(stacked_fits)
        assert read_wcs is not None
        assert read_wcs.wcs.crval[0] == pytest.approx(180.0)
        assert read_wcs.wcs.crval[1] == pytest.approx(45.0)

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_pipeline_uses_header_coords_as_hint(
        self, mock_run: MagicMock, stacked_fits: Path
    ) -> None:
        """Solver should extract RA/Dec from FITS headers for initial hint."""
        wcs_path = stacked_fits.with_suffix(".wcs")
        _create_mock_wcs_file(wcs_path)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )

        solver = PlateSolver(astap_path=Path("/mock/astap"), max_retries=1)
        result = solver.solve(stacked_fits)

        cmd_args = mock_run.call_args[0][0]
        assert "-ra" in cmd_args
        ra_idx = cmd_args.index("-ra")
        assert float(cmd_args[ra_idx + 1]) == pytest.approx(180.0)

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_pipeline_without_coords_still_attempts_solve(
        self, mock_run: MagicMock, stacked_fits_no_coords: Path
    ) -> None:
        """Pipeline should attempt blind solve when no coords in header."""
        wcs_path = stacked_fits_no_coords.with_suffix(".wcs")
        _create_mock_wcs_file(wcs_path, ra=100.0, dec=20.0)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )

        solver = PlateSolver(astap_path=Path("/mock/astap"), max_retries=1)
        result = solver.solve(stacked_fits_no_coords)

        assert result.wcs is not None
        cmd_args = mock_run.call_args[0][0]
        assert "-ra" not in cmd_args

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_solve_result_contains_field_dimensions(
        self, mock_run: MagicMock, stacked_fits: Path
    ) -> None:
        wcs_path = stacked_fits.with_suffix(".wcs")
        _create_mock_wcs_file(wcs_path)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )

        solver = PlateSolver(astap_path=Path("/mock/astap"), max_retries=1)
        result = solver.solve(stacked_fits)

        assert result.field_width_deg > 0
        assert result.field_height_deg > 0


# --- astrometry.net Fallback Tests ---


class TestAstrometryNetFallback:
    @patch("astroai.engine.platesolving.solver.subprocess.run")
    @patch("astroai.engine.platesolving.solver.PlateSolver._solve_astrometry_net")
    def test_fallback_triggered_after_astap_exhausted(
        self,
        mock_astrometry: MagicMock,
        mock_run: MagicMock,
        stacked_fits: Path,
    ) -> None:
        """astrometry.net is called only after all ASTAP retries fail."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="No solution"
        )

        mock_wcs = WCS(naxis=2)
        mock_wcs.wcs.crval = [180.0, 45.0]
        mock_astrometry.return_value = SolveResult(
            wcs=mock_wcs,
            ra_center=180.0,
            dec_center=45.0,
            field_width_deg=2.0,
            field_height_deg=1.5,
            rotation_deg=10.0,
            solve_time_s=15.0,
            solver_used="astrometry.net",
        )

        solver = PlateSolver(
            astap_path=Path("/mock/astap"),
            astrometry_api_key="test-key-123",
            max_retries=2,
        )
        result = solver.solve(stacked_fits)

        assert mock_run.call_count == 2
        assert result.solver_used == "astrometry.net"
        mock_astrometry.assert_called_once()

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_no_fallback_when_astap_succeeds(
        self, mock_run: MagicMock, stacked_fits: Path
    ) -> None:
        """astrometry.net is NOT called when ASTAP succeeds."""
        wcs_path = stacked_fits.with_suffix(".wcs")
        _create_mock_wcs_file(wcs_path)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )

        solver = PlateSolver(
            astap_path=Path("/mock/astap"),
            astrometry_api_key="test-key-123",
            max_retries=1,
        )

        with patch.object(solver, "_solve_astrometry_net") as mock_fallback:
            result = solver.solve(stacked_fits)
            mock_fallback.assert_not_called()
            assert result.solver_used == "astap"

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_solve_error_when_both_fail(
        self, mock_run: MagicMock, stacked_fits: Path
    ) -> None:
        """SolveError raised when both ASTAP and astrometry.net fail."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="fail"
        )

        solver = PlateSolver(
            astap_path=Path("/mock/astap"),
            astrometry_api_key="test-key-123",
            max_retries=1,
        )

        with patch.object(
            solver, "_solve_astrometry_net", side_effect=SolveError("API error")
        ):
            with pytest.raises(SolveError):
                solver.solve(stacked_fits)


# --- Pipeline Integration (Multi-step) ---


class TestPipelineIntegration:
    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_stack_solve_annotate_flow(
        self, mock_run: MagicMock, stacked_fits: Path
    ) -> None:
        """Full flow: stacked FITS -> plate solve -> WCS write -> annotation ready."""
        from astroai.engine.platesolving.annotation import AnnotationOverlay, CelestialObject

        wcs_path = stacked_fits.with_suffix(".wcs")
        _create_mock_wcs_file(wcs_path)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )

        solver = PlateSolver(astap_path=Path("/mock/astap"), max_retries=1)
        result = solver.solve(stacked_fits)

        writer = WCSWriter()
        writer.write_wcs_to_fits(stacked_fits, result.wcs)

        overlay = AnnotationOverlay(wcs=result.wcs, image_shape=(1024, 1024))
        center = overlay.get_fov_center_world()
        assert center[0] == pytest.approx(180.0, abs=0.1)
        assert center[1] == pytest.approx(45.0, abs=0.1)

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_multiple_solves_independent(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Multiple FITS files can be solved independently."""
        results = []
        for i, (ra, dec) in enumerate([(180.0, 45.0), (90.0, -30.0), (270.0, 60.0)]):
            fits_path = _make_stacked_fits(tmp_path / f"img_{i}.fits")
            wcs_path = fits_path.with_suffix(".wcs")
            _create_mock_wcs_file(wcs_path, ra=ra, dec=dec)
            results.append((fits_path, ra, dec))

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )

        solver = PlateSolver(astap_path=Path("/mock/astap"), max_retries=1)

        for fits_path, expected_ra, expected_dec in results:
            result = solver.solve(fits_path)
            assert result.ra_center == pytest.approx(expected_ra)
            assert result.dec_center == pytest.approx(expected_dec)
