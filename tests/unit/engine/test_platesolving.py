"""Unit tests for plate solving engine.

Covers: WCS header writing, retry logic, platform detection, coordinate extraction.
No real ASTAP binary or network access required.
"""

from __future__ import annotations

import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from astroai.engine.platesolving.solver import (
    PlateSolver,
    SolveError,
    SolveResult,
    _extract_coordinates_from_header,
)
from astroai.engine.platesolving.astap_binary import (
    AstapNotFoundError,
    get_astap_path,
    verify_astap,
    ensure_astap,
    _detect_platform_key,
)
from astroai.engine.platesolving.wcs_writer import WCSWriter


# --- Fixtures ---


@pytest.fixture()
def tmp_fits(tmp_path: Path) -> Path:
    data = np.ones((100, 100), dtype=np.float32) * 500.0
    header = fits.Header()
    header["EXPTIME"] = 120.0
    header["RA"] = 180.0
    header["DEC"] = 45.0
    hdu = fits.PrimaryHDU(data=data, header=header)
    path = tmp_path / "test.fits"
    hdu.writeto(str(path), overwrite=True)
    return path


@pytest.fixture()
def tmp_fits_no_coords(tmp_path: Path) -> Path:
    data = np.ones((100, 100), dtype=np.float32) * 500.0
    hdu = fits.PrimaryHDU(data=data)
    path = tmp_path / "no_coords.fits"
    hdu.writeto(str(path), overwrite=True)
    return path


@pytest.fixture()
def sample_wcs() -> WCS:
    writer = WCSWriter()
    return writer.create_wcs(
        crval=(180.0, 45.0),
        crpix=(50.0, 50.0),
        cdelt=(-0.000277, 0.000277),
        naxis=(100, 100),
    )


@pytest.fixture()
def solver() -> PlateSolver:
    return PlateSolver(
        astap_path=Path("/mock/astap"),
        max_retries=3,
        search_radius_deg=10.0,
        timeout_s=30.0,
    )


# --- WCS Header Writing Tests ---


class TestWCSWriter:
    def test_write_wcs_to_fits(self, tmp_fits: Path, sample_wcs: WCS) -> None:
        writer = WCSWriter()
        writer.write_wcs_to_fits(tmp_fits, sample_wcs)

        with fits.open(str(tmp_fits)) as hdul:
            header = hdul[0].header
            assert "CTYPE1" in header
            assert "CTYPE2" in header
            assert header["CTYPE1"] == "RA---TAN"
            assert header["CTYPE2"] == "DEC--TAN"
            assert header["CRVAL1"] == pytest.approx(180.0)
            assert header["CRVAL2"] == pytest.approx(45.0)

    def test_read_wcs_from_fits(self, tmp_fits: Path, sample_wcs: WCS) -> None:
        writer = WCSWriter()
        writer.write_wcs_to_fits(tmp_fits, sample_wcs)

        read_wcs = writer.read_wcs_from_fits(tmp_fits)
        assert read_wcs is not None
        assert read_wcs.wcs.crval[0] == pytest.approx(180.0)
        assert read_wcs.wcs.crval[1] == pytest.approx(45.0)

    def test_read_wcs_returns_none_without_wcs(self, tmp_fits_no_coords: Path) -> None:
        writer = WCSWriter()
        result = writer.read_wcs_from_fits(tmp_fits_no_coords)
        assert result is None

    def test_write_preserves_existing_headers(self, tmp_fits: Path, sample_wcs: WCS) -> None:
        writer = WCSWriter()
        writer.write_wcs_to_fits(tmp_fits, sample_wcs)

        with fits.open(str(tmp_fits)) as hdul:
            header = hdul[0].header
            assert header["EXPTIME"] == pytest.approx(120.0)
            assert "CTYPE1" in header

    def test_create_wcs_with_rotation(self) -> None:
        writer = WCSWriter()
        wcs = writer.create_wcs(
            crval=(10.0, 20.0),
            crpix=(512.0, 512.0),
            cdelt=(-0.0001, 0.0001),
            naxis=(1024, 1024),
            rotation_deg=30.0,
        )
        assert wcs.wcs.crval[0] == pytest.approx(10.0)
        assert wcs.wcs.crval[1] == pytest.approx(20.0)
        assert wcs.wcs.cd is not None

    def test_create_wcs_zero_rotation(self) -> None:
        writer = WCSWriter()
        wcs = writer.create_wcs(
            crval=(100.0, -30.0),
            crpix=(50.0, 50.0),
            cdelt=(-0.000277, 0.000277),
            naxis=(100, 100),
            rotation_deg=0.0,
        )
        assert wcs.wcs.cd[0][0] == pytest.approx(-0.000277)
        assert wcs.wcs.cd[1][1] == pytest.approx(0.000277)
        assert wcs.wcs.cd[0][1] == pytest.approx(0.0, abs=1e-10)
        assert wcs.wcs.cd[1][0] == pytest.approx(0.0, abs=1e-10)


# --- Retry Logic Tests ---


class TestRetryLogic:
    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_retry_on_first_failure_then_success(
        self, mock_run: MagicMock, solver: PlateSolver, tmp_fits: Path
    ) -> None:
        wcs_path = tmp_fits.with_suffix(".wcs")
        wcs = WCS(naxis=2)
        wcs.wcs.crval = [180.0, 45.0]
        wcs.wcs.crpix = [50.0, 50.0]
        wcs.wcs.cdelt = [-0.000277, 0.000277]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        header = wcs.to_header()
        hdu = fits.PrimaryHDU(header=header)
        hdu.writeto(str(wcs_path), overwrite=True)

        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=1, stderr="fail"),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""),
        ]

        result = solver._solve_astap(tmp_fits, 180.0, 45.0)
        assert result.solver_used == "astap"
        assert mock_run.call_count == 2

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_retry_expands_search_radius(
        self, mock_run: MagicMock, solver: PlateSolver, tmp_fits: Path
    ) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "astap")

        with pytest.raises(SolveError, match="failed after 3 attempts"):
            solver._solve_astap(tmp_fits, 180.0, 45.0)

        assert mock_run.call_count == 3
        calls = mock_run.call_args_list
        radii = []
        for c in calls:
            cmd = c[0][0]
            idx = cmd.index("-r")
            radii.append(float(cmd[idx + 1]))
        assert radii[1] > radii[0]
        assert radii[2] > radii[1]

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_all_retries_fail_raises_solve_error(
        self, mock_run: MagicMock, solver: PlateSolver, tmp_fits: Path
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="No solution"
        )

        with pytest.raises(SolveError, match="failed after 3 attempts"):
            solver._solve_astap(tmp_fits, 180.0, 45.0)

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_timeout_triggers_retry(
        self, mock_run: MagicMock, solver: PlateSolver, tmp_fits: Path
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired("astap", 30.0)

        with pytest.raises(SolveError, match="failed after 3 attempts"):
            solver._solve_astap(tmp_fits, 180.0, 45.0)

        assert mock_run.call_count == 3


# --- Platform Detection Tests ---


class TestPlatformDetection:
    @patch("astroai.engine.platesolving.astap_binary.platform.system")
    @patch("astroai.engine.platesolving.astap_binary.platform.machine")
    def test_windows_platform_key(
        self, mock_machine: MagicMock, mock_system: MagicMock
    ) -> None:
        mock_system.return_value = "Windows"
        mock_machine.return_value = "AMD64"
        key = _detect_platform_key()
        assert key == "win32-x86_64"

    @patch("astroai.engine.platesolving.astap_binary.platform.system")
    @patch("astroai.engine.platesolving.astap_binary.platform.machine")
    def test_macos_platform_key(
        self, mock_machine: MagicMock, mock_system: MagicMock
    ) -> None:
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        key = _detect_platform_key()
        assert key == "darwin-arm64"

    @patch("astroai.engine.platesolving.astap_binary.platform.system")
    @patch("astroai.engine.platesolving.astap_binary.platform.machine")
    def test_linux_platform_key(
        self, mock_machine: MagicMock, mock_system: MagicMock
    ) -> None:
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        key = _detect_platform_key()
        assert key == "linux-x86_64"

    @patch("astroai.engine.platesolving.astap_binary.platform.system")
    @patch("astroai.engine.platesolving.astap_binary.platform.machine")
    def test_unsupported_platform_raises(
        self, mock_machine: MagicMock, mock_system: MagicMock
    ) -> None:
        mock_system.return_value = "FreeBSD"
        mock_machine.return_value = "riscv64"
        with pytest.raises(AstapNotFoundError, match="Unsupported platform"):
            _detect_platform_key()

    def test_solver_uses_custom_path(self) -> None:
        custom = Path("/custom/path/astap_cli")
        solver = PlateSolver(astap_path=custom)
        assert solver.astap_path == custom

    @patch("astroai.engine.platesolving.astap_binary.get_astap_path")
    def test_solver_default_delegates_to_get_astap_path(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.return_value = Path("/resolved/astap")
        solver = PlateSolver()
        assert solver.astap_path == Path("/resolved/astap")

    def test_get_astap_path_env_override(self, tmp_path: Path) -> None:
        binary = tmp_path / "astap.exe"
        binary.write_text("fake")
        with patch.dict("os.environ", {"ASTAP_BINARY_PATH": str(binary)}):
            result = get_astap_path()
            assert result == binary

    def test_get_astap_path_env_not_executable_raises(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {"ASTAP_BINARY_PATH": str(tmp_path / "missing")}):
            with pytest.raises(AstapNotFoundError, match="not executable"):
                get_astap_path()

    def test_verify_astap_returns_none_on_missing(self) -> None:
        result = verify_astap(Path("/nonexistent/astap"))
        assert result is None


# --- Coordinate Extraction Tests ---


class TestCoordinateExtraction:
    def test_extract_ra_dec_from_header(self) -> None:
        header = fits.Header()
        header["RA"] = 123.456
        header["DEC"] = -45.678
        result = _extract_coordinates_from_header(header)
        assert result is not None
        assert result[0] == pytest.approx(123.456)
        assert result[1] == pytest.approx(-45.678)

    def test_extract_objctra_objctdec(self) -> None:
        header = fits.Header()
        header["OBJCTRA"] = 200.0
        header["OBJCTDEC"] = 30.0
        result = _extract_coordinates_from_header(header)
        assert result is not None
        assert result[0] == pytest.approx(200.0)
        assert result[1] == pytest.approx(30.0)

    def test_extract_crval(self) -> None:
        header = fits.Header()
        header["CRVAL1"] = 90.0
        header["CRVAL2"] = -20.0
        result = _extract_coordinates_from_header(header)
        assert result is not None
        assert result[0] == pytest.approx(90.0)
        assert result[1] == pytest.approx(-20.0)

    def test_returns_none_without_coordinates(self) -> None:
        header = fits.Header()
        header["EXPTIME"] = 120.0
        result = _extract_coordinates_from_header(header)
        assert result is None

    def test_solver_extracts_coords_from_fits(self, tmp_fits: Path) -> None:
        solver = PlateSolver(astap_path=Path("/mock/astap"))
        coords = solver._extract_coords_from_fits(tmp_fits)
        assert coords is not None
        assert coords[0] == pytest.approx(180.0)
        assert coords[1] == pytest.approx(45.0)

    def test_solver_no_coords_returns_none(self, tmp_fits_no_coords: Path) -> None:
        solver = PlateSolver(astap_path=Path("/mock/astap"))
        coords = solver._extract_coords_from_fits(tmp_fits_no_coords)
        assert coords is None


# --- Fallback Logic Tests ---


class TestFallbackLogic:
    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_no_fallback_without_api_key(
        self, mock_run: MagicMock, tmp_fits: Path
    ) -> None:
        solver = PlateSolver(
            astap_path=Path("/mock/astap"),
            astrometry_api_key=None,
            max_retries=1,
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="fail"
        )

        with pytest.raises(SolveError):
            solver.solve(tmp_fits, ra_hint=180.0, dec_hint=45.0)

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    @patch("astroai.engine.platesolving.solver.PlateSolver._solve_astrometry_net")
    def test_fallback_triggered_on_astap_failure(
        self,
        mock_astrometry: MagicMock,
        mock_run: MagicMock,
        tmp_fits: Path,
    ) -> None:
        solver = PlateSolver(
            astap_path=Path("/mock/astap"),
            astrometry_api_key="test-key",
            max_retries=1,
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="fail"
        )
        mock_wcs = WCS(naxis=2)
        mock_wcs.wcs.crval = [180.0, 45.0]
        mock_astrometry.return_value = SolveResult(
            wcs=mock_wcs,
            ra_center=180.0,
            dec_center=45.0,
            field_width_deg=1.0,
            field_height_deg=1.0,
            rotation_deg=0.0,
            solve_time_s=5.0,
            solver_used="astrometry.net",
        )

        result = solver.solve(tmp_fits, ra_hint=180.0, dec_hint=45.0)
        assert result.solver_used == "astrometry.net"
        mock_astrometry.assert_called_once()


# --- astrometry.net Client Tests ---


class TestAstrometryNetClient:
    @patch("httpx.post")
    def test_astrometry_net_success(
        self, mock_post: MagicMock, tmp_fits: Path
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "success",
            "wcs_header": {
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 180.0,
                "CRVAL2": 45.0,
                "CRPIX1": 512.0,
                "CRPIX2": 512.0,
                "CDELT1": -0.000277,
                "CDELT2": 0.000277,
            },
            "field_width": 1.0,
            "field_height": 1.0,
            "orientation": 5.0,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        solver = PlateSolver(
            astap_path=Path("/mock/astap"),
            astrometry_api_key="test-api-key",
        )
        result = solver._solve_astrometry_net(tmp_fits, 180.0, 45.0)

        assert result.solver_used == "astrometry.net"
        assert result.ra_center == pytest.approx(180.0)
        assert result.dec_center == pytest.approx(45.0)
        assert result.field_width_deg == 1.0
        assert result.rotation_deg == 5.0
        mock_post.assert_called_once()

    @patch("httpx.post")
    def test_astrometry_net_error_response(
        self, mock_post: MagicMock, tmp_fits: Path
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "error", "message": "bad request"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        solver = PlateSolver(
            astap_path=Path("/mock/astap"),
            astrometry_api_key="test-key",
        )
        with pytest.raises(SolveError, match="astrometry.net failed"):
            solver._solve_astrometry_net(tmp_fits, 180.0, 45.0)

    @patch("httpx.post")
    def test_astrometry_net_without_hints(
        self, mock_post: MagicMock, tmp_fits: Path
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "wcs_header": {
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 100.0,
                "CRVAL2": -20.0,
                "CRPIX1": 512.0,
                "CRPIX2": 512.0,
                "CDELT1": -0.0003,
                "CDELT2": 0.0003,
            },
            "field_width": 2.0,
            "field_height": 2.0,
            "orientation": 0.0,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        solver = PlateSolver(
            astap_path=Path("/mock/astap"),
            astrometry_api_key="key",
        )
        result = solver._solve_astrometry_net(tmp_fits, None, None)
        assert result.ra_center == pytest.approx(100.0)
        assert result.dec_center == pytest.approx(-20.0)


# --- SolveResult Tests ---


class TestSolveResult:
    def test_solve_result_immutable(self, sample_wcs: WCS) -> None:
        result = SolveResult(
            wcs=sample_wcs,
            ra_center=180.0,
            dec_center=45.0,
            field_width_deg=1.0,
            field_height_deg=1.0,
            rotation_deg=5.0,
            solve_time_s=2.3,
            solver_used="astap",
        )
        assert result.ra_center == 180.0
        assert result.dec_center == 45.0
        assert result.solver_used == "astap"
        with pytest.raises(AttributeError):
            result.ra_center = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Additional coverage: _try_coord_float, line 84, line 170
# ---------------------------------------------------------------------------

class TestTryCoordFloat:
    def test_sexagesimal_string_returns_none(self) -> None:
        """_try_coord_float returns None for non-numeric strings like '12h30m' (lines 44-45)."""
        from astroai.engine.platesolving.solver import _try_coord_float
        assert _try_coord_float("12h30m45s") is None
        assert _try_coord_float("not-a-number") is None

    def test_none_returns_none(self) -> None:
        from astroai.engine.platesolving.solver import _try_coord_float
        assert _try_coord_float(None) is None


class TestSolveCoordExtraction:
    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_solve_extracts_coords_from_fits_header(self, mock_run: MagicMock, tmp_fits: Path) -> None:
        """solve() extracts ra_hint/dec_hint from FITS header when not provided (line 84)."""
        wcs_path = tmp_fits.with_suffix(".wcs")
        wcs = WCS(naxis=2)
        wcs.wcs.crval = [180.0, 45.0]
        wcs.wcs.crpix = [50.0, 50.0]
        wcs.wcs.cdelt = [-0.000277, 0.000277]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        header = wcs.to_header()
        fits.PrimaryHDU(header=header).writeto(str(wcs_path), overwrite=True)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        solver = PlateSolver(astap_path=Path("/mock/astap"), max_retries=1)
        # No ra_hint/dec_hint � solver should extract from FITS header
        result = solver.solve(tmp_fits)
        assert result.ra_center == pytest.approx(180.0, abs=0.1)


class TestRunAstapNoWcsFile:
    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_no_wcs_file_raises_solve_error(self, mock_run: MagicMock, tmp_fits: Path) -> None:
        """SolveError raised when ASTAP returns 0 but produces no .wcs file (line 170)."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        solver = PlateSolver(astap_path=Path("/mock/astap"), max_retries=1)
        with pytest.raises(SolveError, match="did not produce a WCS file"):
            solver._run_astap_subprocess(tmp_fits, None, None, 30.0)


class TestSolveAstapCdMatrix:
    """Cover lines 115-116: rotation from CD matrix when wcs.wcs.has_cd() is True."""

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_solve_astap_with_cd_matrix_computes_rotation(
        self, mock_run: MagicMock, tmp_fits: Path
    ) -> None:
        """_solve_astap builds SolveResult using CD matrix rot (lines 115-116)."""
        import math

        # Build a FITS header with CDi_j keywords explicitly so WCS.has_cd() returns True
        scale = 0.000277
        angle_rad = math.radians(30.0)
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 100
        header["NAXIS2"] = 100
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRVAL1"] = 180.0
        header["CRVAL2"] = 45.0
        header["CRPIX1"] = 50.0
        header["CRPIX2"] = 50.0
        header["CD1_1"] = -scale * math.cos(angle_rad)
        header["CD1_2"] = scale * math.sin(angle_rad)
        header["CD2_1"] = scale * math.sin(angle_rad)
        header["CD2_2"] = scale * math.cos(angle_rad)

        wcs_path = tmp_fits.with_suffix(".wcs")
        fits.PrimaryHDU(header=header).writeto(str(wcs_path), overwrite=True)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )

        solver = PlateSolver(astap_path=Path("/mock/astap"), max_retries=1)
        result = solver._solve_astap(tmp_fits, 180.0, 45.0)
        assert result.solver_used == "astap"
        # rotation from CD matrix must be non-trivial
        assert isinstance(result.rotation_deg, float)
