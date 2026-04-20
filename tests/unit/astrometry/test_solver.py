"""Unit tests for astroai.astrometry.solver and catalog."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits

from astroai.astrometry.catalog import (
    AstapCatalog,
    CatalogManager,
    WcsSolution,
    pixel_to_radec,
)
from astroai.astrometry.solver import AstapSolver, SolverError
from astroai.engine.platesolving.solver import PlateSolver, SolveError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_wcs_solution(
    ra: float = 83.82,
    dec: float = -5.39,
    scale_arcsec: float = 1.5,
) -> WcsSolution:
    scale_deg = scale_arcsec / 3600.0
    return WcsSolution(
        ra_center=ra,
        dec_center=dec,
        pixel_scale_arcsec=scale_arcsec,
        rotation_deg=0.0,
        fov_width_deg=scale_deg * 512,
        fov_height_deg=scale_deg * 512,
        cd_matrix=(-scale_deg, 0.0, 0.0, scale_deg),
        crpix1=256.5,
        crpix2=256.5,
    )


def _write_wcs_fits(path: Path, solution: WcsSolution) -> None:
    hdr = fits.Header()
    hdr["CRVAL1"] = solution.ra_center
    hdr["CRVAL2"] = solution.dec_center
    hdr["CRPIX1"] = solution.crpix1
    hdr["CRPIX2"] = solution.crpix2
    hdr["CD1_1"] = solution.cd_matrix[0]
    hdr["CD1_2"] = solution.cd_matrix[1]
    hdr["CD2_1"] = solution.cd_matrix[2]
    hdr["CD2_2"] = solution.cd_matrix[3]
    data = np.zeros((512, 512), dtype=np.float32)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# WcsSolution tests
# ---------------------------------------------------------------------------

class TestWcsSolution:
    def test_pixel_scale_deg(self) -> None:
        sol = _make_wcs_solution(scale_arcsec=3.6)
        assert sol.pixel_scale_deg == pytest.approx(0.001, abs=1e-9)

    def test_frozen(self) -> None:
        sol = _make_wcs_solution()
        with pytest.raises(Exception):
            sol.ra_center = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# pixel_to_radec tests
# ---------------------------------------------------------------------------

class TestPixelToRadec:
    def test_center_pixel_returns_center_coords(self) -> None:
        sol = _make_wcs_solution(ra=100.0, dec=30.0, scale_arcsec=1.0)
        # crpix is 1-indexed; center pixel in 0-indexed coords:
        cx = sol.crpix1 - 1.0
        cy = sol.crpix2 - 1.0
        ra, dec = pixel_to_radec(sol, np.array([cx]), np.array([cy]))
        assert float(ra[0]) == pytest.approx(100.0, abs=1e-4)
        assert float(dec[0]) == pytest.approx(30.0, abs=1e-4)

    def test_ra_wraps_360(self) -> None:
        sol = _make_wcs_solution(ra=359.9, dec=0.0, scale_arcsec=1.0)
        ra, _ = pixel_to_radec(sol, np.array([1000.0]), np.array([0.0]))
        assert 0.0 <= float(ra[0]) < 360.0


# ---------------------------------------------------------------------------
# AstapSolver — _parse_wcs tests (no subprocess needed)
# ---------------------------------------------------------------------------

class TestAstapSolverParseWcs:
    def test_parse_wcs_file(self) -> None:
        sol_expected = _make_wcs_solution(ra=83.82, dec=-5.39)
        with tempfile.NamedTemporaryFile(suffix=".wcs", delete=False) as f:
            wcs_path = Path(f.name)
        try:
            _write_wcs_fits(wcs_path, sol_expected)
            sol = AstapSolver._parse_wcs(wcs_path)
            assert sol.ra_center == pytest.approx(83.82, abs=1e-4)
            assert sol.dec_center == pytest.approx(-5.39, abs=1e-4)
            assert sol.pixel_scale_arcsec == pytest.approx(1.5, abs=0.01)
        finally:
            wcs_path.unlink(missing_ok=True)

    def test_parse_wcs_computes_pixel_scale_from_cd(self) -> None:
        sol_in = _make_wcs_solution(scale_arcsec=2.0)
        with tempfile.NamedTemporaryFile(suffix=".wcs", delete=False) as f:
            wcs_path = Path(f.name)
        try:
            _write_wcs_fits(wcs_path, sol_in)
            sol = AstapSolver._parse_wcs(wcs_path)
            assert sol.pixel_scale_arcsec == pytest.approx(2.0, abs=0.01)
        finally:
            wcs_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AstapSolver.solve — subprocess mocking
# ---------------------------------------------------------------------------

class TestAstapSolverSolve:
    def _make_solver(self) -> AstapSolver:
        return AstapSolver(executable="/fake/astap", search_radius_deg=30.0)

    def test_solve_raises_file_not_found(self) -> None:
        solver = self._make_solver()
        with pytest.raises(FileNotFoundError):
            solver.solve(Path("/nonexistent/file.fits"))

    def test_solve_raises_on_nonzero_exit(self, tmp_path: Path) -> None:
        fits_path = tmp_path / "test.fits"
        fits.PrimaryHDU(data=np.zeros((64, 64), dtype=np.float32)).writeto(fits_path)

        solver = self._make_solver()
        mock_result = MagicMock(returncode=1, stdout="", stderr="No solution found")
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(SolverError, match="failed after"):
                solver.solve(fits_path)

    def test_solve_success_with_mock(self, tmp_path: Path) -> None:
        fits_path = tmp_path / "test.fits"
        fits.PrimaryHDU(data=np.zeros((64, 64), dtype=np.float32)).writeto(fits_path)

        expected = _make_wcs_solution(ra=10.0, dec=20.0)
        wcs_path = fits_path.with_suffix(".wcs")
        _write_wcs_fits(wcs_path, expected)

        solver = self._make_solver()
        mock_result = MagicMock(returncode=0, stdout="Solution found", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            sol = solver.solve(fits_path)

        assert sol.ra_center == pytest.approx(10.0, abs=1e-4)
        assert sol.dec_center == pytest.approx(20.0, abs=1e-4)

    def test_solve_timeout_raises_solver_error(self, tmp_path: Path) -> None:
        import subprocess
        fits_path = tmp_path / "test.fits"
        fits.PrimaryHDU(data=np.zeros((64, 64), dtype=np.float32)).writeto(fits_path)

        solver = self._make_solver()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("astap", 1)):
            with pytest.raises(SolverError, match="timed out"):
                solver.solve(fits_path)


# ---------------------------------------------------------------------------
# AstapSolver.solve_array
# ---------------------------------------------------------------------------

class TestAstapSolverSolveArray:
    def test_solve_array_with_mock(self) -> None:
        image = np.zeros((128, 128), dtype=np.float32)
        expected = _make_wcs_solution(ra=50.0, dec=-10.0)

        def _fake_solve(fits_path: Path) -> WcsSolution:
            return expected

        solver = AstapSolver(executable="/fake/astap")
        with patch.object(solver, "solve", side_effect=_fake_solve):
            sol = solver.solve_array(image)

        assert sol.ra_center == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# AstapSolver — retry with expanded radius
# ---------------------------------------------------------------------------

class TestPlateSolverRetry:
    """Tests for PlateSolver retry logic with expanding search radius."""

    def test_retry_expands_radius(self, tmp_path: Path) -> None:
        fits_path = tmp_path / "test.fits"
        fits.PrimaryHDU(data=np.zeros((64, 64), dtype=np.float32)).writeto(fits_path)

        expected = _make_wcs_solution(ra=10.0, dec=20.0)
        wcs_path = fits_path.with_suffix(".wcs")

        call_count = 0
        radii_seen: list[float] = []

        def _fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            r_idx = cmd.index("-r") + 1
            radii_seen.append(float(cmd[r_idx]))

            if call_count < 3:
                return MagicMock(returncode=1, stdout="", stderr="No solution")
            _write_wcs_fits(wcs_path, expected)
            return MagicMock(returncode=0, stdout="OK", stderr="")

        solver = PlateSolver(
            astap_path=Path("/fake/astap"), search_radius_deg=10.0, max_retries=3
        )
        with patch("subprocess.run", side_effect=_fake_run):
            result = solver.solve(fits_path)

        assert call_count == 3
        assert radii_seen[0] == pytest.approx(10.0)
        assert radii_seen[1] == pytest.approx(15.0)  # PlateSolver uses 1.5x multiplier
        assert radii_seen[2] == pytest.approx(22.5)
        assert result.ra_center == pytest.approx(10.0, abs=0.01)

    def test_all_retries_exhausted_raises(self, tmp_path: Path) -> None:
        fits_path = tmp_path / "test.fits"
        fits.PrimaryHDU(data=np.zeros((64, 64), dtype=np.float32)).writeto(fits_path)

        mock_result = MagicMock(returncode=1, stdout="", stderr="No solution")
        solver = PlateSolver(astap_path=Path("/fake/astap"), max_retries=2)

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(SolveError, match="failed after 2 attempts"):
                solver.solve(fits_path)

    def test_timeout_triggers_retry(self, tmp_path: Path) -> None:
        import subprocess as sp

        fits_path = tmp_path / "test.fits"
        fits.PrimaryHDU(data=np.zeros((64, 64), dtype=np.float32)).writeto(fits_path)

        expected = _make_wcs_solution(ra=99.0, dec=10.0)
        wcs_path = fits_path.with_suffix(".wcs")
        call_count = 0

        def _fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise sp.TimeoutExpired(cmd[0], timeout=120)
            _write_wcs_fits(wcs_path, expected)
            return MagicMock(returncode=0, stdout="OK", stderr="")

        solver = PlateSolver(astap_path=Path("/fake/astap"), max_retries=3)
        with patch("subprocess.run", side_effect=_fake_run):
            result = solver.solve(fits_path)

        assert call_count == 2
        assert result.ra_center == pytest.approx(99.0, abs=0.01)


# ---------------------------------------------------------------------------
# CatalogManager tests
# ---------------------------------------------------------------------------

class TestCatalogManager:
    def test_recommend_catalog_wide_fov(self) -> None:
        mgr = CatalogManager(catalog_dir=Path("/tmp/cats"))
        assert mgr.recommend_catalog(5.0) == AstapCatalog.H18

    def test_recommend_catalog_narrow_fov(self) -> None:
        mgr = CatalogManager(catalog_dir=Path("/tmp/cats"))
        assert mgr.recommend_catalog(1.0) == AstapCatalog.D50

    def test_is_installed_false(self, tmp_path: Path) -> None:
        mgr = CatalogManager(catalog_dir=tmp_path)
        assert mgr.is_installed(AstapCatalog.H18) is False

    def test_is_installed_true(self, tmp_path: Path) -> None:
        (tmp_path / "h18_001.290").write_bytes(b"data")
        mgr = CatalogManager(catalog_dir=tmp_path)
        assert mgr.is_installed(AstapCatalog.H18) is True

    def test_ensure_available_raises_when_missing(self, tmp_path: Path) -> None:
        mgr = CatalogManager(catalog_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="H18"):
            mgr.ensure_available(AstapCatalog.H18)

    def test_ensure_available_returns_path(self, tmp_path: Path) -> None:
        (tmp_path / "d50_001.290").write_bytes(b"data")
        mgr = CatalogManager(catalog_dir=tmp_path)
        result = mgr.ensure_available(AstapCatalog.D50)
        assert result == tmp_path

    def test_download_url(self) -> None:
        mgr = CatalogManager()
        url = mgr.download_url(AstapCatalog.H18)
        assert "h18" in url.lower()
        assert "sourceforge" in url
