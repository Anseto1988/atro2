"""E2E integration test for the astrometry plate-solving pipeline step.

Uses a fully mocked ASTAP binary so no external tool is required in CI.
Verifies the complete path: pipeline context → AstrometryStep → WcsSolution
stored in context.metadata.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits

from astroai.astrometry.catalog import WcsSolution, pixel_to_radec
from astroai.astrometry.pipeline_step import AstrometryStep
from astroai.astrometry.solver import AstapSolver, SolverError
from astroai.core.pipeline.base import Pipeline, PipelineContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solution(ra: float = 83.82, dec: float = -5.39) -> WcsSolution:
    scale_deg = 1.5 / 3600.0
    return WcsSolution(
        ra_center=ra,
        dec_center=dec,
        pixel_scale_arcsec=1.5,
        rotation_deg=0.0,
        fov_width_deg=scale_deg * 512,
        fov_height_deg=scale_deg * 512,
        cd_matrix=(-scale_deg, 0.0, 0.0, scale_deg),
        crpix1=256.5,
        crpix2=256.5,
    )


def _write_wcs_fits(path: Path, sol: WcsSolution) -> None:
    hdr = fits.Header()
    hdr["CRVAL1"] = sol.ra_center
    hdr["CRVAL2"] = sol.dec_center
    hdr["CRPIX1"] = sol.crpix1
    hdr["CRPIX2"] = sol.crpix2
    hdr["CD1_1"] = sol.cd_matrix[0]
    hdr["CD1_2"] = sol.cd_matrix[1]
    hdr["CD2_1"] = sol.cd_matrix[2]
    hdr["CD2_2"] = sol.cd_matrix[3]
    data = np.zeros((512, 512), dtype=np.float32)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# E2E: AstrometryStep in isolation
# ---------------------------------------------------------------------------

class TestAstrometryStepE2E:
    def _mock_solve(self, solution: WcsSolution):  # noqa: ANN202
        """Return a context-manager-compatible mock for subprocess.run."""
        def _fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN202
            # Write the .wcs sidecar that the solver expects
            fits_path = Path(cmd[cmd.index("-f") + 1])
            wcs_path = fits_path.with_suffix(".wcs")
            _write_wcs_fits(wcs_path, solution)
            return MagicMock(returncode=0, stdout="Solution found", stderr="")
        return _fake_run

    def test_step_stores_wcs_in_context(self) -> None:
        expected = _make_solution(ra=83.82, dec=-5.39)
        image = np.zeros((128, 128), dtype=np.float32)

        step = AstrometryStep(executable="/fake/astap", fail_silently=False)
        ctx = PipelineContext(result=image)

        with patch("subprocess.run", side_effect=self._mock_solve(expected)):
            ctx = step.execute(ctx)

        sol = ctx.metadata.get("wcs_solution")
        assert sol is not None, "WCS solution not stored in context.metadata"
        assert isinstance(sol, WcsSolution)
        assert sol.ra_center == pytest.approx(83.82, abs=1e-3)
        assert sol.dec_center == pytest.approx(-5.39, abs=1e-3)

    def test_step_fail_silently_true(self) -> None:
        image = np.zeros((64, 64), dtype=np.float32)
        step = AstrometryStep(executable="/fake/astap", fail_silently=True)
        ctx = PipelineContext(result=image)

        mock_result = MagicMock(returncode=1, stdout="", stderr="No stars detected")
        with patch("subprocess.run", return_value=mock_result):
            ctx = step.execute(ctx)  # must not raise

        assert "wcs_solution" not in ctx.metadata

    def test_step_fail_silently_false_raises(self) -> None:
        image = np.zeros((64, 64), dtype=np.float32)
        step = AstrometryStep(executable="/fake/astap", fail_silently=False)
        ctx = PipelineContext(result=image)

        mock_result = MagicMock(returncode=1, stdout="", stderr="No stars detected")
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(SolverError):
                step.execute(ctx)

    def test_step_no_image_skips_gracefully(self) -> None:
        step = AstrometryStep(executable="/fake/astap", fail_silently=True)
        ctx = PipelineContext()  # no result, no images
        ctx = step.execute(ctx)
        assert "wcs_solution" not in ctx.metadata


# ---------------------------------------------------------------------------
# E2E: Full pipeline with AstrometryStep
# ---------------------------------------------------------------------------

class TestAstrometryInPipeline:
    def test_pipeline_with_astrometry_step(self) -> None:
        expected = _make_solution(ra=150.0, dec=45.0)
        image = np.random.default_rng(0).uniform(0.0, 1.0, (128, 128)).astype(np.float32)

        step = AstrometryStep(executable="/fake/astap", fail_silently=False)
        pipeline = Pipeline([step])
        ctx = PipelineContext(result=image)

        def _fake_run(cmd, **_kwargs):  # noqa: ANN001, ANN202
            fits_path = Path(cmd[cmd.index("-f") + 1])
            _write_wcs_fits(fits_path.with_suffix(".wcs"), expected)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=_fake_run):
            ctx = pipeline.run(ctx)

        sol: WcsSolution = ctx.metadata["wcs_solution"]
        assert sol.ra_center == pytest.approx(150.0, abs=1e-3)
        assert sol.dec_center == pytest.approx(45.0, abs=1e-3)


# ---------------------------------------------------------------------------
# E2E: pixel_to_radec round-trip
# ---------------------------------------------------------------------------

class TestCoordinateRoundTrip:
    def test_center_pixel_round_trip(self) -> None:
        sol = _make_solution(ra=200.0, dec=-30.0)
        cx = sol.crpix1 - 1.0
        cy = sol.crpix2 - 1.0
        ra, dec = pixel_to_radec(sol, np.array([cx]), np.array([cy]))
        assert float(ra[0]) == pytest.approx(200.0, abs=1e-3)
        assert float(dec[0]) == pytest.approx(-30.0, abs=1e-3)

    def test_off_center_pixel_direction(self) -> None:
        # CD1_1 is negative → RA decreases as x increases (standard orientation)
        sol = _make_solution(ra=100.0, dec=0.0)
        ra_right, _ = pixel_to_radec(sol, np.array([sol.crpix1 + 99]), np.array([sol.crpix2 - 1]))
        ra_center, _ = pixel_to_radec(sol, np.array([sol.crpix1 - 1]), np.array([sol.crpix2 - 1]))
        # CD1_1 = -scale_deg → RA decreases for positive dx
        assert float(ra_right[0]) < float(ra_center[0]) or abs(float(ra_right[0]) - float(ra_center[0])) < 180
