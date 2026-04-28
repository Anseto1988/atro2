"""Unit tests for PlateSolvingStep pipeline integration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.core.pipeline.platesolving_step import PlateSolvingStep


@pytest.fixture()
def sample_context() -> PipelineContext:
    image = np.ones((100, 100), dtype=np.float32) * 500.0
    ctx = PipelineContext(images=[image])
    ctx.result = image
    return ctx


@pytest.fixture()
def step() -> PlateSolvingStep:
    return PlateSolvingStep(
        astap_path=Path("/mock/astap"),
        search_radius_deg=10.0,
        max_retries=1,
        timeout_s=10.0,
        fail_silently=True,
    )


def _make_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [180.0, 45.0]
    wcs.wcs.crpix = [50.0, 50.0]
    wcs.wcs.cdelt = [-0.000277, 0.000277]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


class TestPlateSolvingStep:
    def test_step_properties(self, step: PlateSolvingStep) -> None:
        assert step.name == "Plate Solving"
        assert step.stage == PipelineStage.ASTROMETRY

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_successful_solve_stores_result(
        self,
        mock_run: MagicMock,
        step: PlateSolvingStep,
        sample_context: PipelineContext,
        tmp_path: Path,
    ) -> None:
        wcs = _make_wcs()
        wcs_header = wcs.to_header()
        hdu = fits.PrimaryHDU(header=wcs_header)

        def side_effect(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            fits_arg = cmd[cmd.index("-f") + 1]
            wcs_path = Path(fits_arg).with_suffix(".wcs")
            hdu.writeto(str(wcs_path), overwrite=True)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = side_effect

        result_ctx = step.execute(sample_context)

        assert "solve_result" in result_ctx.metadata
        assert "wcs" in result_ctx.metadata
        solve_result = result_ctx.metadata["solve_result"]
        assert solve_result.solver_used == "astap"
        assert solve_result.ra_center == pytest.approx(180.0)
        assert solve_result.dec_center == pytest.approx(45.0)

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_fail_silently_skips_on_error(
        self,
        mock_run: MagicMock,
        step: PlateSolvingStep,
        sample_context: PipelineContext,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="No solution"
        )
        result_ctx = step.execute(sample_context)
        assert "solve_result" not in result_ctx.metadata

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_fail_raises_when_not_silent(
        self,
        mock_run: MagicMock,
        sample_context: PipelineContext,
    ) -> None:
        step = PlateSolvingStep(
            astap_path=Path("/mock/astap"),
            max_retries=1,
            fail_silently=False,
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="No solution"
        )
        from astroai.engine.platesolving.solver import SolveError

        with pytest.raises(SolveError):
            step.execute(sample_context)

    def test_no_image_skips(self, step: PlateSolvingStep) -> None:
        ctx = PipelineContext()
        result = step.execute(ctx)
        assert "solve_result" not in result.metadata

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_writes_wcs_to_fits_output(
        self,
        mock_run: MagicMock,
        step: PlateSolvingStep,
        sample_context: PipelineContext,
        tmp_path: Path,
    ) -> None:
        wcs = _make_wcs()
        wcs_header = wcs.to_header()
        hdu = fits.PrimaryHDU(header=wcs_header)

        def side_effect(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            fits_arg = cmd[cmd.index("-f") + 1]
            wcs_path = Path(fits_arg).with_suffix(".wcs")
            hdu.writeto(str(wcs_path), overwrite=True)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = side_effect

        output_fits = tmp_path / "output.fits"
        data = np.ones((100, 100), dtype=np.float32)
        fits.PrimaryHDU(data=data).writeto(str(output_fits), overwrite=True)
        sample_context.metadata["export_path"] = str(output_fits)

        step.execute(sample_context)

        with fits.open(str(output_fits)) as opened:
            header = opened[0].header
            assert "CTYPE1" in header
            assert header["CTYPE1"] == "RA---TAN"

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_uses_ra_dec_hints_from_metadata(
        self,
        mock_run: MagicMock,
        step: PlateSolvingStep,
        sample_context: PipelineContext,
    ) -> None:
        wcs = _make_wcs()
        wcs_header = wcs.to_header()
        hdu = fits.PrimaryHDU(header=wcs_header)

        captured_cmds: list[list[str]] = []

        def side_effect(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured_cmds.append(cmd)
            fits_arg = cmd[cmd.index("-f") + 1]
            wcs_path = Path(fits_arg).with_suffix(".wcs")
            hdu.writeto(str(wcs_path), overwrite=True)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = side_effect

        sample_context.metadata["ra_hint"] = 100.0
        sample_context.metadata["dec_hint"] = -20.0

        step.execute(sample_context)

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        assert "-ra" in cmd
        ra_idx = cmd.index("-ra")
        assert float(cmd[ra_idx + 1]) == pytest.approx(100.0)


class TestPlateSolvingStepImport:
    def test_importable_from_pipeline_package(self) -> None:
        from astroai.core.pipeline import PlateSolvingStep as PS
        assert PS is PlateSolvingStep


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------

class TestPlateSolvingStepCoverageGaps:
    """Tests targeting uncovered lines 93, 143, 147-152."""

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_uses_first_image_when_result_is_none(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Line 93: image = context.images[0] when context.result is None."""
        step = PlateSolvingStep(
            astap_path=Path("/mock/astap"),
            max_retries=1,
            timeout_s=10.0,
            fail_silently=True,
        )
        image = np.ones((100, 100), dtype=np.float32) * 500.0
        ctx = PipelineContext(images=[image])
        # result is None by default, images has one entry
        assert ctx.result is None

        wcs = _make_wcs()
        hdu = fits.PrimaryHDU(header=wcs.to_header())

        def side_effect(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            fits_arg = cmd[cmd.index("-f") + 1]
            wcs_path = Path(fits_arg).with_suffix(".wcs")
            hdu.writeto(str(wcs_path), overwrite=True)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = side_effect
        result_ctx = step.execute(ctx)
        assert "solve_result" in result_ctx.metadata

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_3d_image_first_channel_used(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Line 143: arr = arr[..., 0] when input is 3D (H, W, C)."""
        step = PlateSolvingStep(
            astap_path=Path("/mock/astap"),
            max_retries=1,
            timeout_s=10.0,
            fail_silently=True,
        )
        image_3d = np.ones((100, 100, 3), dtype=np.float32) * 500.0
        ctx = PipelineContext(images=[image_3d])
        ctx.result = image_3d

        wcs = _make_wcs()
        hdu = fits.PrimaryHDU(header=wcs.to_header())

        def side_effect(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            fits_arg = cmd[cmd.index("-f") + 1]
            wcs_path = Path(fits_arg).with_suffix(".wcs")
            hdu.writeto(str(wcs_path), overwrite=True)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = side_effect
        result_ctx = step.execute(ctx)
        assert "solve_result" in result_ctx.metadata

    @patch("astroai.engine.platesolving.solver.subprocess.run")
    def test_pixel_scale_computed_from_metadata(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Lines 147-152: SCALE header set when pixel_size_um and focal_length_mm present."""
        step = PlateSolvingStep(
            astap_path=Path("/mock/astap"),
            max_retries=1,
            timeout_s=10.0,
            fail_silently=True,
        )
        image = np.ones((100, 100), dtype=np.float32) * 500.0
        ctx = PipelineContext(images=[image])
        ctx.result = image
        ctx.metadata["pixel_size_um"] = 3.76
        ctx.metadata["focal_length_mm"] = 500.0

        wcs = _make_wcs()
        hdu = fits.PrimaryHDU(header=wcs.to_header())

        def side_effect(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            fits_arg = cmd[cmd.index("-f") + 1]
            wcs_path = Path(fits_arg).with_suffix(".wcs")
            hdu.writeto(str(wcs_path), overwrite=True)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = side_effect
        result_ctx = step.execute(ctx)
        # Should have solved without error; scale = 206.265 * 3.76 / 500.0 ≈ 1.55
        assert "solve_result" in result_ctx.metadata
