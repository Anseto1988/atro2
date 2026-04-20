"""Unit tests for AstrometryStep pipeline integration."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from astroai.astrometry.catalog import WcsSolution
from astroai.astrometry.pipeline_step import AstrometryStep
from astroai.astrometry.solver import SolverError
from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
)


def _fake_solution() -> WcsSolution:
    return WcsSolution(
        ra_center=180.0,
        dec_center=45.0,
        pixel_scale_arcsec=1.5,
        rotation_deg=0.0,
        fov_width_deg=0.2,
        fov_height_deg=0.2,
        cd_matrix=(-4.17e-4, 0.0, 0.0, 4.17e-4),
        crpix1=256.0,
        crpix2=256.0,
    )


class TestAstrometryStep:
    def test_stage_is_astrometry(self) -> None:
        step = AstrometryStep(executable="/fake/astap")
        assert step.stage == PipelineStage.ASTROMETRY

    def test_name(self) -> None:
        step = AstrometryStep(executable="/fake/astap")
        assert "Plate Solving" in step.name

    def test_execute_stores_wcs_in_metadata(self) -> None:
        step = AstrometryStep(executable="/fake/astap")
        ctx = PipelineContext(
            result=np.zeros((64, 64), dtype=np.float32),
        )
        expected = _fake_solution()

        with patch.object(step._solver, "solve_array", return_value=expected):
            result = step.execute(ctx)

        assert "wcs_solution" in result.metadata
        assert result.metadata["wcs_solution"].ra_center == 180.0

    def test_execute_uses_first_image_when_no_result(self) -> None:
        step = AstrometryStep(executable="/fake/astap")
        ctx = PipelineContext(
            images=[np.zeros((32, 32), dtype=np.float32)],
        )
        expected = _fake_solution()

        with patch.object(step._solver, "solve_array", return_value=expected):
            result = step.execute(ctx)

        assert result.metadata["wcs_solution"].dec_center == 45.0

    def test_execute_skips_when_no_image(self) -> None:
        step = AstrometryStep(executable="/fake/astap")
        ctx = PipelineContext()

        result = step.execute(ctx)
        assert "wcs_solution" not in result.metadata

    def test_fail_silently_true_catches_solver_error(self) -> None:
        step = AstrometryStep(executable="/fake/astap", fail_silently=True)
        ctx = PipelineContext(result=np.zeros((64, 64), dtype=np.float32))

        with patch.object(
            step._solver, "solve_array", side_effect=SolverError("no solution")
        ):
            result = step.execute(ctx)

        assert "wcs_solution" not in result.metadata

    def test_fail_silently_false_raises(self) -> None:
        step = AstrometryStep(executable="/fake/astap", fail_silently=False)
        ctx = PipelineContext(result=np.zeros((64, 64), dtype=np.float32))

        with patch.object(
            step._solver, "solve_array", side_effect=SolverError("no solution")
        ):
            with pytest.raises(SolverError):
                step.execute(ctx)

    def test_progress_callback_called(self) -> None:
        step = AstrometryStep(executable="/fake/astap")
        ctx = PipelineContext(result=np.zeros((64, 64), dtype=np.float32))
        events: list[PipelineProgress] = []

        with patch.object(step._solver, "solve_array", return_value=_fake_solution()):
            step.execute(ctx, progress=events.append)

        assert len(events) == 2
        assert events[0].stage == PipelineStage.ASTROMETRY
        assert events[0].message == "Running plate solver…"
        assert events[1].message == "Plate solving complete"

    def test_pixel_size_and_focal_length_passed(self) -> None:
        step = AstrometryStep(executable="/fake/astap")
        ctx = PipelineContext(
            result=np.zeros((64, 64), dtype=np.float32),
            metadata={"pixel_size_um": 3.76, "focal_length_mm": 600},
        )

        call_args: list[dict[str, object]] = []

        def _capture_solve(image: object, fits_header: object = None) -> WcsSolution:
            call_args.append({"fits_header": fits_header})
            return _fake_solution()

        with patch.object(step._solver, "solve_array", side_effect=_capture_solve):
            step.execute(ctx)

        assert call_args[0]["fits_header"] is not None
        assert "SCALE" in call_args[0]["fits_header"]  # type: ignore[operator]
