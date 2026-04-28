from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroai.astrometry.catalog import WcsSolution
from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
)
from astroai.engine.photometry.models import PhotometryResult, StarMeasurement
from astroai.engine.photometry.pipeline_step import PhotometryStep


def _fake_wcs() -> WcsSolution:
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


def _fake_photometry_result() -> PhotometryResult:
    stars = [
        StarMeasurement(
            star_id=i, ra=180.0 + i * 0.01, dec=45.0 + i * 0.01,
            x_pixel=100.0 + i, y_pixel=200.0 + i,
            instr_mag=-12.0 + i * 0.5,
            catalog_mag=10.0 + i * 0.5,
            cal_mag=10.1 + i * 0.5,
            residual=0.1,
        )
        for i in range(3)
    ]
    return PhotometryResult(stars=stars, r_squared=0.96, n_matched=3)


class TestPhotometryStep:
    def test_name(self) -> None:
        step = PhotometryStep()
        assert step.name == "Photometry"

    def test_stage(self) -> None:
        step = PhotometryStep()
        assert step.stage == PipelineStage.PHOTOMETRY

    def test_execute_stores_result_in_metadata(self) -> None:
        step = PhotometryStep()
        ctx = PipelineContext(
            result=np.zeros((64, 64), dtype=np.float32),
            metadata={"wcs_solution": _fake_wcs()},
        )
        expected = _fake_photometry_result()

        with patch.object(step._engine, "run", return_value=expected):
            result_ctx = step.execute(ctx)

        assert "photometry_result" in result_ctx.metadata
        assert result_ctx.metadata["photometry_result"].n_matched == 3

    def test_execute_uses_wcs_key(self) -> None:
        step = PhotometryStep()
        ctx = PipelineContext(
            result=np.zeros((64, 64), dtype=np.float32),
            metadata={"wcs": _fake_wcs()},
        )
        expected = _fake_photometry_result()

        with patch.object(step._engine, "run", return_value=expected):
            result_ctx = step.execute(ctx)

        assert "photometry_result" in result_ctx.metadata

    def test_execute_without_wcs_returns_unchanged(self) -> None:
        step = PhotometryStep(fail_silently=True)
        ctx = PipelineContext(
            result=np.zeros((64, 64), dtype=np.float32),
        )

        result_ctx = step.execute(ctx)

        assert "photometry_result" not in result_ctx.metadata

    def test_execute_without_wcs_raises_when_not_silent(self) -> None:
        step = PhotometryStep(fail_silently=False)
        ctx = PipelineContext(
            result=np.zeros((64, 64), dtype=np.float32),
        )

        with pytest.raises(RuntimeError, match="no WCS"):
            step.execute(ctx)

    def test_execute_uses_first_image_when_no_result(self) -> None:
        step = PhotometryStep()
        ctx = PipelineContext(
            images=[np.zeros((64, 64), dtype=np.float32)],
            metadata={"wcs_solution": _fake_wcs()},
        )
        expected = _fake_photometry_result()

        with patch.object(step._engine, "run", return_value=expected):
            result_ctx = step.execute(ctx)

        assert "photometry_result" in result_ctx.metadata

    def test_execute_no_image_fail_silently(self) -> None:
        step = PhotometryStep(fail_silently=True)
        ctx = PipelineContext(
            metadata={"wcs_solution": _fake_wcs()},
        )

        result_ctx = step.execute(ctx)
        assert "photometry_result" not in result_ctx.metadata

    def test_execute_engine_error_fail_silently(self) -> None:
        step = PhotometryStep(fail_silently=True)
        ctx = PipelineContext(
            result=np.zeros((64, 64), dtype=np.float32),
            metadata={"wcs_solution": _fake_wcs()},
        )

        with patch.object(step._engine, "run", side_effect=ValueError("engine error")):
            result_ctx = step.execute(ctx)

        assert "photometry_result" not in result_ctx.metadata

    def test_execute_engine_error_raises_when_not_silent(self) -> None:
        step = PhotometryStep(fail_silently=False)
        ctx = PipelineContext(
            result=np.zeros((64, 64), dtype=np.float32),
            metadata={"wcs_solution": _fake_wcs()},
        )

        with patch.object(step._engine, "run", side_effect=ValueError("engine error")):
            with pytest.raises(ValueError, match="engine error"):
                step.execute(ctx)

    def test_progress_callback(self) -> None:
        step = PhotometryStep()
        ctx = PipelineContext(
            result=np.zeros((64, 64), dtype=np.float32),
            metadata={"wcs_solution": _fake_wcs()},
        )
        events: list[PipelineProgress] = []

        with patch.object(step._engine, "run", return_value=_fake_photometry_result()):
            step.execute(ctx, progress=events.append)

        assert len(events) == 2
        assert events[0].stage == PipelineStage.PHOTOMETRY
        assert events[1].stage == PipelineStage.PHOTOMETRY

    def test_execute_no_image_raises_when_not_silent(self) -> None:
        """fail_silently=False with no image raises RuntimeError (line 65)."""
        step = PhotometryStep(fail_silently=False)
        ctx = PipelineContext(metadata={"wcs_solution": _fake_wcs()})
        with pytest.raises(RuntimeError, match="no image"):
            step.execute(ctx)
