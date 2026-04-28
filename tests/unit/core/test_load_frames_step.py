"""Unit tests for LoadFramesStep."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.core.pipeline.load_frames_step import LoadFramesStep


def _write_fits(path: Path, shape: tuple[int, int] = (16, 16)) -> None:
    from astropy.io import fits
    data = np.zeros(shape, dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(str(path), overwrite=True)


class TestLoadFramesStepBasic:
    def test_name(self) -> None:
        step = LoadFramesStep([])
        assert step.name == "Frame-Laden"

    def test_stage_is_calibration(self) -> None:
        step = LoadFramesStep([])
        assert step.stage == PipelineStage.CALIBRATION

    def test_empty_paths_passthrough(self) -> None:
        step = LoadFramesStep([])
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.images == []

    def test_loads_fits_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "frame.fits"
            _write_fits(p, shape=(8, 8))
            step = LoadFramesStep([p])
            ctx = PipelineContext()
            out = step.execute(ctx)
            assert len(out.images) == 1
            assert out.images[0].shape == (8, 8)

    def test_multiple_frames_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                p = Path(tmpdir) / f"frame_{i}.fits"
                _write_fits(p, shape=(4, 4))
                paths.append(p)
            step = LoadFramesStep(paths)
            ctx = PipelineContext()
            out = step.execute(ctx)
            assert len(out.images) == 3

    def test_metadata_paths_stored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "f.fits"
            _write_fits(p)
            step = LoadFramesStep([p])
            out = step.execute(PipelineContext())
            assert "loaded_frame_paths" in out.metadata
            assert len(out.metadata["loaded_frame_paths"]) == 1

    def test_output_dtype_float32(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "f.fits"
            _write_fits(p)
            step = LoadFramesStep([p])
            out = step.execute(PipelineContext())
            assert out.images[0].dtype == np.float32

    def test_progress_called(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "f.fits"
            _write_fits(p)
            step = LoadFramesStep([p])
            calls: list[object] = []
            step.execute(PipelineContext(), progress=lambda x: calls.append(x))
            assert len(calls) >= 2

    def test_accepts_string_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "f.fits"
            _write_fits(p)
            step = LoadFramesStep([str(p)])
            out = step.execute(PipelineContext())
            assert len(out.images) == 1

    def test_nonexistent_file_raises(self) -> None:
        step = LoadFramesStep([Path("/nonexistent/path.fits")])
        with pytest.raises(Exception):
            step.execute(PipelineContext())

    def test_replaces_existing_images_in_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "f.fits"
            _write_fits(p, shape=(4, 4))
            step = LoadFramesStep([p])
            ctx = PipelineContext(images=[np.zeros((8, 8), dtype=np.float32)])
            out = step.execute(ctx)
            assert len(out.images) == 1
            assert out.images[0].shape == (4, 4)
