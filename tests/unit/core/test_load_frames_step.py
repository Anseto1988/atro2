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

    def test_fits_with_no_data_raises(self) -> None:
        from astropy.io import fits
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "empty.fits"
            hdu = fits.PrimaryHDU()  # no data array
            hdu.writeto(str(p), overwrite=True)
            step = LoadFramesStep([p])
            with pytest.raises(ValueError, match="No image data"):
                step.execute(PipelineContext())

    def test_loads_png_via_pil(self) -> None:
        from PIL import Image as PILImage
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "frame.png"
            PILImage.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(str(p))
            step = LoadFramesStep([p])
            out = step.execute(PipelineContext())
            assert len(out.images) == 1
            assert out.images[0].dtype == np.float32


class TestLoadFramesStepRaw:
    """Tests for RAW-format support in _load_frame / LoadFramesStep."""

    def _mock_read_raw(self, monkeypatch, shape=(10, 12, 3), fill=0.5):
        from unittest.mock import MagicMock
        from astroai.core.pipeline import load_frames_step as mod

        fake_rgb = np.full(shape, fill, dtype=np.float32)
        fake_meta = MagicMock()
        fake_meta.width = shape[1]
        fake_meta.height = shape[0]

        monkeypatch.setattr(
            mod,
            "_load_frame",
            lambda path: np.mean(fake_rgb, axis=2).astype(np.float32),
        )
        return fake_rgb

    def test_raw_cr2_loads_as_float32(self, monkeypatch) -> None:
        from astroai.core.pipeline.load_frames_step import _load_frame
        from unittest.mock import patch, MagicMock
        from astroai.core.io.raw_io import RAW_EXTENSIONS

        fake_rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
        fake_meta = MagicMock()
        fake_meta.width = 8
        fake_meta.height = 8

        with patch("astroai.core.io.raw_io.read_raw", return_value=(fake_rgb, fake_meta)):
            result = _load_frame(Path("/fake/img.cr2"))

        assert result.dtype == np.float32
        assert result.shape == (8, 8)

    def test_raw_nef_loads_as_float32(self, monkeypatch) -> None:
        from astroai.core.pipeline.load_frames_step import _load_frame
        from unittest.mock import patch, MagicMock

        fake_rgb = np.full((6, 10, 3), 0.25, dtype=np.float32)
        fake_meta = MagicMock()

        with patch("astroai.core.io.raw_io.read_raw", return_value=(fake_rgb, fake_meta)):
            result = _load_frame(Path("/fake/img.nef"))

        assert result.shape == (6, 10)
        np.testing.assert_allclose(result[0, 0], 0.25, rtol=1e-5)

    def test_raw_arw_luminance_mean(self, monkeypatch) -> None:
        from astroai.core.pipeline.load_frames_step import _load_frame
        from unittest.mock import patch, MagicMock

        # R=0.3, G=0.6, B=0.9 → mean=0.6
        fake_rgb = np.zeros((4, 4, 3), dtype=np.float32)
        fake_rgb[:, :, 0] = 0.3
        fake_rgb[:, :, 1] = 0.6
        fake_rgb[:, :, 2] = 0.9
        fake_meta = MagicMock()

        with patch("astroai.core.io.raw_io.read_raw", return_value=(fake_rgb, fake_meta)):
            result = _load_frame(Path("/fake/img.arw"))

        np.testing.assert_allclose(result[0, 0], 0.6, rtol=1e-5)

    def test_raw_in_pipeline_step(self, monkeypatch) -> None:
        from unittest.mock import patch, MagicMock

        fake_rgb = np.full((5, 5, 3), 1.0, dtype=np.float32)
        fake_meta = MagicMock()

        with patch("astroai.core.io.raw_io.read_raw", return_value=(fake_rgb, fake_meta)):
            step = LoadFramesStep([Path("/fake/img.cr2")])
            out = step.execute(PipelineContext())

        assert len(out.images) == 1
        assert out.images[0].dtype == np.float32
        assert out.images[0].shape == (5, 5)
