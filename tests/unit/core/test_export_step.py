from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astroai.core.io.fits_io import ImageMetadata
from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.core.pipeline.export_step import ExportFormat, ExportStep


class TestExportStep:
    def test_export_xisf(self, tmp_path: Path) -> None:
        data = np.random.default_rng(1).random((1, 32, 32)).astype(np.float32)
        ctx = PipelineContext(images=[data])
        step = ExportStep(tmp_path, fmt=ExportFormat.XISF)
        result = step.execute(ctx)
        assert (tmp_path / "output.xisf").exists()
        assert "export_path" in result.metadata

    def test_export_tiff(self, tmp_path: Path) -> None:
        data = np.random.default_rng(2).random((1, 20, 20)).astype(np.float32)
        ctx = PipelineContext(images=[data])
        step = ExportStep(tmp_path, fmt=ExportFormat.TIFF32)
        result = step.execute(ctx)
        assert (tmp_path / "output.tif").exists()

    def test_export_fits(self, tmp_path: Path) -> None:
        data = np.random.default_rng(3).random((1, 16, 16)).astype(np.float32)
        ctx = PipelineContext(images=[data])
        step = ExportStep(tmp_path, fmt=ExportFormat.FITS)
        result = step.execute(ctx)
        assert (tmp_path / "output.fits").exists()

    def test_custom_filename(self, tmp_path: Path) -> None:
        data = np.zeros((1, 8, 8), dtype=np.float32)
        ctx = PipelineContext(images=[data])
        step = ExportStep(tmp_path, fmt=ExportFormat.XISF, filename="my_image")
        step.execute(ctx)
        assert (tmp_path / "my_image.xisf").exists()

    def test_uses_result_over_images(self, tmp_path: Path) -> None:
        result_data = np.ones((1, 10, 10), dtype=np.float32)
        ctx = PipelineContext(images=[], result=result_data)
        step = ExportStep(tmp_path, fmt=ExportFormat.FITS)
        step.execute(ctx)
        assert (tmp_path / "output.fits").exists()

    def test_raises_on_empty_context(self, tmp_path: Path) -> None:
        ctx = PipelineContext(images=[])
        step = ExportStep(tmp_path, fmt=ExportFormat.XISF)
        with pytest.raises(ValueError, match="No image data"):
            step.execute(ctx)

    def test_stage_is_saving(self) -> None:
        step = ExportStep("/tmp", fmt=ExportFormat.XISF)
        assert step.stage == PipelineStage.SAVING

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "sub" / "dir"
        data = np.zeros((1, 4, 4), dtype=np.float32)
        ctx = PipelineContext(images=[data])
        ExportStep(out_dir, fmt=ExportFormat.FITS).execute(ctx)
        assert out_dir.exists()
