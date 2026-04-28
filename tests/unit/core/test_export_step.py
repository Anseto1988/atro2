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


class TestExportStepStarOutputs:
    def test_export_starless(self, tmp_path: Path) -> None:
        data = np.random.default_rng(10).random((32, 32)).astype(np.float32)
        starless = np.random.default_rng(11).random((32, 32)).astype(np.float32)
        ctx = PipelineContext(result=data, starless_image=starless)
        step = ExportStep(
            tmp_path, fmt=ExportFormat.FITS,
            filename="astro", export_starless=True,
        )
        result = step.execute(ctx)
        assert (tmp_path / "astro.fits").exists()
        assert (tmp_path / "astro_starless.fits").exists()
        assert "export_starless_path" in result.metadata

    def test_export_star_mask(self, tmp_path: Path) -> None:
        data = np.random.default_rng(12).random((32, 32)).astype(np.float32)
        mask = (np.random.default_rng(13).random((32, 32)) > 0.9).astype(np.float32)
        ctx = PipelineContext(result=data, star_mask=mask)
        step = ExportStep(
            tmp_path, fmt=ExportFormat.TIFF32,
            filename="nebula", export_star_mask=True,
        )
        result = step.execute(ctx)
        assert (tmp_path / "nebula.tif").exists()
        assert (tmp_path / "nebula_starmask.tif").exists()
        assert "export_starmask_path" in result.metadata

    def test_no_starless_skips_export(self, tmp_path: Path) -> None:
        data = np.zeros((16, 16), dtype=np.float32)
        ctx = PipelineContext(result=data)
        step = ExportStep(
            tmp_path, fmt=ExportFormat.FITS,
            export_starless=True, export_star_mask=True,
        )
        result = step.execute(ctx)
        assert (tmp_path / "output.fits").exists()
        assert "export_starless_path" not in result.metadata
        assert "export_starmask_path" not in result.metadata

    def test_both_star_outputs(self, tmp_path: Path) -> None:
        data = np.random.default_rng(14).random((32, 32)).astype(np.float32)
        starless = np.random.default_rng(15).random((32, 32)).astype(np.float32)
        mask = (np.random.default_rng(16).random((32, 32)) > 0.8).astype(np.float32)
        ctx = PipelineContext(result=data, starless_image=starless, star_mask=mask)
        step = ExportStep(
            tmp_path, fmt=ExportFormat.XISF,
            filename="m42", export_starless=True, export_star_mask=True,
        )
        result = step.execute(ctx)
        assert (tmp_path / "m42.xisf").exists()
        assert (tmp_path / "m42_starless.xisf").exists()
        assert (tmp_path / "m42_starmask.xisf").exists()


class TestExportStepMetadata:
    def test_metadata_read_from_context_metadata(self, tmp_path: Path) -> None:
        """ExportStep picks up metadata from context.metadata['metadata'] when not set directly (line 79)."""
        data = np.random.default_rng(17).random((16, 16)).astype(np.float32)
        meta = ImageMetadata(exposure=1.0, gain_iso=100, date_obs="2024-01-01T00:00:00")
        ctx = PipelineContext(result=data, metadata={"metadata": meta})
        step = ExportStep(tmp_path, fmt=ExportFormat.FITS, filename="ctx_meta")
        result = step.execute(ctx)
        assert (tmp_path / "ctx_meta.fits").exists()
