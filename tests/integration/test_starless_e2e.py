"""End-to-end integration test for the Starless pipeline feature.

Verifies the complete flow: synthetic starfield -> StarRemovalStep -> ExportStep
across all supported formats (FITS, TIFF, XISF) with starless enabled/disabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.core.pipeline.base import Pipeline, PipelineContext, PipelineProgress
from astroai.core.pipeline.export_step import ExportFormat, ExportStep
from astroai.processing.stars import StarRemovalStep


# --- Synthetic data generators ---


def make_synthetic_starfield(
    height: int = 128,
    width: int = 128,
    n_stars: int = 15,
    noise_std: float = 5.0,
    seed: int = 42,
) -> NDArray[np.floating[Any]]:
    """Generate a synthetic starfield image (float64) with Gaussian PSF stars."""
    rng = np.random.default_rng(seed)
    img = rng.normal(loc=100.0, scale=noise_std, size=(height, width)).astype(np.float64)
    img = np.clip(img, 0, None)

    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(n_stars):
        cy = rng.integers(10, height - 10)
        cx = rng.integers(10, width - 10)
        flux = rng.uniform(500, 3000)
        sigma = rng.uniform(1.5, 3.0)
        star = flux * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        img += star.astype(np.float64)

    return img


# --- Fixtures ---


@pytest.fixture()
def starfield() -> NDArray[np.floating[Any]]:
    """128x128 float64 synthetic starfield."""
    return make_synthetic_starfield(height=128, width=128, n_stars=15, seed=42)


@pytest.fixture()
def context_with_result(starfield: NDArray[np.floating[Any]]) -> PipelineContext:
    """PipelineContext with the starfield set as result."""
    ctx = PipelineContext()
    ctx.result = starfield
    return ctx


# --- StarRemovalStep Tests ---


class TestStarRemovalStepIntegration:
    """Verify StarRemovalStep populates starless_image and star_mask on PipelineContext."""

    def test_star_removal_populates_starless_image(
        self, context_with_result: PipelineContext
    ) -> None:
        step = StarRemovalStep()
        result_ctx = step.execute(context_with_result)

        assert result_ctx.starless_image is not None
        assert result_ctx.starless_image.shape == (128, 128)
        assert np.isfinite(result_ctx.starless_image).all()

    def test_star_removal_populates_star_mask(
        self, context_with_result: PipelineContext
    ) -> None:
        step = StarRemovalStep()
        result_ctx = step.execute(context_with_result)

        assert result_ctx.star_mask is not None
        assert result_ctx.star_mask.shape == (128, 128)
        assert result_ctx.star_mask.dtype == np.float32
        # Mask should be binary (0.0 or 1.0)
        unique_values = np.unique(result_ctx.star_mask)
        assert all(v in (0.0, 1.0) for v in unique_values)

    def test_star_removal_modifies_result(
        self, context_with_result: PipelineContext
    ) -> None:
        original_result = context_with_result.result.copy()
        step = StarRemovalStep()
        result_ctx = step.execute(context_with_result)

        # Result should now be the starless image (stars removed)
        assert result_ctx.result is not None
        # Starless image should differ from original (stars are removed)
        assert not np.array_equal(result_ctx.result, original_result)

    def test_star_removal_with_no_data_returns_context(self) -> None:
        """If no image data is available, step returns context unchanged."""
        ctx = PipelineContext()
        step = StarRemovalStep()
        result_ctx = step.execute(ctx)

        assert result_ctx.starless_image is None
        assert result_ctx.star_mask is None

    def test_star_removal_mask_marks_star_regions(
        self, context_with_result: PipelineContext
    ) -> None:
        """Star mask should have some True values where stars were detected."""
        step = StarRemovalStep()
        result_ctx = step.execute(context_with_result)

        assert result_ctx.star_mask is not None
        # There should be some star pixels detected (our starfield has 15 stars)
        assert result_ctx.star_mask.sum() > 0


# --- Export Step with Starless Tests ---


class TestExportStepStarless:
    """Verify ExportStep writes starless and starmask files for each format."""

    @pytest.fixture()
    def context_after_star_removal(
        self, context_with_result: PipelineContext
    ) -> PipelineContext:
        """Run StarRemovalStep to populate starless_image and star_mask."""
        step = StarRemovalStep()
        return step.execute(context_with_result)

    @pytest.mark.parametrize(
        "fmt,ext",
        [
            (ExportFormat.FITS, ".fits"),
            (ExportFormat.TIFF32, ".tif"),
            (ExportFormat.XISF, ".xisf"),
        ],
    )
    def test_export_creates_all_starless_files(
        self,
        context_after_star_removal: PipelineContext,
        tmp_path: Path,
        fmt: ExportFormat,
        ext: str,
    ) -> None:
        """Export with starless enabled should produce main, _starless, and _starmask files."""
        filename = "astro_output"
        export_step = ExportStep(
            output_dir=tmp_path,
            fmt=fmt,
            filename=filename,
            export_starless=True,
            export_star_mask=True,
        )

        result_ctx = export_step.execute(context_after_star_removal)

        # Verify all expected output files exist
        main_file = tmp_path / f"{filename}{ext}"
        starless_file = tmp_path / f"{filename}_starless{ext}"
        starmask_file = tmp_path / f"{filename}_starmask{ext}"

        assert main_file.exists(), f"Main output file missing: {main_file}"
        assert starless_file.exists(), f"Starless file missing: {starless_file}"
        assert starmask_file.exists(), f"Star mask file missing: {starmask_file}"

        # Verify metadata paths recorded
        assert "export_path" in result_ctx.metadata
        assert "export_starless_path" in result_ctx.metadata
        assert "export_starmask_path" in result_ctx.metadata

    @pytest.mark.parametrize(
        "fmt,ext",
        [
            (ExportFormat.FITS, ".fits"),
            (ExportFormat.TIFF32, ".tif"),
            (ExportFormat.XISF, ".xisf"),
        ],
    )
    def test_export_files_are_not_empty(
        self,
        context_after_star_removal: PipelineContext,
        tmp_path: Path,
        fmt: ExportFormat,
        ext: str,
    ) -> None:
        """All exported files should have non-zero size."""
        filename = "result"
        export_step = ExportStep(
            output_dir=tmp_path,
            fmt=fmt,
            filename=filename,
            export_starless=True,
            export_star_mask=True,
        )

        export_step.execute(context_after_star_removal)

        for suffix in ("", "_starless", "_starmask"):
            filepath = tmp_path / f"{filename}{suffix}{ext}"
            assert filepath.stat().st_size > 0, f"File is empty: {filepath}"

    @pytest.mark.parametrize(
        "fmt,ext",
        [
            (ExportFormat.FITS, ".fits"),
            (ExportFormat.TIFF32, ".tif"),
            (ExportFormat.XISF, ".xisf"),
        ],
    )
    def test_export_without_starless_flag_skips_extra_files(
        self,
        context_after_star_removal: PipelineContext,
        tmp_path: Path,
        fmt: ExportFormat,
        ext: str,
    ) -> None:
        """When export_starless=False, only main file should be created."""
        filename = "no_starless"
        export_step = ExportStep(
            output_dir=tmp_path,
            fmt=fmt,
            filename=filename,
            export_starless=False,
            export_star_mask=False,
        )

        export_step.execute(context_after_star_removal)

        main_file = tmp_path / f"{filename}{ext}"
        starless_file = tmp_path / f"{filename}_starless{ext}"
        starmask_file = tmp_path / f"{filename}_starmask{ext}"

        assert main_file.exists()
        assert not starless_file.exists(), "Starless file should NOT be created"
        assert not starmask_file.exists(), "Starmask file should NOT be created"


# --- Full Pipeline E2E (Starless Flow) ---


class TestStarlessPipelineE2E:
    """Full E2E: StarRemovalStep + ExportStep in a Pipeline."""

    @pytest.mark.parametrize(
        "fmt,ext",
        [
            (ExportFormat.FITS, ".fits"),
            (ExportFormat.TIFF32, ".tif"),
            (ExportFormat.XISF, ".xisf"),
        ],
    )
    def test_full_starless_pipeline(
        self,
        starfield: NDArray[np.floating[Any]],
        tmp_path: Path,
        fmt: ExportFormat,
        ext: str,
    ) -> None:
        """Run complete pipeline: StarRemoval -> Export with starless enabled."""
        filename = "starless_e2e"
        pipeline = Pipeline([
            StarRemovalStep(),
            ExportStep(
                output_dir=tmp_path,
                fmt=fmt,
                filename=filename,
                export_starless=True,
                export_star_mask=True,
            ),
        ])

        ctx = PipelineContext(result=starfield)
        result_ctx = pipeline.run(ctx)

        # Verify pipeline completed with starless data
        assert result_ctx.starless_image is not None
        assert result_ctx.star_mask is not None

        # Verify all output files
        assert (tmp_path / f"{filename}{ext}").exists()
        assert (tmp_path / f"{filename}_starless{ext}").exists()
        assert (tmp_path / f"{filename}_starmask{ext}").exists()

    @pytest.mark.parametrize(
        "fmt,ext",
        [
            (ExportFormat.FITS, ".fits"),
            (ExportFormat.TIFF32, ".tif"),
            (ExportFormat.XISF, ".xisf"),
        ],
    )
    def test_pipeline_without_star_removal_no_starless_files(
        self,
        starfield: NDArray[np.floating[Any]],
        tmp_path: Path,
        fmt: ExportFormat,
        ext: str,
    ) -> None:
        """Pipeline without StarRemovalStep should NOT produce _starless files
        even if export flags are set, because context has no starless data."""
        filename = "no_star_removal"
        pipeline = Pipeline([
            ExportStep(
                output_dir=tmp_path,
                fmt=fmt,
                filename=filename,
                export_starless=True,
                export_star_mask=True,
            ),
        ])

        ctx = PipelineContext(result=starfield)
        result_ctx = pipeline.run(ctx)

        # Main file should exist
        assert (tmp_path / f"{filename}{ext}").exists()

        # Starless files should NOT exist (no StarRemovalStep was run)
        starless_file = tmp_path / f"{filename}_starless{ext}"
        starmask_file = tmp_path / f"{filename}_starmask{ext}"
        assert not starless_file.exists(), (
            "Starless file should NOT exist without StarRemovalStep"
        )
        assert not starmask_file.exists(), (
            "Starmask file should NOT exist without StarRemovalStep"
        )

        # Context should have no starless data
        assert result_ctx.starless_image is None
        assert result_ctx.star_mask is None

    def test_export_only_starless_no_mask(
        self,
        starfield: NDArray[np.floating[Any]],
        tmp_path: Path,
    ) -> None:
        """Export with export_starless=True but export_star_mask=False."""
        filename = "starless_only"
        pipeline = Pipeline([
            StarRemovalStep(),
            ExportStep(
                output_dir=tmp_path,
                fmt=ExportFormat.FITS,
                filename=filename,
                export_starless=True,
                export_star_mask=False,
            ),
        ])

        ctx = PipelineContext(result=starfield)
        pipeline.run(ctx)

        assert (tmp_path / f"{filename}.fits").exists()
        assert (tmp_path / f"{filename}_starless.fits").exists()
        assert not (tmp_path / f"{filename}_starmask.fits").exists()

    def test_export_only_mask_no_starless(
        self,
        starfield: NDArray[np.floating[Any]],
        tmp_path: Path,
    ) -> None:
        """Export with export_starless=False but export_star_mask=True."""
        filename = "mask_only"
        pipeline = Pipeline([
            StarRemovalStep(),
            ExportStep(
                output_dir=tmp_path,
                fmt=ExportFormat.XISF,
                filename=filename,
                export_starless=False,
                export_star_mask=True,
            ),
        ])

        ctx = PipelineContext(result=starfield)
        pipeline.run(ctx)

        assert (tmp_path / f"{filename}.xisf").exists()
        assert not (tmp_path / f"{filename}_starless.xisf").exists()
        assert (tmp_path / f"{filename}_starmask.xisf").exists()

    def test_starless_output_differs_from_original(
        self,
        starfield: NDArray[np.floating[Any]],
    ) -> None:
        """Starless result should be meaningfully different from the input starfield."""
        ctx = PipelineContext(result=starfield.copy())
        step = StarRemovalStep()
        result_ctx = step.execute(ctx)

        assert result_ctx.starless_image is not None
        # The difference between original and starless should be non-trivial
        diff = np.abs(starfield - result_ctx.starless_image)
        assert diff.max() > 0, "Star removal had no effect"

    def test_pipeline_progress_tracking(
        self,
        starfield: NDArray[np.floating[Any]],
        tmp_path: Path,
    ) -> None:
        """Pipeline should report progress for both StarRemoval and Export steps."""
        progress_log: list[PipelineProgress] = []

        def track_progress(p: PipelineProgress) -> None:
            progress_log.append(p)

        pipeline = Pipeline([
            StarRemovalStep(),
            ExportStep(
                output_dir=tmp_path,
                fmt=ExportFormat.FITS,
                filename="progress_test",
                export_starless=True,
                export_star_mask=True,
            ),
        ])

        ctx = PipelineContext(result=starfield)
        pipeline.run(ctx, progress=track_progress)

        assert len(progress_log) > 0
        messages = [p.message for p in progress_log]
        # Should see Running messages for both steps
        assert any("Star Removal" in m for m in messages)
        assert any("export" in m.lower() for m in messages)

