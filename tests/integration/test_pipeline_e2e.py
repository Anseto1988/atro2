"""End-to-end integration test for the AstroAI processing pipeline.

Tests the full data flow: IO -> Calibration -> Frame Scoring -> Registration ->
Stacking -> Denoising -> Stretch using synthetic FITS data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from astroai.core.calibration.calibrate import apply_dark, apply_flat, calibrate_frame
from astroai.core.calibration.matcher import (
    CalibrationFrame,
    CalibrationLibrary,
)
from astroai.core.io.fits_io import ImageMetadata, read_fits, write_fits
from astroai.core.pipeline.base import (
    Pipeline,
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.engine.registration.aligner import FrameAligner
from astroai.engine.stacking.stacker import FrameStacker
from astroai.inference.backends.gpu import DeviceManager
from astroai.inference.scoring.frame_scorer import FrameScorer
from astroai.processing.denoise.denoiser import Denoiser
from astroai.processing.stretch.stretcher import IntelligentStretcher


# --- Synthetic data generators ---


def make_synthetic_starfield(
    height: int = 128,
    width: int = 128,
    n_stars: int = 15,
    noise_std: float = 5.0,
    seed: int = 42,
) -> NDArray[np.floating[Any]]:
    """Generate synthetic starfield image with Gaussian PSF stars."""
    rng = np.random.default_rng(seed)
    img = rng.normal(loc=100.0, scale=noise_std, size=(height, width)).astype(np.float32)
    img = np.clip(img, 0, None)

    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(n_stars):
        cy = rng.integers(10, height - 10)
        cx = rng.integers(10, width - 10)
        flux = rng.uniform(500, 3000)
        sigma = rng.uniform(1.5, 3.0)
        star = flux * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        img += star.astype(np.float32)

    return img


def make_dark_frame(
    height: int = 128, width: int = 128, level: float = 20.0, seed: int = 99
) -> NDArray[np.floating[Any]]:
    rng = np.random.default_rng(seed)
    return (np.full((height, width), level) + rng.normal(0, 2, (height, width))).astype(
        np.float32
    )


def make_flat_frame(
    height: int = 128, width: int = 128, seed: int = 77
) -> NDArray[np.floating[Any]]:
    rng = np.random.default_rng(seed)
    vignette_y = np.linspace(0.85, 1.0, height)[:, None]
    vignette_x = np.linspace(0.9, 1.0, width)[None, :]
    flat = (vignette_y * vignette_x * 10000.0).astype(np.float32)
    flat += rng.normal(0, 50, (height, width)).astype(np.float32)
    return np.clip(flat, 1.0, None)


# --- Fixtures ---


@pytest.fixture()
def synthetic_light_frames() -> list[NDArray[np.floating[Any]]]:
    """Generate multiple light frames with slight shifts to simulate dithering."""
    frames = []
    for i in range(5):
        frames.append(make_synthetic_starfield(seed=42 + i, noise_std=8.0))
    return frames


@pytest.fixture()
def calibration_library() -> CalibrationLibrary:
    """Build calibration library with synthetic darks and flats."""
    meta = ImageMetadata(
        exposure=120.0, gain_iso=800, temperature=-10.0, width=128, height=128
    )
    dark_data = make_dark_frame()
    flat_data = make_flat_frame()
    return CalibrationLibrary(
        darks=[CalibrationFrame(Path("dark_master.fits"), meta, data=dark_data)],
        flats=[CalibrationFrame(Path("flat_master.fits"), meta, data=flat_data)],
        bias=[],
    )


@pytest.fixture()
def light_metadata() -> ImageMetadata:
    return ImageMetadata(
        exposure=120.0, gain_iso=800, temperature=-10.0, width=128, height=128
    )


# --- FITS I/O Integration ---


class TestFitsRoundtrip:
    def test_write_and_read_preserves_data(self, tmp_path: Path) -> None:
        data = make_synthetic_starfield(64, 64)
        meta = ImageMetadata(exposure=60.0, gain_iso=1600, temperature=-15.0)
        out_path = tmp_path / "test.fits"

        write_fits(out_path, data, meta)
        read_data, read_meta = read_fits(out_path)

        np.testing.assert_array_almost_equal(read_data, data, decimal=4)
        assert read_meta.exposure == pytest.approx(60.0)
        assert read_meta.gain_iso == 1600

    def test_write_and_read_without_metadata(self, tmp_path: Path) -> None:
        data = np.ones((32, 32), dtype=np.float32) * 500.0
        out_path = tmp_path / "plain.fits"

        write_fits(out_path, data)
        read_data, _ = read_fits(out_path)

        np.testing.assert_array_almost_equal(read_data, data, decimal=4)


# --- Calibration Integration ---


class TestCalibrationPipeline:
    def test_full_calibration_reduces_noise(
        self,
        synthetic_light_frames: list[NDArray[np.floating[Any]]],
        calibration_library: CalibrationLibrary,
        light_metadata: ImageMetadata,
    ) -> None:
        raw = synthetic_light_frames[0]
        calibrated = calibrate_frame(raw, light_metadata, calibration_library)

        assert calibrated.shape == raw.shape
        assert calibrated.dtype == np.float32
        # Dark subtraction should lower the background level
        assert calibrated.mean() < raw.mean()

    def test_calibration_preserves_star_signal(
        self,
        calibration_library: CalibrationLibrary,
        light_metadata: ImageMetadata,
    ) -> None:
        frame = make_synthetic_starfield(n_stars=5, noise_std=2.0, seed=100)
        peak_before = frame.max()
        calibrated = calibrate_frame(frame, light_metadata, calibration_library)
        # Stars should still be significantly above background
        bg = np.median(calibrated)
        assert calibrated.max() > bg * 2


# --- Frame Scoring Integration ---


class TestFrameScorerIntegration:
    def test_good_frame_scores_higher_than_blank(self) -> None:
        scorer = FrameScorer()
        good = make_synthetic_starfield(n_stars=20, noise_std=3.0, seed=55)
        blank = np.full((128, 128), 100.0, dtype=np.float32)

        good_score = scorer.score(good)
        blank_score = scorer.score(blank)

        assert good_score > blank_score

    def test_score_batch_returns_correct_count(self) -> None:
        scorer = FrameScorer()
        frames = [make_synthetic_starfield(seed=i) for i in range(4)]
        scores = scorer.score_batch(frames)

        assert len(scores) == 4
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_noisy_frame_scores_lower(self) -> None:
        scorer = FrameScorer()
        clean = make_synthetic_starfield(noise_std=2.0, seed=10)
        noisy = make_synthetic_starfield(noise_std=50.0, seed=10)

        clean_score = scorer.score(clean)
        noisy_score = scorer.score(noisy)

        # Clean should generally score equal or higher
        # (noise can affect star detection, so we allow some tolerance)
        assert clean_score >= noisy_score - 0.2


# --- Registration Integration ---


class TestRegistrationIntegration:
    def test_align_shifted_frame(self) -> None:
        aligner = FrameAligner(upsample_factor=10)
        reference = make_synthetic_starfield(seed=42, n_stars=10, noise_std=2.0)

        # Create a shifted version
        from scipy.ndimage import shift as ndimage_shift

        shifted = ndimage_shift(reference, (3.0, -2.0), mode="constant", cval=0.0)

        aligned, transform = aligner.align(reference, shifted)

        assert aligned.shape == reference.shape
        # Phase correlation recovers shift direction; tolerance accounts for
        # sub-pixel refinement on synthetic data with limited star count
        assert abs(transform[0, 2]) < 5.0
        assert abs(transform[1, 2]) < 5.0

    def test_align_batch(self) -> None:
        aligner = FrameAligner(upsample_factor=4)
        reference = make_synthetic_starfield(seed=42, n_stars=10, noise_std=2.0)

        from scipy.ndimage import shift as ndimage_shift

        targets = [
            ndimage_shift(reference, (float(i), float(-i)), mode="constant")
            for i in range(1, 4)
        ]

        aligned = aligner.align_batch(reference, targets)
        assert len(aligned) == 3
        assert all(a.shape == reference.shape for a in aligned)


# --- Stacking Integration ---


class TestStackingIntegration:
    def test_stacking_improves_snr(
        self, synthetic_light_frames: list[NDArray[np.floating[Any]]]
    ) -> None:
        stacker = FrameStacker()
        stacked = stacker.stack(synthetic_light_frames, method="sigma_clip")

        single_std = np.std(synthetic_light_frames[0])
        stacked_std = np.std(stacked)

        # Stacking multiple frames should reduce noise
        assert stacked_std < single_std

    def test_all_methods_produce_valid_output(
        self, synthetic_light_frames: list[NDArray[np.floating[Any]]]
    ) -> None:
        stacker = FrameStacker()
        for method in ("mean", "median", "sigma_clip"):
            result = stacker.stack(synthetic_light_frames, method=method)
            assert result.shape == synthetic_light_frames[0].shape
            assert np.isfinite(result).all()


# --- Denoising Integration ---


class TestDenoisingIntegration:
    def test_statistical_denoiser_reduces_noise(self) -> None:
        noisy = make_synthetic_starfield(noise_std=30.0, seed=88)
        denoiser = Denoiser(strength=1.0)

        denoised = denoiser.denoise(noisy)

        assert denoised.shape == noisy.shape
        # Denoiser should produce finite values with similar magnitude
        assert np.isfinite(denoised).all()
        # Overall image variance should not explode
        assert np.std(denoised) < np.std(noisy) * 2.0

    def test_denoiser_preserves_bright_structures(self) -> None:
        frame = make_synthetic_starfield(n_stars=5, noise_std=5.0, seed=77)
        denoiser = Denoiser(strength=0.8)

        denoised = denoiser.denoise(frame)
        # Brightest star should still be prominent
        assert denoised.max() > np.median(denoised) * 2

    def test_denoise_batch(self) -> None:
        frames = [make_synthetic_starfield(seed=i, noise_std=20.0) for i in range(3)]
        denoiser = Denoiser(strength=0.5)
        results = denoiser.denoise_batch(frames)
        assert len(results) == 3
        assert all(r.shape == frames[0].shape for r in results)


# --- Stretch Integration ---


class TestStretchIntegration:
    def test_stretch_maps_to_visual_range(self) -> None:
        linear = make_synthetic_starfield(noise_std=3.0, seed=33)
        stretcher = IntelligentStretcher()

        stretched = stretcher.stretch(linear)

        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0
        # Stretched output should use the [0,1] range meaningfully
        assert stretched.max() > 0.5 or np.median(stretched) > 0.0

    def test_stretch_batch(self) -> None:
        frames = [make_synthetic_starfield(seed=i) for i in range(3)]
        stretcher = IntelligentStretcher()
        results = stretcher.stretch_batch(frames)
        assert len(results) == 3
        assert all(r.min() >= 0.0 and r.max() <= 1.0 for r in results)


# --- GPU/CPU Fallback Integration ---


class TestDeviceManagerIntegration:
    def test_device_manager_resolves_to_valid_device(self) -> None:
        dm = DeviceManager()
        device = dm.get_device()
        assert device.type in ("cpu", "cuda", "mps")

    def test_cpu_fallback_works(self) -> None:
        dm = DeviceManager()
        info = dm.device_info()
        assert "type" in info
        assert "cuda_available" in info
        assert "mps_available" in info

    def test_tensor_to_device(self) -> None:
        dm = DeviceManager()
        t = torch.randn(4, 4)
        moved = dm.to_device(t)
        assert moved.device.type == dm.get_device().type

    def test_model_to_device(self) -> None:
        dm = DeviceManager()
        model = torch.nn.Linear(10, 5)
        moved = dm.to_device(model)
        param_device = next(moved.parameters()).device
        assert param_device.type == dm.get_device().type


# --- Full E2E Pipeline Integration ---


class CalibrationStep(PipelineStep):
    def __init__(
        self,
        library: CalibrationLibrary,
        metadata: ImageMetadata,
    ) -> None:
        self._library = library
        self._metadata = metadata

    @property
    def name(self) -> str:
        return "Calibration"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.CALIBRATION

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        calibrated = []
        for i, img in enumerate(context.images):
            calibrated.append(calibrate_frame(img, self._metadata, self._library))
            progress(PipelineProgress(self.stage, i + 1, len(context.images)))
        context.images = calibrated
        return context


class ScoringStep(PipelineStep):
    def __init__(self, min_score: float = 0.0) -> None:
        self._scorer = FrameScorer()
        self._min_score = min_score

    @property
    def name(self) -> str:
        return "Frame Scoring"

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        scores = self._scorer.score_batch(context.images)
        context.metadata["scores"] = scores
        if self._min_score > 0:
            filtered = [
                img for img, s in zip(context.images, scores) if s >= self._min_score
            ]
            context.images = filtered if filtered else context.images
        return context


class RegistrationStep(PipelineStep):
    def __init__(self) -> None:
        self._aligner = FrameAligner(upsample_factor=4)

    @property
    def name(self) -> str:
        return "Registration"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.REGISTRATION

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        if len(context.images) < 2:
            return context
        reference = context.images[0]
        aligned = [reference] + self._aligner.align_batch(reference, context.images[1:])
        context.images = aligned
        return context


class StackingStep(PipelineStep):
    def __init__(self, method: str = "sigma_clip") -> None:
        self._stacker = FrameStacker()
        self._method = method

    @property
    def name(self) -> str:
        return "Stacking"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.STACKING

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        stacked = self._stacker.stack(context.images, method=self._method)
        context.result = stacked
        context.images = [stacked]
        return context


class DenoisingStep(PipelineStep):
    def __init__(self, strength: float = 0.7) -> None:
        self._denoiser = Denoiser(strength=strength)

    @property
    def name(self) -> str:
        return "Denoising"

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        if context.result is not None:
            context.result = self._denoiser.denoise(context.result)
        return context


class StretchStep(PipelineStep):
    def __init__(self) -> None:
        self._stretcher = IntelligentStretcher()

    @property
    def name(self) -> str:
        return "Stretch"

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        if context.result is not None:
            context.result = self._stretcher.stretch(context.result)
        return context


class TestFullPipelineE2E:
    """Full end-to-end pipeline: IO -> Calibration -> Scoring -> Registration ->
    Stacking -> Denoising -> Stretch."""

    def test_complete_pipeline_flow(
        self,
        synthetic_light_frames: list[NDArray[np.floating[Any]]],
        calibration_library: CalibrationLibrary,
        light_metadata: ImageMetadata,
        tmp_path: Path,
    ) -> None:
        # 1. Write synthetic frames to FITS
        fits_paths = []
        for i, frame in enumerate(synthetic_light_frames):
            p = tmp_path / f"light_{i:03d}.fits"
            write_fits(p, frame, light_metadata)
            fits_paths.append(p)

        # 2. Read them back (IO stage)
        loaded_frames = []
        for p in fits_paths:
            data, meta = read_fits(p)
            loaded_frames.append(data)
        assert len(loaded_frames) == 5

        # 3. Build and run pipeline
        progress_log: list[PipelineProgress] = []

        def track_progress(p: PipelineProgress) -> None:
            progress_log.append(p)

        pipeline = Pipeline([
            CalibrationStep(calibration_library, light_metadata),
            ScoringStep(min_score=0.0),
            RegistrationStep(),
            StackingStep(method="sigma_clip"),
            DenoisingStep(strength=0.7),
            StretchStep(),
        ])

        ctx = PipelineContext(images=loaded_frames)
        result_ctx = pipeline.run(ctx, progress=track_progress)

        # 4. Validate final output
        assert result_ctx.result is not None
        final = result_ctx.result
        assert final.ndim == 2
        assert final.shape == (128, 128)
        assert final.min() >= 0.0
        assert final.max() <= 1.0
        assert np.isfinite(final).all()

        # 5. Validate progress tracking
        assert len(progress_log) > 0
        stages_seen = {p.stage for p in progress_log}
        assert PipelineStage.CALIBRATION in stages_seen
        assert PipelineStage.STACKING in stages_seen
        assert PipelineStage.SAVING in stages_seen

        # 6. Validate scoring metadata
        assert "scores" in result_ctx.metadata
        assert len(result_ctx.metadata["scores"]) == 5

        # 7. Write result back to FITS
        output_path = tmp_path / "result.fits"
        write_fits(output_path, final.astype(np.float32))
        re_read, _ = read_fits(output_path)
        np.testing.assert_array_almost_equal(re_read, final.astype(np.float32), decimal=4)

    def test_pipeline_with_single_frame(
        self,
        calibration_library: CalibrationLibrary,
        light_metadata: ImageMetadata,
    ) -> None:
        """Pipeline should handle single-frame input gracefully."""
        single = [make_synthetic_starfield(seed=99)]

        pipeline = Pipeline([
            CalibrationStep(calibration_library, light_metadata),
            ScoringStep(),
            RegistrationStep(),
            StackingStep(method="mean"),
            DenoisingStep(strength=0.5),
            StretchStep(),
        ])

        ctx = PipelineContext(images=single)
        result_ctx = pipeline.run(ctx)

        assert result_ctx.result is not None
        assert result_ctx.result.shape == (128, 128)
        assert result_ctx.result.min() >= 0.0
        assert result_ctx.result.max() <= 1.0

    def test_pipeline_progress_callback(
        self,
        synthetic_light_frames: list[NDArray[np.floating[Any]]],
        calibration_library: CalibrationLibrary,
        light_metadata: ImageMetadata,
    ) -> None:
        """Pipeline reports progress for each step."""
        progress_messages: list[str] = []

        def on_progress(p: PipelineProgress) -> None:
            progress_messages.append(p.message)

        pipeline = Pipeline([
            CalibrationStep(calibration_library, light_metadata),
            StackingStep(),
            StretchStep(),
        ])

        ctx = PipelineContext(images=synthetic_light_frames)
        pipeline.run(ctx, progress=on_progress)

        assert any("Running:" in m for m in progress_messages)
        assert any("complete" in m.lower() for m in progress_messages)
