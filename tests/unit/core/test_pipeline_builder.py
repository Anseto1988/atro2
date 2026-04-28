"""Tests for PipelineBuilder."""
from __future__ import annotations

from pathlib import Path

import pytest

from astroai.core.pipeline.base import PipelineStage
from astroai.core.pipeline.builder import PipelineBuilder
from astroai.core.pipeline.export_step import ExportFormat, ExportStep
from astroai.core.pipeline.frame_selection_step import FrameSelectionStep
from astroai.engine.registration.pipeline_step import RegistrationStep
from astroai.engine.stacking.pipeline_step import StackingStep
from astroai.processing.background.pipeline_step import BackgroundRemovalStep
from astroai.processing.channels.pipeline_step import ChannelCombineStep, CombineMode
from astroai.processing.color.pipeline_step import ColorCalibrationStep
from astroai.processing.deconvolution.pipeline_step import DeconvolutionStep
from astroai.processing.denoise.pipeline_step import DenoiseStep
from astroai.processing.flat.pipeline_step import SyntheticFlatStep
from astroai.processing.stars.pipeline_step import StarRemovalStep
from astroai.processing.stretch.pipeline_step import StretchStep
from astroai.ui.models import PipelineModel


@pytest.fixture()
def model() -> PipelineModel:
    return PipelineModel()


@pytest.fixture()
def builder() -> PipelineBuilder:
    return PipelineBuilder()


class TestBuildCalibrationPipeline:
    def test_empty_when_no_optional_steps_enabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_calibration_pipeline(model)
        assert len(pipeline._steps) == 0

    def test_includes_frame_selection_when_enabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.frame_selection_enabled = True
        pipeline = builder.build_calibration_pipeline(model)
        assert any(isinstance(s, FrameSelectionStep) for s in pipeline._steps)

    def test_frame_selection_uses_model_min_score(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.frame_selection_enabled = True
        model.frame_selection_min_score = 0.7
        pipeline = builder.build_calibration_pipeline(model)
        step = next(s for s in pipeline._steps if isinstance(s, FrameSelectionStep))
        assert step._min_score == pytest.approx(0.7)

    def test_frame_selection_uses_model_max_rejected_fraction(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.frame_selection_enabled = True
        model.frame_selection_max_rejected_fraction = 0.6
        pipeline = builder.build_calibration_pipeline(model)
        step = next(s for s in pipeline._steps if isinstance(s, FrameSelectionStep))
        assert step._max_rejected_fraction == pytest.approx(0.6)

    def test_includes_synthetic_flat_when_enabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.synthetic_flat_enabled = True
        pipeline = builder.build_calibration_pipeline(model)
        assert any(isinstance(s, SyntheticFlatStep) for s in pipeline._steps)

    def test_synthetic_flat_uses_model_smoothing_sigma(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.synthetic_flat_enabled = True
        model.synthetic_flat_smoothing_sigma = 12.0
        pipeline = builder.build_calibration_pipeline(model)
        step = next(s for s in pipeline._steps if isinstance(s, SyntheticFlatStep))
        assert step._generator._smoothing_sigma == pytest.approx(12.0)

    def test_both_steps_in_correct_order(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.frame_selection_enabled = True
        model.synthetic_flat_enabled = True
        pipeline = builder.build_calibration_pipeline(model)
        assert len(pipeline._steps) == 2
        assert isinstance(pipeline._steps[0], FrameSelectionStep)
        assert isinstance(pipeline._steps[1], SyntheticFlatStep)


class TestBuildProcessingPipeline:
    def test_always_includes_stretch(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_processing_pipeline(model)
        assert any(isinstance(s, StretchStep) for s in pipeline._steps)

    def test_always_includes_denoise(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_processing_pipeline(model)
        assert any(isinstance(s, DenoiseStep) for s in pipeline._steps)

    def test_stretch_uses_model_target_background(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.stretch_target_background = 0.3
        pipeline = builder.build_processing_pipeline(model)
        step = next(s for s in pipeline._steps if isinstance(s, StretchStep))
        assert step._stretcher._target_bg == pytest.approx(0.3)

    def test_denoise_uses_model_strength(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.denoise_strength = 0.5
        pipeline = builder.build_processing_pipeline(model)
        step = next(s for s in pipeline._steps if isinstance(s, DenoiseStep))
        assert step._denoiser._strength == pytest.approx(0.5)

    def test_denoise_uses_model_tile_size(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.denoise_tile_size = 256
        pipeline = builder.build_processing_pipeline(model)
        step = next(s for s in pipeline._steps if isinstance(s, DenoiseStep))
        assert step._denoiser._tile_size == 256

    def test_background_removal_not_included_when_disabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_processing_pipeline(model)
        assert not any(isinstance(s, BackgroundRemovalStep) for s in pipeline._steps)

    def test_background_removal_included_when_enabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.background_removal_enabled = True
        pipeline = builder.build_processing_pipeline(model)
        assert any(isinstance(s, BackgroundRemovalStep) for s in pipeline._steps)

    def test_color_calibration_not_included_when_disabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_processing_pipeline(model)
        assert not any(isinstance(s, ColorCalibrationStep) for s in pipeline._steps)

    def test_color_calibration_included_when_enabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.color_calibration_enabled = True
        pipeline = builder.build_processing_pipeline(model)
        assert any(isinstance(s, ColorCalibrationStep) for s in pipeline._steps)

    def test_deconvolution_not_included_when_disabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_processing_pipeline(model)
        assert not any(isinstance(s, DeconvolutionStep) for s in pipeline._steps)

    def test_deconvolution_included_when_enabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.deconvolution_enabled = True
        pipeline = builder.build_processing_pipeline(model)
        assert any(isinstance(s, DeconvolutionStep) for s in pipeline._steps)

    def test_deconvolution_uses_model_iterations(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.deconvolution_enabled = True
        model.deconvolution_iterations = 20
        pipeline = builder.build_processing_pipeline(model)
        step = next(s for s in pipeline._steps if isinstance(s, DeconvolutionStep))
        assert step._deconvolver._iterations == 20

    def test_starless_not_included_when_disabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_processing_pipeline(model)
        assert not any(isinstance(s, StarRemovalStep) for s in pipeline._steps)

    def test_starless_included_when_enabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.starless_enabled = True
        pipeline = builder.build_processing_pipeline(model)
        assert any(isinstance(s, StarRemovalStep) for s in pipeline._steps)

    def test_starless_uses_model_reduce_params(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.starless_enabled = True
        model.star_reduce_enabled = True
        model.star_reduce_factor = 0.3
        pipeline = builder.build_processing_pipeline(model)
        step = next(s for s in pipeline._steps if isinstance(s, StarRemovalStep))
        assert step._reduce_enabled is True
        assert step._reduce_factor == pytest.approx(0.3)

    def test_channel_combine_uses_model_mode(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.channel_combine_enabled = True
        model.channel_combine_mode = "narrowband"
        pipeline = builder.build_processing_pipeline(model)
        step = next(s for s in pipeline._steps if isinstance(s, ChannelCombineStep))
        assert step._mode is CombineMode.NARROWBAND

    def test_unknown_background_method_falls_back_to_rbf(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.background_removal_enabled = True
        model._background_removal_method = "unknown"
        pipeline = builder.build_processing_pipeline(model)
        assert any(isinstance(s, BackgroundRemovalStep) for s in pipeline._steps)

    def test_stretch_before_denoise(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_processing_pipeline(model)
        types = [type(s) for s in pipeline._steps]
        assert types.index(StretchStep) < types.index(DenoiseStep)

    def test_all_optional_steps_included_together(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.background_removal_enabled = True
        model.color_calibration_enabled = True
        model.deconvolution_enabled = True
        model.starless_enabled = True
        model.channel_combine_enabled = True
        pipeline = builder.build_processing_pipeline(model)
        step_types = {type(s) for s in pipeline._steps}
        assert BackgroundRemovalStep in step_types
        assert ColorCalibrationStep in step_types
        assert DeconvolutionStep in step_types
        assert StarRemovalStep in step_types
        assert ChannelCombineStep in step_types
        assert StretchStep in step_types
        assert DenoiseStep in step_types


class TestBuildExportStep:
    def test_returns_export_step(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        step = builder.build_export_step(model, Path("/tmp/out"))
        assert isinstance(step, ExportStep)

    def test_default_format_is_fits(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        step = builder.build_export_step(model, Path("/tmp/out"))
        assert step._format is ExportFormat.FITS

    def test_format_xisf_when_set(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.output_format = "xisf"
        step = builder.build_export_step(model, Path("/tmp/out"))
        assert step._format is ExportFormat.XISF

    def test_starless_export_when_enabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.starless_enabled = True
        step = builder.build_export_step(model, Path("/tmp/out"))
        assert step._export_starless is True

    def test_no_starless_export_when_disabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.starless_enabled = False
        step = builder.build_export_step(model, Path("/tmp/out"))
        assert step._export_starless is False

    def test_custom_output_dir_is_stored(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        out = Path("/custom/output")
        step = builder.build_export_step(model, out)
        assert step._output_dir == out


class TestBuildRegistrationPipeline:
    def test_returns_pipeline_with_registration_step(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_registration_pipeline(model)
        assert len(pipeline._steps) == 1
        assert isinstance(pipeline._steps[0], RegistrationStep)

    def test_uses_model_upsample_factor(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.registration_upsample_factor = 25
        pipeline = builder.build_registration_pipeline(model)
        step = pipeline._steps[0]
        assert step._aligner.upsample_factor == 25

    def test_uses_model_reference_frame_index(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.registration_reference_frame_index = 3
        pipeline = builder.build_registration_pipeline(model)
        step = pipeline._steps[0]
        assert step._ref_index == 3

    def test_default_upsample_factor(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_registration_pipeline(model)
        step = pipeline._steps[0]
        assert step._aligner.upsample_factor == 10

    def test_default_ref_index(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_registration_pipeline(model)
        step = pipeline._steps[0]
        assert step._ref_index == 0


class TestBuildStackingPipeline:
    def test_returns_pipeline_with_two_steps(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_stacking_pipeline(model)
        assert len(pipeline._steps) == 2

    def test_registration_step_first(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_stacking_pipeline(model)
        assert isinstance(pipeline._steps[0], RegistrationStep)

    def test_stacking_step_second(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_stacking_pipeline(model)
        assert isinstance(pipeline._steps[1], StackingStep)

    def test_uses_model_stacking_method(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.stacking_method = "mean"
        pipeline = builder.build_stacking_pipeline(model)
        step = pipeline._steps[1]
        assert step._method == "mean"

    def test_uses_model_sigma_low(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.stacking_sigma_low = 1.5
        pipeline = builder.build_stacking_pipeline(model)
        step = pipeline._steps[1]
        assert step._sigma_low == pytest.approx(1.5)

    def test_uses_model_sigma_high(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.stacking_sigma_high = 3.5
        pipeline = builder.build_stacking_pipeline(model)
        step = pipeline._steps[1]
        assert step._sigma_high == pytest.approx(3.5)

    def test_uses_model_upsample_factor(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.registration_upsample_factor = 15
        pipeline = builder.build_stacking_pipeline(model)
        reg_step = pipeline._steps[0]
        assert reg_step._aligner.upsample_factor == 15

    def test_default_method_is_sigma_clip(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_stacking_pipeline(model)
        step = pipeline._steps[1]
        assert step._method == "sigma_clip"


class TestBuildFullPipeline:
    def test_includes_load_frames_step_first(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        from astroai.core.pipeline.load_frames_step import LoadFramesStep

        pipeline = builder.build_full_pipeline(model, [])
        assert isinstance(pipeline._steps[0], LoadFramesStep)

    def test_includes_registration_step(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_full_pipeline(model, [])
        assert any(isinstance(s, RegistrationStep) for s in pipeline._steps)

    def test_includes_stacking_step(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_full_pipeline(model, [])
        assert any(isinstance(s, StackingStep) for s in pipeline._steps)

    def test_includes_stretch_and_denoise(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_full_pipeline(model, [])
        types = {type(s) for s in pipeline._steps}
        assert StretchStep in types
        assert DenoiseStep in types

    def test_load_step_has_given_paths(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        from astroai.core.pipeline.load_frames_step import LoadFramesStep

        paths = [Path("/tmp/a.fits"), Path("/tmp/b.fits")]
        pipeline = builder.build_full_pipeline(model, paths)
        load_step = pipeline._steps[0]
        assert isinstance(load_step, LoadFramesStep)
        assert load_step._paths == paths

    def test_step_order_load_before_registration(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        from astroai.core.pipeline.load_frames_step import LoadFramesStep

        pipeline = builder.build_full_pipeline(model, [])
        types = [type(s) for s in pipeline._steps]
        assert types.index(LoadFramesStep) < types.index(RegistrationStep)

    def test_step_order_registration_before_stacking(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        pipeline = builder.build_full_pipeline(model, [])
        types = [type(s) for s in pipeline._steps]
        assert types.index(RegistrationStep) < types.index(StackingStep)

    def test_frame_selection_included_when_enabled(
        self, builder: PipelineBuilder, model: PipelineModel
    ) -> None:
        model.frame_selection_enabled = True
        pipeline = builder.build_full_pipeline(model, [])
        assert any(isinstance(s, FrameSelectionStep) for s in pipeline._steps)
