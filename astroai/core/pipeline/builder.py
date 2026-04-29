"""Factory that constructs a configured Pipeline from PipelineModel state."""
from __future__ import annotations

from pathlib import Path

from astroai.core.pipeline.base import Pipeline, PipelineStep
from astroai.core.pipeline.comet_stack_step import CometStackStep
from astroai.core.pipeline.export_step import ExportFormat, ExportStep
from astroai.engine.comet.stacker import TrackingMode
from astroai.core.pipeline.frame_selection_step import FrameSelectionStep
from astroai.engine.drizzle.pipeline_step import DrizzleStep
from astroai.engine.mosaic.pipeline_step import MosaicStep
from astroai.engine.registration.pipeline_step import RegistrationStep
from astroai.processing.background.extractor import ModelMethod
from astroai.processing.background.pipeline_step import BackgroundRemovalStep
from astroai.processing.channels.pipeline_step import ChannelCombineStep, CombineMode
from astroai.processing.channels.narrowband_mapper import NarrowbandPalette
from astroai.processing.color.calibrator import CatalogSource
from astroai.processing.color.pipeline_step import ColorCalibrationStep
from astroai.processing.deconvolution.pipeline_step import DeconvolutionStep
from astroai.processing.denoise.pipeline_step import DenoiseStep
from astroai.processing.flat.pipeline_step import SyntheticFlatStep
from astroai.processing.sharpening.pipeline_step import SharpeningStep
from astroai.processing.stars.pipeline_step import StarRemovalStep
from astroai.processing.curves.pipeline_step import CurvesStep
from astroai.processing.stretch.pipeline_step import StretchStep
from astroai.ui.models import PipelineModel

__all__ = ["PipelineBuilder"]

_TRACKING_MODE: dict[str, TrackingMode] = {
    "stars": "stars",
    "comet": "comet",
    "blend": "blend",
}

_BACKGROUND_METHOD: dict[str, ModelMethod] = {
    "rbf": ModelMethod.RBF,
    "poly": ModelMethod.POLYNOMIAL,
}

_COMBINE_MODE: dict[str, CombineMode] = {
    "lrgb": CombineMode.LRGB,
    "narrowband": CombineMode.NARROWBAND,
}

_NARROWBAND_PALETTE: dict[str, NarrowbandPalette] = {
    "sho": NarrowbandPalette.SHO,
    "hoo": NarrowbandPalette.HOO,
    "nho": NarrowbandPalette.NHO,
}

_CATALOG_SOURCE: dict[str, CatalogSource] = {
    "gaia_dr3": CatalogSource.GAIA_DR3,
    "2mass": CatalogSource.TWOMASS,
}

_EXPORT_FORMAT: dict[str, ExportFormat] = {
    "xisf": ExportFormat.XISF,
    "tiff": ExportFormat.TIFF32,
    "fits": ExportFormat.FITS,
}


class PipelineBuilder:
    """Reads PipelineModel config and instantiates concrete pipeline step objects."""

    def build_calibration_pipeline(self, model: PipelineModel) -> Pipeline:
        """Pre-stacking pipeline: optional frame selection + synthetic flat."""
        steps: list[PipelineStep] = []
        if model.frame_selection_enabled:
            steps.append(FrameSelectionStep(
                min_score=model.frame_selection_min_score,
                max_rejected_fraction=model.frame_selection_max_rejected_fraction,
            ))
        if model.synthetic_flat_enabled:
            steps.append(SyntheticFlatStep(
                tile_size=model.synthetic_flat_tile_size,
                smoothing_sigma=model.synthetic_flat_smoothing_sigma,
            ))
        return Pipeline(steps)

    def build_registration_pipeline(self, model: PipelineModel) -> Pipeline:
        """Registration pipeline: star-detection or phase-correlation."""
        step = RegistrationStep(
            upsample_factor=model.registration_upsample_factor,
            reference_frame_index=model.registration_reference_frame_index,
            method=model.registration_method,  # type: ignore[arg-type]
        )
        return Pipeline([step])

    def build_processing_pipeline(self, model: PipelineModel) -> Pipeline:
        """Post-stacking processing pipeline in correct execution order."""
        steps: list[PipelineStep] = []

        if model.drizzle_enabled:
            steps.append(DrizzleStep(
                drop_size=model.drizzle_drop_size,
                scale=model.drizzle_scale,
                pixfrac=model.drizzle_pixfrac,
            ))

        if model.mosaic_enabled:
            steps.append(MosaicStep(
                blend_mode=model.mosaic_blend_mode,
                gradient_correct=model.mosaic_gradient_correct,
                output_scale=model.mosaic_output_scale,
            ))

        if model.channel_combine_enabled:
            mode = _COMBINE_MODE.get(model.channel_combine_mode, CombineMode.LRGB)
            palette = _NARROWBAND_PALETTE.get(
                model.channel_combine_palette.lower(), NarrowbandPalette.SHO
            )
            steps.append(ChannelCombineStep(mode=mode, palette=palette))

        if model.comet_stack_enabled:
            tracking_mode = _TRACKING_MODE.get(model.comet_tracking_mode, "blend")
            steps.append(CometStackStep(
                tracking_mode=tracking_mode,
                blend_factor=model.comet_blend_factor,
            ))

        if model.background_removal_enabled:
            method = _BACKGROUND_METHOD.get(
                model.background_removal_method, ModelMethod.RBF
            )
            steps.append(BackgroundRemovalStep(
                tile_size=model.background_removal_tile_size,
                method=method,
                preserve_median=model.background_removal_preserve_median,
            ))

        steps.append(StretchStep(
            target_background=model.stretch_target_background,
            shadow_clipping_sigmas=model.stretch_shadow_clipping_sigmas,
            linked_channels=model.stretch_linked_channels,
        ))

        if model.curves_enabled:
            steps.append(CurvesStep(
                rgb_points=model.curves_rgb_points,
                r_points=model.curves_r_points,
                g_points=model.curves_g_points,
                b_points=model.curves_b_points,
            ))

        if model.color_calibration_enabled:
            catalog = _CATALOG_SOURCE.get(
                model.color_calibration_catalog, CatalogSource.GAIA_DR3
            )
            steps.append(ColorCalibrationStep(
                catalog=catalog,
                sample_radius_px=model.color_calibration_sample_radius,
            ))

        steps.append(DenoiseStep(
            strength=model.denoise_strength,
            tile_size=model.denoise_tile_size,
            tile_overlap=model.denoise_tile_overlap,
        ))

        if model.deconvolution_enabled:
            steps.append(DeconvolutionStep(
                iterations=model.deconvolution_iterations,
                psf_sigma=model.deconvolution_psf_sigma,
            ))

        if model.sharpening_enabled:
            steps.append(SharpeningStep(
                radius=model.sharpening_radius,
                amount=model.sharpening_amount,
                threshold=model.sharpening_threshold,
            ))

        if model.starless_enabled:
            steps.append(StarRemovalStep(
                detection_sigma=model.star_detection_sigma,
                min_star_area=model.star_min_area,
                max_star_area=model.star_max_area,
                mask_dilation=model.star_mask_dilation,
                reduce_enabled=model.star_reduce_enabled,
                reduce_factor=model.star_reduce_factor,
            ))

        return Pipeline(steps)

    def build_stacking_pipeline(self, model: PipelineModel) -> Pipeline:
        """Registration + stacking pipeline."""
        from astroai.engine.registration.pipeline_step import RegistrationStep
        from astroai.engine.stacking.pipeline_step import StackingStep

        steps: list[PipelineStep] = [
            RegistrationStep(
                upsample_factor=model.registration_upsample_factor,
                reference_frame_index=model.registration_reference_frame_index,
                method=model.registration_method,  # type: ignore[arg-type]
            ),
            StackingStep(
                method=model.stacking_method,
                sigma_low=model.stacking_sigma_low,
                sigma_high=model.stacking_sigma_high,
            ),
        ]
        return Pipeline(steps)

    def build_full_pipeline(
        self,
        model: PipelineModel,
        frame_paths: list[Path],
    ) -> Pipeline:
        """Complete pipeline: load frames → calibration → registration → stacking → processing."""
        from astroai.core.pipeline.load_frames_step import LoadFramesStep

        steps: list[PipelineStep] = [LoadFramesStep(frame_paths)]
        steps.extend(self.build_calibration_pipeline(model)._steps)
        steps.extend(self.build_stacking_pipeline(model)._steps)
        steps.extend(self.build_processing_pipeline(model)._steps)
        return Pipeline(steps)

    def build_export_step(
        self,
        model: PipelineModel,
        output_dir: Path,
        filename: str = "output",
    ) -> ExportStep:
        """Create an ExportStep from model config and the given output directory."""
        cfg = model.export_config()
        fmt = _EXPORT_FORMAT.get(str(cfg.get("fmt_value", "xisf")), ExportFormat.XISF)
        return ExportStep(
            output_dir=output_dir,
            fmt=fmt,
            filename=filename,
            export_starless=bool(cfg.get("export_starless", False)),
            export_star_mask=bool(cfg.get("export_star_mask", False)),
        )
