"""Tests for AstroProject data model."""
from __future__ import annotations

import pytest
from astroai.project.project_file import (
    AstroProject,
    CalibrationConfig,
    ChannelCombineConfig,
    ColorCalibrationConfig,
    CometStackConfig,
    DeconvolutionConfig,
    DenoiseConfig,
    DrizzleConfig,
    FrameEntry,
    MosaicConfig,
    ProjectMetadata,
    RegistrationConfig,
    StackingConfig,
    StarProcessingConfig,
    StarlessConfig,
    StretchConfig,
    SyntheticFlatConfig,
    FrameSelectionConfig,
    BackgroundRemovalConfig,
    AnnotationConfig,
    PROJECT_FILE_VERSION,
)


class TestProjectMetadata:
    def test_default_version(self):
        meta = ProjectMetadata()
        assert meta.version == PROJECT_FILE_VERSION

    def test_timestamps_populated(self):
        meta = ProjectMetadata()
        assert meta.created_at
        assert meta.modified_at


class TestAstroProject:
    def test_default_construction(self):
        proj = AstroProject()
        assert proj.metadata.version == PROJECT_FILE_VERSION
        assert proj.input_frames == []
        assert proj.output_format == "fits"

    def test_touch_updates_modified_at(self):
        proj = AstroProject()
        old = proj.metadata.modified_at
        import time
        time.sleep(0.01)
        proj.touch()
        assert proj.metadata.modified_at >= old

    def test_to_dict_roundtrip(self):
        proj = AstroProject(
            input_frames=[FrameEntry(path="/tmp/test.fits", exposure=30.0, selected=True)],
            stacking=StackingConfig(method="median", sigma_low=3.0, sigma_high=3.0),
            denoise=DenoiseConfig(strength=0.8, model_type="pytorch"),
        )
        data = proj.to_dict()
        restored = AstroProject.from_dict(data)
        assert restored.input_frames[0].path == "/tmp/test.fits"
        assert restored.input_frames[0].exposure == 30.0
        assert restored.stacking.method == "median"
        assert restored.denoise.strength == 0.8

    def test_frame_entry_notes_default_empty(self) -> None:
        entry = FrameEntry(path="/tmp/frame.fits")
        assert entry.notes == ""

    def test_frame_entry_notes_roundtrip(self) -> None:
        proj = AstroProject(
            input_frames=[FrameEntry(path="/tmp/f.fits", notes="cloudy pass")]
        )
        data = proj.to_dict()
        restored = AstroProject.from_dict(data)
        assert restored.input_frames[0].notes == "cloudy pass"

    def test_frame_entry_notes_backward_compat(self) -> None:
        data = {"input_frames": [{"path": "/tmp/old.fits", "selected": True}]}
        proj = AstroProject.from_dict(data)
        assert proj.input_frames[0].notes == ""

    def test_from_dict_missing_keys_uses_defaults(self):
        proj = AstroProject.from_dict({})
        assert proj.metadata.version == PROJECT_FILE_VERSION
        assert proj.stretch.target_background == 0.25

    def test_calibration_config_defaults(self):
        cfg = CalibrationConfig()
        assert cfg.dark_frames == []
        assert cfg.flat_frames == []
        assert cfg.bias_frames == []

    def test_stretch_config_values(self):
        cfg = StretchConfig(target_background=0.3, linked_channels=False)
        assert cfg.target_background == 0.3
        assert cfg.linked_channels is False

    def test_star_processing_config(self):
        cfg = StarProcessingConfig(reduce_enabled=True, reduce_factor=0.7)
        assert cfg.reduce_enabled is True
        assert cfg.detection_sigma == 4.0

    def test_to_dict_includes_new_configs(self):
        proj = AstroProject()
        data = proj.to_dict()
        assert "drizzle" in data
        assert "mosaic" in data
        assert "channel_combine" in data
        assert "color_calibration" in data
        assert "deconvolution" in data
        assert "starless" in data
        assert "synthetic_flat" in data
        assert "comet_stack" in data
        assert "frame_selection" in data
        assert "background_removal" in data

    def test_drizzle_config_defaults(self):
        cfg = DrizzleConfig()
        assert cfg.enabled is False
        assert cfg.drop_size == pytest.approx(0.7)
        assert cfg.scale == pytest.approx(1.0)
        assert cfg.pixfrac == pytest.approx(1.0)

    def test_mosaic_config_defaults(self):
        cfg = MosaicConfig()
        assert cfg.enabled is False
        assert cfg.blend_mode == "average"
        assert cfg.gradient_correct is True
        assert cfg.panels == []

    def test_channel_combine_config_defaults(self):
        cfg = ChannelCombineConfig()
        assert cfg.enabled is False
        assert cfg.mode == "lrgb"
        assert cfg.palette == "SHO"

    def test_color_calibration_config_defaults(self):
        cfg = ColorCalibrationConfig()
        assert cfg.enabled is False
        assert cfg.catalog == "gaia_dr3"
        assert cfg.sample_radius == 8

    def test_deconvolution_config_defaults(self):
        cfg = DeconvolutionConfig()
        assert cfg.enabled is False
        assert cfg.iterations == 10
        assert cfg.psf_sigma == pytest.approx(1.0)

    def test_starless_config_defaults(self):
        cfg = StarlessConfig()
        assert cfg.enabled is False
        assert cfg.strength == pytest.approx(1.0)
        assert cfg.format == "xisf"
        assert cfg.save_star_mask is True

    def test_from_dict_missing_new_keys_uses_defaults(self):
        # Old-format project files without new config keys should deserialize cleanly
        data: dict = {
            "metadata": {"version": "1.0"},
            "stacking": {"method": "median", "sigma_low": 2.0, "sigma_high": 2.0},
        }
        proj = AstroProject.from_dict(data)
        assert proj.drizzle.enabled is False
        assert proj.mosaic.blend_mode == "average"
        assert proj.channel_combine.mode == "lrgb"
        assert proj.color_calibration.catalog == "gaia_dr3"
        assert proj.deconvolution.iterations == 10
        assert proj.starless.format == "xisf"
        assert proj.synthetic_flat.enabled is False
        assert proj.synthetic_flat.tile_size == 64
        assert proj.comet_stack.enabled is False
        assert proj.comet_stack.tracking_mode == "blend"
        assert proj.frame_selection.enabled is False
        assert proj.frame_selection.min_score == pytest.approx(0.5)
        assert proj.background_removal.enabled is False
        assert proj.background_removal.method == "rbf"

    def test_full_roundtrip_with_new_configs(self):
        proj = AstroProject(
            drizzle=DrizzleConfig(enabled=True, drop_size=0.5, scale=2.0, pixfrac=0.9),
            mosaic=MosaicConfig(enabled=True, blend_mode="linear", panels=["/img/a.fits"]),
            channel_combine=ChannelCombineConfig(enabled=True, mode="narrowband", palette="HOO"),
            color_calibration=ColorCalibrationConfig(enabled=True, catalog="gaia_dr3", sample_radius=12),
            deconvolution=DeconvolutionConfig(enabled=True, iterations=20, psf_sigma=1.5),
            starless=StarlessConfig(enabled=True, strength=0.8, format="fits", save_star_mask=False),
            synthetic_flat=SyntheticFlatConfig(enabled=True, tile_size=128, smoothing_sigma=5.0),
            comet_stack=CometStackConfig(enabled=True, tracking_mode="comet", blend_factor=0.3),
        )
        data = proj.to_dict()
        restored = AstroProject.from_dict(data)
        assert restored.drizzle.drop_size == pytest.approx(0.5)
        assert restored.drizzle.scale == pytest.approx(2.0)
        assert restored.mosaic.blend_mode == "linear"
        assert restored.mosaic.panels == ["/img/a.fits"]
        assert restored.channel_combine.mode == "narrowband"
        assert restored.channel_combine.palette == "HOO"
        assert restored.color_calibration.sample_radius == 12
        assert restored.deconvolution.iterations == 20
        assert restored.starless.strength == pytest.approx(0.8)
        assert restored.starless.save_star_mask is False
        assert restored.synthetic_flat.enabled is True
        assert restored.synthetic_flat.tile_size == 128
        assert restored.synthetic_flat.smoothing_sigma == pytest.approx(5.0)
        assert restored.comet_stack.enabled is True
        assert restored.comet_stack.tracking_mode == "comet"
        assert restored.comet_stack.blend_factor == pytest.approx(0.3)

    def test_synthetic_flat_config_defaults(self):
        cfg = SyntheticFlatConfig()
        assert cfg.enabled is False
        assert cfg.tile_size == 64
        assert cfg.smoothing_sigma == pytest.approx(8.0)

    def test_comet_stack_config_defaults(self):
        cfg = CometStackConfig()
        assert cfg.enabled is False
        assert cfg.tracking_mode == "blend"
        assert cfg.blend_factor == pytest.approx(0.5)

    def test_frame_selection_config_defaults(self):
        cfg = FrameSelectionConfig()
        assert cfg.enabled is False
        assert cfg.min_score == pytest.approx(0.5)
        assert cfg.max_rejected_fraction == pytest.approx(0.8)

    def test_background_removal_config_defaults(self):
        cfg = BackgroundRemovalConfig()
        assert cfg.enabled is False
        assert cfg.tile_size == 64
        assert cfg.method == "rbf"
        assert cfg.preserve_median is True

    def test_registration_config_defaults(self):
        cfg = RegistrationConfig()
        assert cfg.enabled is True
        assert cfg.upsample_factor == 10
        assert cfg.reference_frame_index == 0

    def test_registration_roundtrip(self):
        proj = AstroProject(
            registration=RegistrationConfig(upsample_factor=50, reference_frame_index=2),
        )
        data = proj.to_dict()
        restored = AstroProject.from_dict(data)
        assert restored.registration.upsample_factor == 50
        assert restored.registration.reference_frame_index == 2

    def test_registration_included_in_to_dict(self):
        proj = AstroProject()
        data = proj.to_dict()
        assert "registration" in data


class TestAnnotationConfig:
    def test_annotation_defaults(self):
        cfg = AnnotationConfig()
        assert cfg.show_dso is True
        assert cfg.show_stars is True
        assert cfg.show_boundaries is False
        assert cfg.show_grid is False

    def test_annotation_roundtrip(self):
        proj = AstroProject(
            annotation=AnnotationConfig(
                show_dso=False, show_stars=True, show_boundaries=True, show_grid=True
            ),
        )
        data = proj.to_dict()
        restored = AstroProject.from_dict(data)
        assert restored.annotation.show_dso is False
        assert restored.annotation.show_stars is True
        assert restored.annotation.show_boundaries is True
        assert restored.annotation.show_grid is True

    def test_annotation_included_in_to_dict(self):
        proj = AstroProject()
        data = proj.to_dict()
        assert "annotation" in data

    def test_annotation_defaults_for_legacy_files(self):
        data = AstroProject().to_dict()
        del data["annotation"]
        restored = AstroProject.from_dict(data)
        assert restored.annotation.show_dso is True
        assert restored.annotation.show_boundaries is False


class TestAstroProjectDirty:
    def test_new_project_is_clean(self):
        proj = AstroProject()
        assert proj.is_dirty is False

    def test_touch_marks_dirty(self):
        proj = AstroProject()
        proj.touch()
        assert proj.is_dirty is True

    def test_mark_clean_clears_dirty(self):
        proj = AstroProject()
        proj.touch()
        proj.mark_clean()
        assert proj.is_dirty is False

    def test_touch_updates_modified_at(self):
        import time
        proj = AstroProject()
        before = proj.metadata.modified_at
        time.sleep(0.01)
        proj.touch()
        assert proj.metadata.modified_at > before

    def test_dirty_not_in_to_dict(self):
        proj = AstroProject()
        proj.touch()
        data = proj.to_dict()
        assert "_dirty" not in data
        assert "is_dirty" not in data

    def test_loaded_project_starts_clean(self):
        proj = AstroProject()
        proj.touch()
        data = proj.to_dict()
        restored = AstroProject.from_dict(data)
        assert restored.is_dirty is False

    def test_multiple_touch_still_dirty(self):
        proj = AstroProject()
        proj.touch()
        proj.touch()
        assert proj.is_dirty is True
