"""Tests for AstroProject data model."""
from __future__ import annotations

import pytest
from astroai.project.project_file import (
    AstroProject,
    CalibrationConfig,
    DenoiseConfig,
    FrameEntry,
    ProjectMetadata,
    RegistrationConfig,
    StackingConfig,
    StarProcessingConfig,
    StretchConfig,
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
