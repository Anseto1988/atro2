"""End-to-end integration test for project persistence.

Verifies: Pipeline config -> ProjectSerializer.save -> load -> roundtrip integrity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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
)
from astroai.project.serializer import ProjectSerializer, ProjectSerializerError


@pytest.fixture()
def complex_project() -> AstroProject:
    """Fully populated AstroProject simulating real workflow state."""
    return AstroProject(
        metadata=ProjectMetadata(
            name="NGC 7000 North America Nebula",
            description="Ha + OIII narrowband session 2024-08-15",
        ),
        input_frames=[
            FrameEntry(
                path="/data/lights/Ha_001.fits",
                exposure=300.0,
                gain_iso=100,
                temperature=-10.0,
                quality_score=0.92,
                selected=True,
            ),
            FrameEntry(
                path="/data/lights/Ha_002.fits",
                exposure=300.0,
                gain_iso=100,
                temperature=-10.5,
                quality_score=0.88,
                selected=True,
            ),
            FrameEntry(
                path="/data/lights/Ha_003.fits",
                exposure=300.0,
                gain_iso=100,
                temperature=-9.8,
                quality_score=0.45,
                selected=False,
            ),
        ],
        calibration=CalibrationConfig(
            dark_frames=["/data/darks/dark_001.fits", "/data/darks/dark_002.fits"],
            flat_frames=["/data/flats/flat_001.fits"],
            bias_frames=["/data/bias/bias_001.fits"],
        ),
        registration=RegistrationConfig(
            enabled=True,
            reference_frame_index=0,
            upsample_factor=20,
        ),
        stacking=StackingConfig(
            method="sigma_clip",
            sigma_low=2.0,
            sigma_high=3.0,
        ),
        stretch=StretchConfig(
            enabled=True,
            target_background=0.20,
            shadow_clipping_sigmas=-3.0,
            linked_channels=False,
        ),
        denoise=DenoiseConfig(
            enabled=True,
            strength=0.8,
            tile_size=256,
            tile_overlap=32,
            model_type="onnx",
        ),
        star_processing=StarProcessingConfig(
            reduce_enabled=True,
            reduce_factor=0.7,
            detection_sigma=3.5,
            min_area=5,
            max_area=3000,
            mask_dilation=4,
        ),
        output_path="/data/output/ngc7000_final.fits",
        output_format="fits",
    )


class TestProjectPersistenceE2E:
    """Integration tests for project save/load roundtrip."""

    def test_save_load_roundtrip(self, complex_project: AstroProject) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_project.astroai"
            ProjectSerializer.save(complex_project, path)
            loaded = ProjectSerializer.load(path)

        assert loaded.metadata.name == complex_project.metadata.name
        assert loaded.metadata.description == complex_project.metadata.description
        assert len(loaded.input_frames) == 3
        assert loaded.output_path == "/data/output/ngc7000_final.fits"

    def test_roundtrip_frame_entries(self, complex_project: AstroProject) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "project.astroai"
            ProjectSerializer.save(complex_project, path)
            loaded = ProjectSerializer.load(path)

        for original, restored in zip(
            complex_project.input_frames, loaded.input_frames
        ):
            assert restored.path == original.path
            assert restored.exposure == original.exposure
            assert restored.gain_iso == original.gain_iso
            assert restored.temperature == original.temperature
            assert restored.quality_score == original.quality_score
            assert restored.selected == original.selected

    def test_roundtrip_calibration_config(
        self, complex_project: AstroProject
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "project.astroai"
            ProjectSerializer.save(complex_project, path)
            loaded = ProjectSerializer.load(path)

        assert loaded.calibration.dark_frames == complex_project.calibration.dark_frames
        assert loaded.calibration.flat_frames == complex_project.calibration.flat_frames
        assert loaded.calibration.bias_frames == complex_project.calibration.bias_frames

    def test_roundtrip_processing_config(
        self, complex_project: AstroProject
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "project.astroai"
            ProjectSerializer.save(complex_project, path)
            loaded = ProjectSerializer.load(path)

        assert loaded.stacking.method == "sigma_clip"
        assert loaded.stacking.sigma_low == 2.0
        assert loaded.stacking.sigma_high == 3.0
        assert loaded.stretch.target_background == 0.20
        assert loaded.stretch.linked_channels is False
        assert loaded.denoise.strength == 0.8
        assert loaded.denoise.tile_size == 256
        assert loaded.star_processing.reduce_enabled is True
        assert loaded.star_processing.reduce_factor == 0.7

    def test_roundtrip_registration_config(
        self, complex_project: AstroProject
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "project.astroai"
            ProjectSerializer.save(complex_project, path)
            loaded = ProjectSerializer.load(path)

        assert loaded.registration.enabled is True
        assert loaded.registration.reference_frame_index == 0
        assert loaded.registration.upsample_factor == 20

    def test_touch_updates_modified_at(self, complex_project: AstroProject) -> None:
        original_modified = complex_project.metadata.modified_at
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "project.astroai"
            ProjectSerializer.save(complex_project, path)
            loaded = ProjectSerializer.load(path)

        assert loaded.metadata.modified_at >= original_modified

    def test_save_creates_parent_directories(self) -> None:
        project = AstroProject(metadata=ProjectMetadata(name="Test"))
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c" / "project.astroai"
            ProjectSerializer.save(project, nested)
            assert nested.exists()
            loaded = ProjectSerializer.load(nested)
            assert loaded.metadata.name == "Test"

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(ProjectSerializerError, match="nicht gefunden"):
            ProjectSerializer.load(Path("/nonexistent/path.astroai"))

    def test_default_project_roundtrip(self) -> None:
        project = AstroProject()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "default.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)

        assert loaded.metadata.name == "Untitled"
        assert loaded.input_frames == []
        assert loaded.output_format == "fits"

    def test_to_dict_from_dict_equivalence(
        self, complex_project: AstroProject
    ) -> None:
        data = complex_project.to_dict()
        restored = AstroProject.from_dict(data)

        assert restored.metadata.name == complex_project.metadata.name
        assert len(restored.input_frames) == len(complex_project.input_frames)
        assert restored.stacking.method == complex_project.stacking.method
        assert restored.denoise.strength == complex_project.denoise.strength
        assert restored.output_path == complex_project.output_path
