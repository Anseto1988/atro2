"""End-to-end integration test for project persistence.

Verifies: Pipeline config -> ProjectSerializer.save -> load -> roundtrip integrity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from astroai.project.project_file import (
    AnnotationConfig,
    AstroProject,
    CalibrationConfig,
    CurvesConfig,
    DenoiseConfig,
    FrameEntry,
    FrameSelectionConfig,
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

    def test_roundtrip_frame_selection_config(self) -> None:
        project = AstroProject(
            frame_selection=FrameSelectionConfig(
                enabled=True,
                min_score=0.65,
                max_rejected_fraction=0.5,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "project.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)

        assert loaded.frame_selection.enabled is True
        assert loaded.frame_selection.min_score == pytest.approx(0.65)
        assert loaded.frame_selection.max_rejected_fraction == pytest.approx(0.5)

    def test_frame_selection_defaults_for_legacy_files(self) -> None:
        # Simulate a legacy project file without frame_selection key
        data: dict = {
            "metadata": {"version": "1.0"},
            "stacking": {"method": "median", "sigma_low": 2.5, "sigma_high": 2.5},
        }
        project = AstroProject.from_dict(data)
        assert project.frame_selection.enabled is False
        assert project.frame_selection.min_score == pytest.approx(0.5)


class TestAnnotationPersistenceE2E:
    """Integration tests: AnnotationConfig roundtrip via ProjectSerializer."""

    def test_annotation_save_load_roundtrip(self) -> None:
        project = AstroProject(
            annotation=AnnotationConfig(
                show_dso=False,
                show_stars=True,
                show_boundaries=True,
                show_grid=True,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotation_test.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)

        assert loaded.annotation.show_dso is False
        assert loaded.annotation.show_stars is True
        assert loaded.annotation.show_boundaries is True
        assert loaded.annotation.show_grid is True

    def test_annotation_defaults_preserved_in_roundtrip(self) -> None:
        project = AstroProject()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "default_annotation.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)

        defaults = AnnotationConfig()
        assert loaded.annotation.show_dso == defaults.show_dso
        assert loaded.annotation.show_stars == defaults.show_stars
        assert loaded.annotation.show_boundaries == defaults.show_boundaries
        assert loaded.annotation.show_grid == defaults.show_grid

    def test_annotation_absent_in_legacy_file_uses_defaults(self) -> None:
        data = AstroProject().to_dict()
        del data["annotation"]
        project = AstroProject.from_dict(data)

        defaults = AnnotationConfig()
        assert project.annotation.show_dso == defaults.show_dso
        assert project.annotation.show_boundaries == defaults.show_boundaries

    def test_complex_project_with_annotation_roundtrip(self) -> None:
        project = AstroProject(
            metadata=ProjectMetadata(name="M42 Test"),
            registration=RegistrationConfig(
                upsample_factor=20, reference_frame_index=1
            ),
            stacking=StackingConfig(method="median"),
            annotation=AnnotationConfig(
                show_dso=True,
                show_stars=False,
                show_boundaries=False,
                show_grid=True,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "complex.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)

        assert loaded.annotation.show_dso is True
        assert loaded.annotation.show_stars is False
        assert loaded.annotation.show_grid is True
        assert loaded.registration.upsample_factor == 20
        assert loaded.stacking.method == "median"

    def test_all_annotation_flags_false_roundtrip(self) -> None:
        project = AstroProject(
            annotation=AnnotationConfig(
                show_dso=False,
                show_stars=False,
                show_boundaries=False,
                show_grid=False,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "all_false.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)

        assert loaded.annotation.show_dso is False
        assert loaded.annotation.show_stars is False
        assert loaded.annotation.show_boundaries is False
        assert loaded.annotation.show_grid is False


class TestSessionNotesPersistenceE2E:
    def test_description_roundtrip(self) -> None:
        project = AstroProject(
            metadata=ProjectMetadata(description="Beobachtungsnacht 2026-04-28, Seeing 4/5")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)
        assert loaded.metadata.description == "Beobachtungsnacht 2026-04-28, Seeing 4/5"

    def test_empty_description_roundtrip(self) -> None:
        project = AstroProject(metadata=ProjectMetadata(description=""))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)
        assert loaded.metadata.description == ""

    def test_multiline_description_roundtrip(self) -> None:
        text = "Zeile 1\nZeile 2\nSonderzeichen: äöü — ★"
        project = AstroProject(metadata=ProjectMetadata(description=text))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multiline.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)
        assert loaded.metadata.description == text

    def test_description_default_is_empty(self) -> None:
        project = AstroProject()
        assert project.metadata.description == ""

    def test_description_survives_complex_project(self) -> None:
        project = AstroProject(
            metadata=ProjectMetadata(name="M42 Session", description="Orion Nebula, 42x120s"),
            registration=RegistrationConfig(upsample_factor=20),
            stacking=StackingConfig(method="median"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "complex_notes.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)
        assert loaded.metadata.description == "Orion Nebula, 42x120s"
        assert loaded.metadata.name == "M42 Session"
        assert loaded.registration.upsample_factor == 20


class TestCurvesConfigPersistenceE2E:
    def test_enabled_flag_roundtrip(self) -> None:
        project = AstroProject(curves=CurvesConfig(enabled=True))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curves.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)
        assert loaded.curves.enabled is True

    def test_disabled_by_default(self) -> None:
        project = AstroProject()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curves_default.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)
        assert loaded.curves.enabled is False

    def test_rgb_points_roundtrip(self) -> None:
        pts = [[0.0, 0.0], [0.25, 0.4], [0.75, 0.6], [1.0, 1.0]]
        project = AstroProject(curves=CurvesConfig(enabled=True, rgb_points=pts))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curves_pts.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)
        assert loaded.curves.rgb_points == pts

    def test_per_channel_points_roundtrip(self) -> None:
        r_pts = [[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]]
        g_pts = [[0.0, 0.0], [0.5, 0.3], [1.0, 1.0]]
        b_pts = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        project = AstroProject(
            curves=CurvesConfig(
                enabled=True, r_points=r_pts, g_points=g_pts, b_points=b_pts
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curves_ch.astroai"
            ProjectSerializer.save(project, path)
            loaded = ProjectSerializer.load(path)
        assert loaded.curves.r_points == r_pts
        assert loaded.curves.g_points == g_pts
        assert loaded.curves.b_points == b_pts

    def test_legacy_file_missing_curves_uses_defaults(self) -> None:
        import json
        project_dict = AstroProject().to_dict()
        del project_dict["curves"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.astroai"
            path.write_text(json.dumps(project_dict), encoding="utf-8")
            loaded = ProjectSerializer.load(path)
        assert loaded.curves.enabled is False
        assert len(loaded.curves.rgb_points) == 2
