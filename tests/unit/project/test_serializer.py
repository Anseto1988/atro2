"""Tests for ProjectSerializer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from astroai.project.project_file import AstroProject, FrameEntry, DenoiseConfig
from astroai.project.serializer import ProjectSerializer, ProjectSerializerError


@pytest.fixture
def tmp_project_path(tmp_path: Path) -> Path:
    return tmp_path / "test.astroai"


class TestProjectSerializer:
    def test_save_creates_file(self, tmp_project_path: Path):
        proj = AstroProject()
        ProjectSerializer.save(proj, tmp_project_path)
        assert tmp_project_path.exists()

    def test_save_valid_json(self, tmp_project_path: Path):
        proj = AstroProject()
        ProjectSerializer.save(proj, tmp_project_path)
        data = json.loads(tmp_project_path.read_text(encoding="utf-8"))
        assert data["metadata"]["version"] == "1.0"

    def test_load_roundtrip(self, tmp_project_path: Path):
        proj = AstroProject(
            input_frames=[FrameEntry(path="C:/images/m42.fits", exposure=120.0)],
            denoise=DenoiseConfig(strength=0.6),
        )
        ProjectSerializer.save(proj, tmp_project_path)
        loaded = ProjectSerializer.load(tmp_project_path)
        assert loaded.input_frames[0].path == "C:/images/m42.fits"
        assert loaded.denoise.strength == 0.6

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(ProjectSerializerError, match="nicht gefunden"):
            ProjectSerializer.load(tmp_path / "nope.astroai")

    def test_load_invalid_json_raises(self, tmp_project_path: Path):
        tmp_project_path.write_text("not json!", encoding="utf-8")
        with pytest.raises(ProjectSerializerError, match="Lesen fehlgeschlagen"):
            ProjectSerializer.load(tmp_project_path)

    def test_load_incompatible_version_raises(self, tmp_project_path: Path):
        data = AstroProject().to_dict()
        data["metadata"]["version"] = "99.0"
        tmp_project_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ProjectSerializerError, match="Inkompatible"):
            ProjectSerializer.load(tmp_project_path)

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "project.astroai"
        ProjectSerializer.save(AstroProject(), deep_path)
        assert deep_path.exists()
