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

    def test_save_os_error_raises_serializer_error(self, tmp_path: Path) -> None:
        """OSError during write raises ProjectSerializerError (lines 23-24)."""
        from unittest.mock import patch
        proj = AstroProject()
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            with pytest.raises(ProjectSerializerError, match="Speichern fehlgeschlagen"):
                ProjectSerializer.save(proj, tmp_path / "proj.astroai")

    def test_is_compatible_non_numeric_version_returns_false(self) -> None:
        """Non-numeric version string returns False via ValueError (lines 49-50)."""
        assert ProjectSerializer._is_compatible("not.a.version") is False

    def test_is_compatible_empty_string_returns_false(self) -> None:
        """Empty version string returns False (lines 49-50)."""
        assert ProjectSerializer._is_compatible("") is False
