"""Tests for project validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from astroai.project.project_file import AstroProject, CalibrationConfig, FrameEntry
from astroai.project.validator import ValidationIssue, ValidationResult, validate_project


def _project(*frames: FrameEntry) -> AstroProject:
    proj = AstroProject()
    proj.input_frames = list(frames)
    return proj


def _frame(path: str = "light.fits", selected: bool = True, exists: bool = True) -> FrameEntry:
    return FrameEntry(path=path, selected=selected)


class TestValidationResult:
    def test_empty_result_has_no_errors(self) -> None:
        vr = ValidationResult()
        assert not vr.has_errors
        assert not vr.has_warnings

    def test_errors_property_filters(self) -> None:
        vr = ValidationResult(
            issues=[
                ValidationIssue("error", "E1", "err"),
                ValidationIssue("warning", "W1", "warn"),
            ]
        )
        assert len(vr.errors) == 1
        assert len(vr.warnings) == 1

    def test_summary_ok_when_empty(self) -> None:
        assert ValidationResult().summary() == "Projekt OK"

    def test_summary_counts_issues(self) -> None:
        vr = ValidationResult(
            issues=[
                ValidationIssue("error", "E1", "e"),
                ValidationIssue("warning", "W1", "w"),
                ValidationIssue("warning", "W2", "w"),
            ]
        )
        s = vr.summary()
        assert "1 Fehler" in s
        assert "2 Warnung" in s


class TestValidateProjectErrors:
    def test_invalid_object_gives_error(self) -> None:
        result = validate_project(object())
        assert result.has_errors
        assert result.errors[0].code == "INVALID_PROJECT"

    def test_no_frames_gives_no_frames_error(self) -> None:
        result = validate_project(_project())
        assert any(i.code == "NO_FRAMES" for i in result.errors)

    def test_all_deselected_gives_no_selected_error(self) -> None:
        proj = _project(_frame(selected=False), _frame(selected=False))
        result = validate_project(proj)
        assert any(i.code == "NO_SELECTED_FRAMES" for i in result.errors)

    def test_one_selected_no_error(self, tmp_path: Path) -> None:
        f = tmp_path / "light.fits"
        f.write_bytes(b"")
        proj = _project(FrameEntry(path=str(f), selected=True))
        result = validate_project(proj)
        assert not result.has_errors


class TestValidateProjectWarnings:
    def test_missing_light_file_gives_warning(self) -> None:
        proj = _project(FrameEntry(path="/nonexistent/light.fits", selected=True))
        result = validate_project(proj)
        assert any(i.code == "MISSING_LIGHT_FILES" for i in result.warnings)

    def test_missing_light_file_detail_truncated_at_3(self) -> None:
        frames = [FrameEntry(path=f"/nope/l{i}.fits", selected=True) for i in range(5)]
        result = validate_project(_project(*frames))
        issue = next(i for i in result.issues if i.code == "MISSING_LIGHT_FILES")
        assert "+2 weitere" in issue.detail

    def test_missing_dark_gives_warning(self, tmp_path: Path) -> None:
        f = tmp_path / "light.fits"
        f.write_bytes(b"")
        proj = _project(FrameEntry(path=str(f), selected=True))
        proj.calibration = CalibrationConfig(dark_frames=["/missing/dark.fits"])
        result = validate_project(proj)
        assert any(i.code == "MISSING_DARK_FILES" for i in result.warnings)

    def test_missing_flat_gives_warning(self, tmp_path: Path) -> None:
        f = tmp_path / "light.fits"
        f.write_bytes(b"")
        proj = _project(FrameEntry(path=str(f), selected=True))
        proj.calibration = CalibrationConfig(flat_frames=["/missing/flat.fits"])
        result = validate_project(proj)
        assert any(i.code == "MISSING_FLAT_FILES" for i in result.warnings)

    def test_missing_bias_gives_warning(self, tmp_path: Path) -> None:
        f = tmp_path / "light.fits"
        f.write_bytes(b"")
        proj = _project(FrameEntry(path=str(f), selected=True))
        proj.calibration = CalibrationConfig(bias_frames=["/missing/bias.fits"])
        result = validate_project(proj)
        assert any(i.code == "MISSING_BIAS_FILES" for i in result.warnings)

    def test_existing_calib_files_no_warning(self, tmp_path: Path) -> None:
        light = tmp_path / "light.fits"
        dark = tmp_path / "dark.fits"
        light.write_bytes(b"")
        dark.write_bytes(b"")
        proj = _project(FrameEntry(path=str(light), selected=True))
        proj.calibration = CalibrationConfig(dark_frames=[str(dark)])
        result = validate_project(proj)
        assert not any(i.code == "MISSING_DARK_FILES" for i in result.issues)

    def test_output_dir_missing_gives_warning(self, tmp_path: Path) -> None:
        f = tmp_path / "light.fits"
        f.write_bytes(b"")
        proj = _project(FrameEntry(path=str(f), selected=True))
        proj.output_path = str(tmp_path / "nonexistent_dir" / "out.fits")
        result = validate_project(proj)
        assert any(i.code == "OUTPUT_DIR_MISSING" for i in result.warnings)


class TestValidateProjectClean:
    def test_valid_project_no_issues(self, tmp_path: Path) -> None:
        f = tmp_path / "light.fits"
        f.write_bytes(b"")
        proj = _project(FrameEntry(path=str(f), selected=True))
        result = validate_project(proj)
        assert not result.has_errors
        assert not result.has_warnings
        assert result.summary() == "Projekt OK"

    def test_partial_selection_ok(self, tmp_path: Path) -> None:
        f1 = tmp_path / "l1.fits"
        f1.write_bytes(b"")
        proj = _project(
            FrameEntry(path=str(f1), selected=True),
            FrameEntry(path="/nope/l2.fits", selected=False),
        )
        result = validate_project(proj)
        # deselected missing file should NOT trigger a warning
        assert not any(i.code == "MISSING_LIGHT_FILES" for i in result.issues)
