"""Tests for ProjectSummary computation."""
from __future__ import annotations

import pytest

from astroai.project.project_file import AstroProject, FrameEntry
from astroai.project.summary import ExposureGroup, ProjectSummary, compute_summary


def _project(*frames: FrameEntry) -> AstroProject:
    p = AstroProject()
    p.input_frames = list(frames)
    return p


class TestProjectSummaryProperties:
    def test_unselected_count(self) -> None:
        s = ProjectSummary(total_frames=10, selected_count=7)
        assert s.unselected_count == 3

    def test_total_exposure_hms_seconds_only(self) -> None:
        s = ProjectSummary(total_exposure_s=45.0)
        assert s.total_exposure_hms == "45s"

    def test_total_exposure_hms_minutes(self) -> None:
        s = ProjectSummary(total_exposure_s=125.0)
        assert "2m" in s.total_exposure_hms
        assert "5s" in s.total_exposure_hms

    def test_total_exposure_hms_hours(self) -> None:
        s = ProjectSummary(total_exposure_s=3661.0)
        assert "1h" in s.total_exposure_hms
        assert "1m" in s.total_exposure_hms

    def test_total_exposure_hms_zero(self) -> None:
        assert ProjectSummary().total_exposure_hms == "0s"


class TestComputeSummary:
    def test_empty_project(self) -> None:
        s = compute_summary(_project())
        assert s.total_frames == 0
        assert s.selected_count == 0
        assert s.total_exposure_s == 0.0

    def test_total_frames_count(self) -> None:
        proj = _project(FrameEntry(path="a"), FrameEntry(path="b"), FrameEntry(path="c"))
        s = compute_summary(proj)
        assert s.total_frames == 3

    def test_selected_count(self) -> None:
        proj = _project(
            FrameEntry(path="a", selected=True),
            FrameEntry(path="b", selected=False),
        )
        s = compute_summary(proj)
        assert s.selected_count == 1
        assert s.unselected_count == 1

    def test_total_exposure_sums_selected_only(self) -> None:
        proj = _project(
            FrameEntry(path="a", exposure=120.0, selected=True),
            FrameEntry(path="b", exposure=300.0, selected=False),
        )
        s = compute_summary(proj)
        assert s.total_exposure_s == pytest.approx(120.0)

    def test_exposure_groups_grouped_by_value(self) -> None:
        proj = _project(
            FrameEntry(path="a", exposure=120.0, selected=True),
            FrameEntry(path="b", exposure=120.0, selected=True),
            FrameEntry(path="c", exposure=300.0, selected=True),
        )
        s = compute_summary(proj)
        groups = {g.exposure_s: g.count for g in s.exposure_groups}
        assert groups[120.0] == 2
        assert groups[300.0] == 1

    def test_exposure_groups_sorted(self) -> None:
        proj = _project(
            FrameEntry(path="a", exposure=300.0, selected=True),
            FrameEntry(path="b", exposure=120.0, selected=True),
        )
        s = compute_summary(proj)
        exps = [g.exposure_s for g in s.exposure_groups]
        assert exps == sorted(exps)

    def test_quality_stats_computed(self) -> None:
        proj = _project(
            FrameEntry(path="a", quality_score=0.9, selected=True),
            FrameEntry(path="b", quality_score=0.6, selected=True),
            FrameEntry(path="c", quality_score=0.75, selected=True),
        )
        s = compute_summary(proj)
        assert s.scored_count == 3
        assert s.quality_min == pytest.approx(0.6)
        assert s.quality_max == pytest.approx(0.9)
        assert s.quality_mean == pytest.approx(0.75)

    def test_quality_none_frames_excluded(self) -> None:
        proj = _project(
            FrameEntry(path="a", quality_score=None, selected=True),
            FrameEntry(path="b", quality_score=0.8, selected=True),
        )
        s = compute_summary(proj)
        assert s.scored_count == 1

    def test_quality_all_none_gives_none_stats(self) -> None:
        proj = _project(FrameEntry(path="a", selected=True))
        s = compute_summary(proj)
        assert s.quality_mean is None
        assert s.quality_min is None
        assert s.quality_max is None

    def test_temperature_stats(self) -> None:
        proj = _project(
            FrameEntry(path="a", temperature=-15.0, selected=True),
            FrameEntry(path="b", temperature=-5.0, selected=True),
        )
        s = compute_summary(proj)
        assert s.temp_min == pytest.approx(-15.0)
        assert s.temp_max == pytest.approx(-5.0)

    def test_temperature_excluded_when_all_none(self) -> None:
        proj = _project(FrameEntry(path="a", selected=True))
        s = compute_summary(proj)
        assert s.temp_min is None
        assert s.temp_max is None

    def test_unselected_frames_excluded_from_stats(self) -> None:
        proj = _project(
            FrameEntry(path="a", quality_score=0.9, temperature=-10.0, selected=False),
        )
        s = compute_summary(proj)
        assert s.scored_count == 0
        assert s.temp_min is None

    def test_frames_without_exposure_skipped_in_groups(self) -> None:
        proj = _project(
            FrameEntry(path="a", exposure=None, selected=True),
            FrameEntry(path="b", exposure=120.0, selected=True),
        )
        s = compute_summary(proj)
        assert len(s.exposure_groups) == 1
        assert s.total_exposure_s == pytest.approx(120.0)
