"""Tests for frame statistics CSV export."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from astroai.core.io.frame_stats_export import export_frame_stats
from astroai.project.project_file import FrameEntry


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestExportFrameStats:
    def test_empty_frames_writes_header_only(self, tmp_path: Path) -> None:
        dest = tmp_path / "stats.csv"
        rows = export_frame_stats([], dest)
        assert rows == 0
        assert dest.exists()
        with dest.open(encoding="utf-8") as fh:
            lines = fh.readlines()
        assert len(lines) == 1  # header only

    def test_returns_row_count(self, tmp_path: Path) -> None:
        frames = [FrameEntry(path="a.fits"), FrameEntry(path="b.fits")]
        rows = export_frame_stats(frames, tmp_path / "s.csv")
        assert rows == 2

    def test_filename_column_is_basename(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        export_frame_stats([FrameEntry(path="/some/dir/light001.fits")], dest)
        rows = _read_csv(dest)
        assert rows[0]["filename"] == "light001.fits"

    def test_path_column_is_full_path(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        export_frame_stats([FrameEntry(path="light.fits")], dest)
        rows = _read_csv(dest)
        assert "light.fits" in rows[0]["path"]

    def test_exposure_formatted(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        export_frame_stats([FrameEntry(path="f.fits", exposure=120.0)], dest)
        rows = _read_csv(dest)
        assert rows[0]["exposure_s"] == "120.000"

    def test_gain_iso_integer(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        export_frame_stats([FrameEntry(path="f.fits", gain_iso=800)], dest)
        rows = _read_csv(dest)
        assert rows[0]["gain_iso"] == "800"

    def test_temperature_formatted(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        export_frame_stats([FrameEntry(path="f.fits", temperature=-15.5)], dest)
        rows = _read_csv(dest)
        assert rows[0]["temperature_c"] == "-15.5"

    def test_quality_score_formatted(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        export_frame_stats([FrameEntry(path="f.fits", quality_score=0.9123)], dest)
        rows = _read_csv(dest)
        assert rows[0]["quality_score"] == "0.9123"

    def test_none_fields_are_empty_strings(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        export_frame_stats([FrameEntry(path="f.fits")], dest)
        row = _read_csv(dest)[0]
        assert row["exposure_s"] == ""
        assert row["gain_iso"] == ""
        assert row["temperature_c"] == ""
        assert row["quality_score"] == ""

    def test_selected_flag(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        export_frame_stats(
            [FrameEntry(path="a.fits", selected=True), FrameEntry(path="b.fits", selected=False)],
            dest,
        )
        rows = _read_csv(dest)
        assert rows[0]["selected"] == "1"
        assert rows[1]["selected"] == "0"

    def test_non_frame_entries_skipped(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        rows = export_frame_stats([object(), "not_a_frame", None], dest)  # type: ignore[list-item]
        assert rows == 0

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        dest = tmp_path / "subdir" / "output.csv"
        export_frame_stats([FrameEntry(path="f.fits")], dest)
        assert dest.exists()

    def test_header_columns(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        export_frame_stats([], dest)
        with dest.open(encoding="utf-8") as fh:
            header = fh.readline().strip()
        assert header == "filename,path,exposure_s,gain_iso,temperature_c,quality_score,selected"

    def test_multiple_frames_order_preserved(self, tmp_path: Path) -> None:
        dest = tmp_path / "s.csv"
        frames = [FrameEntry(path=f"frame_{i:02d}.fits") for i in range(5)]
        export_frame_stats(frames, dest)
        rows = _read_csv(dest)
        assert [r["filename"] for r in rows] == [f"frame_{i:02d}.fits" for i in range(5)]
