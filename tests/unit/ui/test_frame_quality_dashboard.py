"""Unit tests for FrameQualityDashboard widget (VER-431)."""
from __future__ import annotations

import csv
import os
import tempfile

import pytest

from astroai.ui.widgets.frame_quality_dashboard import (
    FrameQualityDashboard,
    _STATUS_ACCEPTED,
    _STATUS_REJECTED,
    _COL_INDEX,
    _COL_SCORE,
    _COL_STATUS,
)


@pytest.fixture()
def dashboard(qtbot):
    w = FrameQualityDashboard()
    qtbot.addWidget(w)
    return w


@pytest.fixture()
def populated(dashboard):
    scores = [0.9, 0.3, 0.8, 0.1, 0.75]
    rejected = [1, 3]
    dashboard.set_scores(scores, rejected)
    return dashboard, scores, rejected


# --- Construction ---

class TestConstruction:
    def test_creates_without_error(self, dashboard):
        assert dashboard is not None

    def test_initial_row_count_is_zero(self, dashboard):
        assert dashboard._table.rowCount() == 0

    def test_initial_export_button_disabled(self, dashboard):
        assert not dashboard._export_btn.isEnabled()

    def test_table_has_three_columns(self, dashboard):
        assert dashboard._table.columnCount() == 3

    def test_table_headers_correct(self, dashboard):
        h = dashboard._table.horizontalHeader()
        assert dashboard._table.horizontalHeaderItem(0).text() == "Frame"
        assert dashboard._table.horizontalHeaderItem(1).text() == "Score"
        assert dashboard._table.horizontalHeaderItem(2).text() == "Status"

    def test_sorting_enabled_by_default(self, dashboard):
        assert dashboard._table.isSortingEnabled()


# --- set_scores ---

class TestSetScores:
    def test_row_count_matches_scores(self, populated):
        dashboard, scores, _ = populated
        assert dashboard._table.rowCount() == len(scores)

    def test_export_button_enabled_after_set_scores(self, populated):
        dashboard, _, _ = populated
        assert dashboard._export_btn.isEnabled()

    def test_accepted_frames_show_correct_status(self, populated):
        dashboard, scores, rejected = populated
        rejected_set = set(rejected)
        for row in range(dashboard._table.rowCount()):
            idx = int(dashboard._table.item(row, _COL_INDEX).text())
            if idx not in rejected_set:
                assert dashboard._table.item(row, _COL_STATUS).text() == _STATUS_ACCEPTED

    def test_rejected_frames_show_correct_status(self, populated):
        dashboard, scores, rejected = populated
        rejected_set = set(rejected)
        for row in range(dashboard._table.rowCount()):
            idx = int(dashboard._table.item(row, _COL_INDEX).text())
            if idx in rejected_set:
                assert dashboard._table.item(row, _COL_STATUS).text() == _STATUS_REJECTED

    def test_score_values_in_table(self, populated):
        dashboard, scores, _ = populated
        # Collect (frame_idx, score_text) from all rows (order may vary due to sorting)
        found: dict[int, str] = {}
        for row in range(dashboard._table.rowCount()):
            idx = int(dashboard._table.item(row, _COL_INDEX).text())
            score_text = dashboard._table.item(row, _COL_SCORE).text()
            found[idx] = score_text
        for i, score in enumerate(scores):
            assert found[i] == f"{score:.4f}"

    def test_frame_indices_in_table(self, populated):
        dashboard, scores, _ = populated
        # All frame indices 0..n-1 must appear exactly once
        indices = [
            int(dashboard._table.item(row, _COL_INDEX).text())
            for row in range(dashboard._table.rowCount())
        ]
        assert sorted(indices) == list(range(len(scores)))

    def test_no_rejected_indices_all_accepted(self, dashboard, qtbot):
        dashboard.set_scores([0.5, 0.8, 0.9])
        for i in range(3):
            assert dashboard._table.item(i, _COL_STATUS).text() == _STATUS_ACCEPTED

    def test_set_scores_replaces_previous_data(self, populated):
        dashboard, _, _ = populated
        dashboard.set_scores([0.1, 0.2])
        assert dashboard._table.rowCount() == 2

    def test_empty_scores_disables_export_button(self, populated):
        dashboard, _, _ = populated
        dashboard.set_scores([])
        assert not dashboard._export_btn.isEnabled()


# --- clear ---

class TestClear:
    def test_clear_removes_rows(self, populated):
        dashboard, _, _ = populated
        dashboard.clear()
        assert dashboard._table.rowCount() == 0

    def test_clear_disables_export_button(self, populated):
        dashboard, _, _ = populated
        dashboard.clear()
        assert not dashboard._export_btn.isEnabled()

    def test_clear_resets_internal_scores(self, populated):
        dashboard, _, _ = populated
        dashboard.clear()
        assert dashboard._scores == []

    def test_clear_resets_rejected_set(self, populated):
        dashboard, _, _ = populated
        dashboard.clear()
        assert dashboard._rejected == set()


# --- accepted/rejected/total counts ---

class TestCounts:
    def test_total_count(self, populated):
        dashboard, scores, _ = populated
        assert dashboard.total_count() == len(scores)

    def test_rejected_count(self, populated):
        dashboard, _, rejected = populated
        assert dashboard.rejected_count() == len(rejected)

    def test_accepted_count(self, populated):
        dashboard, scores, rejected = populated
        assert dashboard.accepted_count() == len(scores) - len(rejected)

    def test_counts_zero_after_clear(self, populated):
        dashboard, _, _ = populated
        dashboard.clear()
        assert dashboard.total_count() == 0
        assert dashboard.accepted_count() == 0
        assert dashboard.rejected_count() == 0


# --- row_data ---

class TestRowData:
    def test_row_data_length(self, populated):
        dashboard, scores, _ = populated
        assert len(dashboard.row_data()) == len(scores)

    def test_row_data_scores_correct(self, populated):
        dashboard, scores, _ = populated
        data = dashboard.row_data()
        for idx, score, _ in data:
            assert abs(score - scores[idx]) < 1e-9

    def test_row_data_status_values(self, populated):
        dashboard, scores, rejected = populated
        data = dashboard.row_data()
        for idx, _, status in data:
            expected = _STATUS_REJECTED if idx in rejected else _STATUS_ACCEPTED
            assert status == expected


# --- CSV export ---

class TestCSVExport:
    def test_export_csv_creates_file(self, populated):
        dashboard, _, _ = populated
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
            path = fh.name
        try:
            dashboard.export_csv(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_export_csv_row_count(self, populated):
        dashboard, scores, _ = populated
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
            path = fh.name
        try:
            n = dashboard.export_csv(path)
            assert n == len(scores)
        finally:
            os.unlink(path)

    def test_export_csv_header(self, populated):
        dashboard, _, _ = populated
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fh:
            path = fh.name
        try:
            dashboard.export_csv(path)
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader)
            assert header == ["Frame", "Score", "Status"]
        finally:
            os.unlink(path)

    def test_export_csv_status_values(self, populated):
        dashboard, scores, rejected = populated
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fh:
            path = fh.name
        try:
            dashboard.export_csv(path)
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader)  # skip header
                rows = list(reader)
            for row in rows:
                idx = int(row[0])
                expected_status = _STATUS_REJECTED if idx in rejected else _STATUS_ACCEPTED
                assert row[2] == expected_status
        finally:
            os.unlink(path)


# --- Summary label ---

class TestSummaryLabel:
    def test_summary_shows_frame_count(self, populated):
        dashboard, scores, _ = populated
        assert str(len(scores)) in dashboard._summary_label.text()

    def test_summary_shows_no_scores_when_empty(self, dashboard):
        assert "Keine" in dashboard._summary_label.text()

    def test_summary_after_clear(self, populated):
        dashboard, _, _ = populated
        dashboard.clear()
        assert "Keine" in dashboard._summary_label.text()
