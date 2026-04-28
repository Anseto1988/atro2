"""Unit tests for SmartCalibPanel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits as astrofits

from astroai.project.project_file import AstroProject, FrameEntry
from astroai.ui.widgets.smart_calib_panel import SmartCalibPanel


def _write_fits(path: Path, imagetyp: str = "Dark Frame", exposure: float = 120.0) -> None:
    hdu = astrofits.PrimaryHDU(data=np.zeros((32, 32), dtype=np.float32))
    if imagetyp:
        hdu.header["IMAGETYP"] = imagetyp
    hdu.header["EXPTIME"] = exposure
    hdu.writeto(str(path), overwrite=True)


@pytest.fixture()
def panel(qtbot) -> SmartCalibPanel:  # type: ignore[no-untyped-def]
    w = SmartCalibPanel()
    qtbot.addWidget(w)
    return w


@pytest.fixture()
def project() -> AstroProject:
    return AstroProject()


class TestInitialState:
    def test_scan_button_enabled(self, panel: SmartCalibPanel) -> None:
        assert panel._scan_btn.isEnabled()

    def test_match_button_disabled(self, panel: SmartCalibPanel) -> None:
        assert not panel._match_btn.isEnabled()

    def test_table_empty(self, panel: SmartCalibPanel) -> None:
        assert panel._result_table.rowCount() == 0

    def test_coverage_labels_show_dash(self, panel: SmartCalibPanel) -> None:
        assert panel._dark_coverage_label.text() == "—"
        assert panel._flat_coverage_label.text() == "—"

    def test_status_label_empty(self, panel: SmartCalibPanel) -> None:
        assert panel._status_label.text() == ""

    def test_recursive_unchecked_by_default(self, panel: SmartCalibPanel) -> None:
        assert not panel._recursive_cb.isChecked()


class TestBrowse:
    def test_browse_sets_dir_edit(self, panel: SmartCalibPanel, monkeypatch) -> None:
        monkeypatch.setattr(
            "astroai.ui.widgets.smart_calib_panel.QFileDialog.getExistingDirectory",
            lambda *a, **kw: "/chosen/dir",
        )
        panel._on_browse()
        assert panel._dir_edit.text() == "/chosen/dir"

    def test_browse_cancelled_leaves_dir_unchanged(self, panel: SmartCalibPanel, monkeypatch) -> None:
        panel._dir_edit.setText("/original/dir")
        monkeypatch.setattr(
            "astroai.ui.widgets.smart_calib_panel.QFileDialog.getExistingDirectory",
            lambda *a, **kw: "",
        )
        panel._on_browse()
        assert panel._dir_edit.text() == "/original/dir"


class TestScan:
    def test_invalid_directory_shows_error(self, panel: SmartCalibPanel) -> None:
        panel._dir_edit.setText("/nonexistent/path/xyz_doesnotexist")
        panel._on_scan()
        assert "ngültig" in panel._status_label.text()
        assert not panel._match_btn.isEnabled()

    def test_invalid_dir_clears_table(self, panel: SmartCalibPanel) -> None:
        panel._dir_edit.setText("/nonexistent/path/xyz_doesnotexist")
        panel._on_scan()
        assert panel._result_table.rowCount() == 0

    def test_empty_directory_shows_zero_count(self, panel: SmartCalibPanel, tmp_path: Path) -> None:
        panel._dir_edit.setText(str(tmp_path))
        panel._on_scan()
        assert "0 Frame" in panel._status_label.text()
        assert panel._result_table.rowCount() == 0

    def test_dark_frames_in_table(self, panel: SmartCalibPanel, tmp_path: Path) -> None:
        _write_fits(tmp_path / "dark.fits", imagetyp="Dark Frame")
        panel._dir_edit.setText(str(tmp_path))
        panel._on_scan()
        assert panel._result_table.rowCount() == 1
        assert panel._result_table.item(0, 0).text() == "Dark"
        assert panel._result_table.item(0, 1).text() == "1"

    def test_multiple_types_all_shown(self, panel: SmartCalibPanel, tmp_path: Path) -> None:
        _write_fits(tmp_path / "dark.fits", imagetyp="Dark Frame")
        _write_fits(tmp_path / "flat.fits", imagetyp="Flat Field")
        _write_fits(tmp_path / "bias.fits", imagetyp="Bias")
        panel._dir_edit.setText(str(tmp_path))
        panel._on_scan()
        types = {panel._result_table.item(r, 0).text() for r in range(panel._result_table.rowCount())}
        assert {"Dark", "Flat", "Bias"} <= types

    def test_calib_frames_enable_match_button(self, panel: SmartCalibPanel, tmp_path: Path) -> None:
        _write_fits(tmp_path / "dark.fits", imagetyp="Dark Frame")
        panel._dir_edit.setText(str(tmp_path))
        panel._on_scan()
        assert panel._match_btn.isEnabled()

    def test_only_lights_match_button_stays_disabled(self, panel: SmartCalibPanel, tmp_path: Path) -> None:
        _write_fits(tmp_path / "light.fits", imagetyp="Light Frame")
        panel._dir_edit.setText(str(tmp_path))
        panel._on_scan()
        assert not panel._match_btn.isEnabled()

    def test_rescan_clears_previous_results(self, panel: SmartCalibPanel, tmp_path: Path) -> None:
        _write_fits(tmp_path / "dark.fits", imagetyp="Dark Frame")
        panel._dir_edit.setText(str(tmp_path))
        panel._on_scan()
        assert panel._result_table.rowCount() == 1
        (tmp_path / "dark.fits").unlink()
        panel._on_scan()
        assert panel._result_table.rowCount() == 0

    def test_recursive_finds_nested_frames(self, panel: SmartCalibPanel, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_fits(sub / "dark.fits", imagetyp="Dark Frame")
        panel._dir_edit.setText(str(tmp_path))
        panel._recursive_cb.setChecked(True)
        panel._on_scan()
        assert panel._result_table.rowCount() == 1

    def test_non_recursive_skips_subdirs(self, panel: SmartCalibPanel, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_fits(sub / "dark.fits", imagetyp="Dark Frame")
        panel._dir_edit.setText(str(tmp_path))
        panel._recursive_cb.setChecked(False)
        panel._on_scan()
        assert panel._result_table.rowCount() == 0

    def test_scan_resets_coverage_labels(self, panel: SmartCalibPanel, tmp_path: Path) -> None:
        panel._dark_coverage_label.setText("50%")
        panel._flat_coverage_label.setText("75%")
        panel._dir_edit.setText(str(tmp_path))
        panel._on_scan()
        assert panel._dark_coverage_label.text() == "—"
        assert panel._flat_coverage_label.text() == "—"


class TestApplyMatch:
    def test_no_getter_shows_error(self, panel: SmartCalibPanel) -> None:
        panel._on_apply_match()
        assert "Kein Projekt" in panel._status_label.text()

    def test_getter_returns_none_shows_error(self, panel: SmartCalibPanel) -> None:
        panel.set_project_getter(lambda: None)
        panel._on_apply_match()
        assert "Kein Projekt" in panel._status_label.text()

    def test_empty_project_no_lights_shows_error(
        self, panel: SmartCalibPanel, tmp_path: Path, project: AstroProject
    ) -> None:
        panel.set_project_getter(lambda: project)
        _write_fits(tmp_path / "dark.fits", imagetyp="Dark Frame")
        panel._dir_edit.setText(str(tmp_path))
        panel._on_scan()
        panel._on_apply_match()
        assert "Light-Frame" in panel._status_label.text()

    def test_match_writes_dark_frames_to_project(
        self, panel: SmartCalibPanel, tmp_path: Path, project: AstroProject
    ) -> None:
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        _write_fits(calib_dir / "dark.fits", imagetyp="Dark Frame", exposure=120.0)

        project.input_frames.append(FrameEntry(path=str(tmp_path / "light.fits"), exposure=120.0))
        panel.set_project_getter(lambda: project)
        panel._dir_edit.setText(str(calib_dir))
        panel._on_scan()
        panel._on_apply_match()

        assert len(project.calibration.dark_frames) >= 1

    def test_match_updates_coverage_labels(
        self, panel: SmartCalibPanel, tmp_path: Path, project: AstroProject
    ) -> None:
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        _write_fits(calib_dir / "dark.fits", imagetyp="Dark Frame", exposure=120.0)

        project.input_frames.append(FrameEntry(path=str(tmp_path / "light.fits"), exposure=120.0))
        panel.set_project_getter(lambda: project)
        panel._dir_edit.setText(str(calib_dir))
        panel._on_scan()
        panel._on_apply_match()

        assert panel._dark_coverage_label.text() != "—"
        assert panel._flat_coverage_label.text() != "—"

    def test_match_shows_applied_status(
        self, panel: SmartCalibPanel, tmp_path: Path, project: AstroProject
    ) -> None:
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        _write_fits(calib_dir / "dark.fits", imagetyp="Dark Frame", exposure=120.0)

        project.input_frames.append(FrameEntry(path=str(tmp_path / "light.fits"), exposure=120.0))
        panel.set_project_getter(lambda: project)
        panel._dir_edit.setText(str(calib_dir))
        panel._on_scan()
        panel._on_apply_match()

        assert "angewendet" in panel._status_label.text()

    def test_match_marks_project_dirty(
        self, panel: SmartCalibPanel, tmp_path: Path, project: AstroProject
    ) -> None:
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        _write_fits(calib_dir / "dark.fits", imagetyp="Dark Frame", exposure=120.0)

        project.input_frames.append(FrameEntry(path=str(tmp_path / "light.fits"), exposure=120.0))
        panel.set_project_getter(lambda: project)
        panel._dir_edit.setText(str(calib_dir))
        panel._on_scan()
        project.mark_clean()
        panel._on_apply_match()

        assert project.is_dirty

    def test_coverage_zero_when_no_match(
        self, panel: SmartCalibPanel, tmp_path: Path, project: AstroProject
    ) -> None:
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        # Use very different exposure so no match
        _write_fits(calib_dir / "dark.fits", imagetyp="Dark Frame", exposure=9999.0)

        project.input_frames.append(FrameEntry(path=str(tmp_path / "light.fits"), exposure=120.0))
        panel.set_project_getter(lambda: project)
        panel._dir_edit.setText(str(calib_dir))
        panel._on_scan()
        panel._on_apply_match()

        assert panel._dark_coverage_label.text() == "0%"

    def test_set_project_getter_replaces_getter(
        self, panel: SmartCalibPanel, project: AstroProject
    ) -> None:
        other = AstroProject()
        panel.set_project_getter(lambda: other)
        panel._on_apply_match()
        assert "Light-Frame" in panel._status_label.text()
