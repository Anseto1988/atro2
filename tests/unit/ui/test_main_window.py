"""Tests for MainWindow (app.py) — project actions, signals, menu setup."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QMenu

from astroai.licensing.models import LicenseStatus, LicenseTier
from astroai.ui.license_adapter import QLicenseAdapter
from astroai.ui.main.app import MainWindow


@pytest.fixture()
def adapter(qtbot) -> MagicMock:
    mock = MagicMock(spec=QLicenseAdapter)
    mock.tier = LicenseTier.FREE
    mock.status_changed = MagicMock()
    mock.status_changed.connect = MagicMock()
    mock.verify = MagicMock(return_value=LicenseStatus())
    return mock


@pytest.fixture()
def win(qtbot, adapter) -> MainWindow:
    w = MainWindow(license_adapter=adapter)
    qtbot.addWidget(w)
    return w


class TestMainWindowInit:
    def test_window_title(self, win: MainWindow) -> None:
        assert "AstroAI" in win.windowTitle()

    def test_minimum_size(self, win: MainWindow) -> None:
        assert win.minimumWidth() >= 960
        assert win.minimumHeight() >= 640

    def test_central_widget_exists(self, win: MainWindow) -> None:
        assert win.centralWidget() is not None

    def test_viewer_exists(self, win: MainWindow) -> None:
        assert win._viewer is not None

    def test_histogram_dock(self, win: MainWindow) -> None:
        assert win._histogram is not None

    def test_progress_dock(self, win: MainWindow) -> None:
        assert win._progress is not None

    def test_workflow_dock(self, win: MainWindow) -> None:
        assert win._workflow is not None

    def test_log_dock(self, win: MainWindow) -> None:
        assert win._log_widget is not None

    def test_pipeline_model(self, win: MainWindow) -> None:
        assert win._pipeline is not None
        assert len(win._pipeline.steps) == 13

    def test_statusbar_message(self, win: MainWindow) -> None:
        assert win.statusBar().currentMessage() == "Bereit"

    def test_license_badge_in_statusbar(self, win: MainWindow) -> None:
        assert win._license_badge is not None

    def test_offline_banner_exists(self, win: MainWindow) -> None:
        assert win._offline_banner is not None

    def test_menu_bar_has_datei(self, win: MainWindow) -> None:
        menus = win.menuBar().findChildren(QMenu)
        titles = [m.title() for m in menus]
        assert any("Datei" in t for t in titles)


class TestMainWindowSignalHandlers:
    def test_on_image_loaded(self, win: MainWindow) -> None:
        data = np.random.rand(50, 80).astype(np.float32)
        win._on_image_loaded(data, "test.fits")
        assert win._viewer._raw_data is not None
        assert win._viewer._width == 80
        assert "test.fits" in win.statusBar().currentMessage()

    def test_on_load_error(self, win: MainWindow) -> None:
        win._on_load_error("Datei nicht gefunden")
        assert "Fehler" in win.statusBar().currentMessage()

    def test_on_load_status(self, win: MainWindow) -> None:
        win._on_load_status("Lade test.png...")
        assert win._progress._label.text() == "Lade test.png..."

    def test_on_zoom_changed(self, win: MainWindow) -> None:
        win._on_zoom_changed(2.0)
        msg = win.statusBar().currentMessage()
        assert "Zoom" in msg
        assert "200" in msg

    def test_on_pixel_hovered(self, win: MainWindow) -> None:
        win._on_pixel_hovered(42, 17, 0.75)
        msg = win.statusBar().currentMessage()
        assert "(42, 17)" in msg
        assert "0.75" in msg


class TestMainWindowProjectActions:
    def test_on_new_project(self, win: MainWindow) -> None:
        win._on_new_project()
        assert win._project_path is None
        assert "Neues Projekt" in win.statusBar().currentMessage()
        assert "AstroAI" in win.windowTitle()

    def test_save_project(self, win: MainWindow, tmp_path: Path) -> None:
        save_path = tmp_path / "test_proj.astroai"
        with patch("astroai.ui.main.app.ProjectSerializer.save") as mock_save:
            win._save_project(save_path)
            mock_save.assert_called_once()
        assert win._project_path == save_path
        assert "gespeichert" in win.statusBar().currentMessage()

    def test_save_project_error(self, win: MainWindow, tmp_path: Path) -> None:
        from astroai.project.serializer import ProjectSerializerError

        save_path = tmp_path / "bad.astroai"
        with patch(
            "astroai.ui.main.app.ProjectSerializer.save",
            side_effect=ProjectSerializerError("disk full"),
        ):
            with patch("astroai.ui.main.app.QMessageBox.critical") as mock_crit:
                win._save_project(save_path)
                mock_crit.assert_called_once()

    def test_load_project(self, win: MainWindow, tmp_path: Path) -> None:
        from astroai.project import AstroProject

        proj = AstroProject()
        proj.metadata.name = "TestProjekt"
        load_path = tmp_path / "loaded.astroai"
        with patch("astroai.ui.main.app.ProjectSerializer.load", return_value=proj):
            win._load_project(load_path)
        assert win._project_path == load_path
        assert "geladen" in win.statusBar().currentMessage()
        assert "loaded" in win.windowTitle() or "TestProjekt" in win.windowTitle()

    def test_load_project_error(self, win: MainWindow, tmp_path: Path) -> None:
        from astroai.project.serializer import ProjectSerializerError

        with patch(
            "astroai.ui.main.app.ProjectSerializer.load",
            side_effect=ProjectSerializerError("corrupt"),
        ):
            with patch("astroai.ui.main.app.QMessageBox.warning") as mock_warn:
                win._load_project(tmp_path / "bad.astroai")
                mock_warn.assert_called_once()

    def test_update_title_with_path(self, win: MainWindow, tmp_path: Path) -> None:
        win._project_path = tmp_path / "my_project.astroai"
        win._update_title()
        assert "my_project" in win.windowTitle()

    def test_update_title_without_path(self, win: MainWindow) -> None:
        win._project_path = None
        win._update_title()
        assert "AstroAI" in win.windowTitle()

    def test_rebuild_recent_menu_empty(self, win: MainWindow) -> None:
        win._recent._entries = []
        win._rebuild_recent_menu()
        actions = win._recent_menu.actions()
        assert len(actions) == 1
        assert not actions[0].isEnabled()

    def test_rebuild_recent_menu_with_entries(self, win: MainWindow) -> None:
        win._recent._entries = ["/path/a.astroai", "/path/b.astroai"]
        win._rebuild_recent_menu()
        actions = win._recent_menu.actions()
        assert len(actions) == 2

    @patch("astroai.ui.main.app.QFileDialog.getOpenFileName", return_value=("", ""))
    def test_on_open_project_cancelled(self, mock_dlg, win: MainWindow) -> None:
        win._on_open_project()
        assert win._project_path is None

    @patch("astroai.ui.main.app.QFileDialog.getSaveFileName", return_value=("", ""))
    def test_on_save_project_as_cancelled(self, mock_dlg, win: MainWindow) -> None:
        win._on_save_project_as()
        assert win._project_path is None

    @patch("astroai.ui.main.app.QFileDialog.getSaveFileName")
    def test_on_save_project_as_appends_extension(self, mock_dlg, win: MainWindow, tmp_path: Path) -> None:
        target = tmp_path / "noext"
        mock_dlg.return_value = (str(target), "")
        with patch("astroai.ui.main.app.ProjectSerializer.save"):
            win._on_save_project_as()
        assert win._project_path is not None
        assert win._project_path.suffix == ".astroai"

    def test_on_save_project_delegates_to_save_as_when_no_path(self, win: MainWindow) -> None:
        win._project_path = None
        with patch.object(win, "_on_save_project_as") as mock_as:
            win._on_save_project()
            mock_as.assert_called_once()

    def test_on_save_project_uses_existing_path(self, win: MainWindow, tmp_path: Path) -> None:
        existing = tmp_path / "existing.astroai"
        win._project_path = existing
        with patch.object(win, "_save_project") as mock_save:
            win._on_save_project()
            mock_save.assert_called_once_with(existing)


class TestMainWindowOpenImage:
    @patch("astroai.ui.main.app.QFileDialog.getOpenFileName", return_value=("", ""))
    def test_on_open_image_cancelled(self, mock_dlg, win: MainWindow) -> None:
        win._on_open_image()
        assert not win._file_loader.is_loading

    def test_on_open_image_skips_when_loading(self, win: MainWindow) -> None:
        win._file_loader._thread = MagicMock()
        win._file_loader._thread.isRunning.return_value = True
        with patch("astroai.ui.main.app.QFileDialog.getOpenFileName") as mock_dlg:
            win._on_open_image()
            mock_dlg.assert_not_called()


class TestMainWindowImportFrames:
    @patch("astroai.ui.main.app.QFileDialog.getOpenFileNames", return_value=([], ""))
    def test_on_import_lights_cancelled(self, mock_dlg, win: MainWindow) -> None:
        win._on_import_lights()
        assert len(win._project.input_frames) == 0

    @patch("astroai.ui.main.app.QFileDialog.getOpenFileNames", return_value=(["/a.fits", "/b.fits"], ""))
    def test_on_import_lights_adds_frames(self, mock_dlg, win: MainWindow) -> None:
        win._on_import_lights()
        assert len(win._project.input_frames) == 2
        assert "2 Light" in win.statusBar().currentMessage()

    @patch("astroai.ui.main.app.QFileDialog.getOpenFileNames", return_value=(["/a.fits"], ""))
    def test_on_import_lights_no_duplicate(self, mock_dlg, win: MainWindow) -> None:
        win._on_import_lights()
        win._on_import_lights()
        assert len(win._project.input_frames) == 1

    @patch("astroai.ui.main.app.QFileDialog.getOpenFileNames", return_value=([], ""))
    def test_on_import_darks_cancelled(self, mock_dlg, win: MainWindow) -> None:
        win._on_import_darks()

    @patch("astroai.ui.main.app.QFileDialog.getOpenFileNames", return_value=(["/d1.fits", "/d2.fits"], ""))
    def test_on_import_darks_adds(self, mock_dlg, win: MainWindow) -> None:
        win._on_import_darks()
        assert len(win._project.calibration.dark_frames) == 2
        assert "Dark" in win.statusBar().currentMessage()

    @patch("astroai.ui.main.app.QFileDialog.getOpenFileNames", return_value=([], ""))
    def test_on_import_flats_cancelled(self, mock_dlg, win: MainWindow) -> None:
        win._on_import_flats()

    @patch("astroai.ui.main.app.QFileDialog.getOpenFileNames", return_value=(["/f1.fits"], ""))
    def test_on_import_flats_adds(self, mock_dlg, win: MainWindow) -> None:
        win._on_import_flats()
        assert "/f1.fits" in win._project.calibration.flat_frames
        assert "Flat" in win.statusBar().currentMessage()

    @patch("astroai.ui.main.app.QFileDialog.getOpenFileNames", return_value=([], ""))
    def test_on_import_bias_cancelled(self, mock_dlg, win: MainWindow) -> None:
        win._on_import_bias()

    @patch("astroai.ui.main.app.QFileDialog.getOpenFileNames", return_value=(["/b1.fits"], ""))
    def test_on_import_bias_adds(self, mock_dlg, win: MainWindow) -> None:
        win._on_import_bias()
        assert "/b1.fits" in win._project.calibration.bias_frames
        assert "Bias" in win.statusBar().currentMessage()


class TestMainWindowOpenImagePath:
    @patch("astroai.ui.main.app.QFileDialog.getOpenFileName")
    def test_on_open_image_starts_load(self, mock_dlg, win: MainWindow, tmp_path: Path) -> None:
        img_path = tmp_path / "test.fits"
        img_path.write_bytes(b"")
        mock_dlg.return_value = (str(img_path), "")
        with patch.object(win._file_loader, "load") as mock_load:
            win._on_open_image()
            mock_load.assert_called_once_with(img_path)
        assert win._progress._bar.maximum() == 0  # indeterminate


class TestMainWindowOpenProjectPath:
    @patch("astroai.ui.main.app.QFileDialog.getOpenFileName")
    def test_on_open_project_with_path(self, mock_dlg, win: MainWindow, tmp_path: Path) -> None:
        from astroai.project import AstroProject

        proj_path = tmp_path / "p.astroai"
        mock_dlg.return_value = (str(proj_path), "")
        with patch("astroai.ui.main.app.ProjectSerializer.load", return_value=AstroProject()):
            win._on_open_project()
        assert win._project_path == proj_path


class TestMainWindowMiscSlots:
    def test_on_calibration_finished(self, win: MainWindow) -> None:
        win._benchmark.start()
        win._on_calibration_finished(None)

    def test_on_calibration_error(self, win: MainWindow) -> None:
        win._on_calibration_error("GPU exploded")
        # no exception — benchmark reset, log written

    def test_on_comet_preview_changed_with_image(self, win: MainWindow) -> None:
        data = np.ones((32, 32), dtype=np.float32)
        win._pipeline._comet_star_stack = data
        win._pipeline._comet_nucleus_stack = data
        win._pipeline._comet_tracking_mode = "blend"
        win._on_comet_preview_changed()
        assert win._viewer._raw_data is not None
        assert "Komet" in win.statusBar().currentMessage()

    def test_on_comet_preview_changed_no_image(self, win: MainWindow) -> None:
        win._pipeline._comet_preview_image = None
        win._on_comet_preview_changed()  # no-op, no exception

    def test_on_manage_license(self, win: MainWindow) -> None:
        with patch("astroai.ui.main.app.ActivationDialog") as MockDlg:
            instance = MockDlg.return_value
            instance.exec = MagicMock()
            win._on_manage_license()
            instance.exec.assert_called_once()


class TestLoadStylesheet:
    def test_returns_empty_when_no_qss(self, tmp_path: Path) -> None:
        import astroai.ui.main.app as app_mod
        original = app_mod._RESOURCES
        app_mod._RESOURCES = tmp_path
        try:
            result = app_mod._load_stylesheet()
        finally:
            app_mod._RESOURCES = original
        assert result == ""

    def test_returns_content_when_qss_exists(self, tmp_path: Path) -> None:
        import astroai.ui.main.app as app_mod
        (tmp_path / "dark_theme.qss").write_text("QWidget { color: white; }", encoding="utf-8")
        original = app_mod._RESOURCES
        app_mod._RESOURCES = tmp_path
        try:
            result = app_mod._load_stylesheet()
        finally:
            app_mod._RESOURCES = original
        assert "QWidget" in result


class TestSetWcsSolution:
    def test_set_wcs_none_clears(self, win: MainWindow) -> None:
        win.set_wcs_solution(None)
        assert win._annotation_overlay._wcs is None

    def test_set_wcs_with_wcs_transform_protocol(self, win: MainWindow) -> None:
        from astroai.ui.overlay.sky_objects import WcsTransform

        mock_wcs = MagicMock(spec=WcsTransform)
        win.set_wcs_solution(mock_wcs)
        assert win._annotation_overlay._wcs is mock_wcs

    def test_set_wcs_unknown_type_no_image_shape(self, win: MainWindow) -> None:
        win.set_wcs_solution(object())
        assert win._annotation_overlay._wcs is None

    def test_set_wcs_engine_overlay(self, win: MainWindow) -> None:
        from astroai.engine.platesolving.annotation import AnnotationOverlay as EngineOverlay
        from astroai.ui.overlay.wcs_adapter import WcsAdapter

        mock_overlay = MagicMock(spec=EngineOverlay)
        fake_adapter = MagicMock(spec=WcsAdapter)
        with patch("astroai.ui.overlay.wcs_adapter.WcsAdapter.from_engine_overlay", return_value=fake_adapter):
            win.set_wcs_solution(mock_overlay)
        assert win._annotation_overlay._wcs is fake_adapter

    def test_set_wcs_solve_result(self, win: MainWindow) -> None:
        from astroai.engine.platesolving.solver import SolveResult
        from astroai.ui.overlay.wcs_adapter import WcsAdapter

        mock_result = MagicMock(spec=SolveResult)
        fake_adapter = MagicMock(spec=WcsAdapter)
        with patch("astroai.ui.overlay.wcs_adapter.WcsAdapter.from_solve_result", return_value=fake_adapter):
            win.set_wcs_solution(mock_result, image_shape=(100, 200))
        assert win._annotation_overlay._wcs is fake_adapter

    def test_set_wcs_wcs_solution_sets_sky_overlay(self, win: MainWindow) -> None:
        from astroai.astrometry.catalog import WcsSolution
        from astroai.ui.overlay.sky_objects import WcsTransform

        mock_adapter = MagicMock(spec=WcsTransform)
        # Pass via WcsTransform path to get adapter set, then confirm wcs_solution=None
        win.set_wcs_solution(mock_adapter)  # adapter set via WcsTransform isinstance
        # wcs_solution branch: adapter is not None, but wcs is not a WcsSolution → None
        assert win._sky_overlay._solution is None


class TestMainWindowLicense:
    def test_require_tier_allowed(self, win: MainWindow, adapter) -> None:
        adapter.tier = LicenseTier.PRO_ANNUAL
        assert win.require_tier("Denoise", LicenseTier.PRO_ANNUAL) is True

    def test_require_tier_blocked(self, win: MainWindow, adapter) -> None:
        adapter.tier = LicenseTier.FREE
        with patch("astroai.ui.main.app.UpgradeDialog") as MockDlg:
            instance = MockDlg.return_value
            instance.activate_requested = MagicMock()
            instance.activate_requested.connect = MagicMock()
            instance.exec = MagicMock()
            result = win.require_tier("Denoise", LicenseTier.PRO_ANNUAL)
        assert result is False
