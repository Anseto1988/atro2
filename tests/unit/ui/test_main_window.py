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
        assert len(win._pipeline.steps) == 12

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
