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
        assert len(win._pipeline.steps) == len(win._pipeline.DEFAULT_STEPS)

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
        assert "Zoom" in win._zoom_label.text()
        assert "200" in win._zoom_label.text()

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

    def test_on_import_from_folder_cancelled(self, win: MainWindow, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **kw: "")
        win._on_import_from_folder()
        assert len(win._project.input_frames) == 0

    def test_on_import_from_folder_finds_fits(
        self, win: MainWindow, tmp_path: Path, monkeypatch
    ) -> None:
        from PySide6.QtWidgets import QFileDialog
        (tmp_path / "frame1.fits").write_bytes(b"")
        (tmp_path / "frame2.fit").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory", lambda *a, **kw: str(tmp_path)
        )
        win._on_import_from_folder()
        assert len(win._project.input_frames) == 2

    def test_on_import_from_folder_recursive(
        self, win: MainWindow, tmp_path: Path, monkeypatch
    ) -> None:
        from PySide6.QtWidgets import QFileDialog
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "frame.fits").write_bytes(b"")
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory", lambda *a, **kw: str(tmp_path)
        )
        win._on_import_from_folder()
        assert len(win._project.input_frames) == 1

    def test_on_import_from_folder_empty_shows_message(
        self, win: MainWindow, tmp_path: Path, monkeypatch
    ) -> None:
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory", lambda *a, **kw: str(tmp_path)
        )
        win._on_import_from_folder()
        assert "gefunden" in win.statusBar().currentMessage().lower()


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

    def test_returns_empty_when_qfile_open_fails(self, tmp_path: Path) -> None:
        import astroai.ui.main.app as app_mod
        (tmp_path / "dark_theme.qss").write_text("body{}", encoding="utf-8")
        original = app_mod._RESOURCES
        app_mod._RESOURCES = tmp_path
        try:
            with patch("astroai.ui.main.app.QFile") as mock_qfile_cls:
                mock_f = MagicMock()
                mock_f.open.return_value = False
                mock_qfile_cls.return_value = mock_f
                result = app_mod._load_stylesheet()
        finally:
            app_mod._RESOURCES = original
        assert result == ""


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

        # Create an object that satisfies both WcsTransform (Protocol) and WcsSolution (class)
        class DualWcs(WcsSolution):
            def world_to_pixel(self, ra_deg: float, dec_deg: float) -> tuple[float, float] | None:
                return (0.0, 0.0)
            def pixel_to_world(self, x: float, y: float) -> tuple[float, float] | None:
                return (0.0, 0.0)
            def image_size(self) -> tuple[int, int]:
                return (100, 100)

        dual = object.__new__(DualWcs)
        win.set_wcs_solution(dual)
        # adapter set via WcsTransform, wcs_solution set via WcsSolution isinstance
        assert win._sky_overlay._solution is dual

    def test_set_wcs_with_image_shape_astropy_path(self, win: MainWindow) -> None:
        # Pass an unknown type with image_shape to exercise the astropy WCS block
        win.set_wcs_solution(object(), image_shape=(100, 200))
        assert win._annotation_overlay._wcs is None

    def test_set_wcs_astropy_wcs_instance(self, win: MainWindow) -> None:
        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        win.set_wcs_solution(wcs, image_shape=(100, 200))
        assert win._annotation_overlay._wcs is not None

    def test_set_wcs_engine_overlay_import_error(self, win: MainWindow) -> None:
        import sys
        # Make engine overlay import fail → exercises except ImportError (lines 554-555)
        with patch.dict(sys.modules, {
            "astroai.engine.platesolving.annotation": None,  # type: ignore[dict-item]
        }):
            win.set_wcs_solution(object(), image_shape=(100, 200))
        assert win._annotation_overlay._wcs is None

    def test_set_wcs_solve_result_import_error(self, win: MainWindow) -> None:
        import sys
        # Make SolveResult import fail → exercises except ImportError (lines 563-564)
        with patch.dict(sys.modules, {
            "astroai.engine.platesolving.annotation": None,  # type: ignore[dict-item]
            "astroai.engine.platesolving.solver": None,  # type: ignore[dict-item]
        }):
            win.set_wcs_solution(object(), image_shape=(100, 200))
        assert win._annotation_overlay._wcs is None

    def test_set_wcs_solution_catalog_import_error(self, win: MainWindow) -> None:
        import sys
        from astroai.ui.overlay.sky_objects import WcsTransform
        mock_adapter = MagicMock(spec=WcsTransform)
        # Make WcsSolution import fail → exercises except ImportError (lines 586-587)
        with patch.dict(sys.modules, {
            "astroai.astrometry.catalog": None,  # type: ignore[dict-item]
        }):
            win.set_wcs_solution(mock_adapter)
        # no exception raised

    def test_set_wcs_astropy_import_error(self, win: MainWindow) -> None:
        import sys
        # Make astropy import fail → exercises except ImportError (lines 573-574)
        with patch.dict(sys.modules, {
            "astroai.engine.platesolving.annotation": None,  # type: ignore[dict-item]
            "astroai.engine.platesolving.solver": None,  # type: ignore[dict-item]
            "astropy": None,  # type: ignore[dict-item]
            "astropy.wcs": None,  # type: ignore[dict-item]
        }):
            win.set_wcs_solution(object(), image_shape=(100, 200))
        assert win._annotation_overlay._wcs is None


class TestMainFunction:
    def test_main_creates_window_and_exits(self, qtbot) -> None:
        import sys
        from astroai.ui.main.app import main

        with patch("astroai.ui.main.app.QApplication.instance") as mock_inst, \
             patch("astroai.ui.main.app.MainWindow") as MockWin, \
             patch("astroai.ui.main.app.sys.exit") as mock_exit:
            mock_app = MagicMock()
            mock_app.exec.return_value = 0
            mock_inst.return_value = mock_app
            MockWin.return_value = MagicMock()
            main()
            mock_exit.assert_called_once_with(0)


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


class TestProjectSync:
    """Tests for _sync_model_to_project and _sync_project_to_model."""

    def test_sync_model_to_project_synthetic_flat(self, win: MainWindow) -> None:
        win._pipeline.synthetic_flat_enabled = True
        win._pipeline.synthetic_flat_tile_size = 128
        win._pipeline.synthetic_flat_smoothing_sigma = 12.0
        win._sync_model_to_project()
        assert win._project.synthetic_flat.enabled is True
        assert win._project.synthetic_flat.tile_size == 128
        assert win._project.synthetic_flat.smoothing_sigma == 12.0

    def test_sync_model_to_project_comet_stack(self, win: MainWindow) -> None:
        win._pipeline.comet_stack_enabled = True
        win._pipeline.comet_tracking_mode = "comet"
        win._pipeline.comet_blend_factor = 0.3
        win._sync_model_to_project()
        assert win._project.comet_stack.enabled is True
        assert win._project.comet_stack.tracking_mode == "comet"
        assert win._project.comet_stack.blend_factor == pytest.approx(0.3)

    def test_sync_project_to_model_synthetic_flat(self, win: MainWindow) -> None:
        win._project.synthetic_flat.enabled = True
        win._project.synthetic_flat.tile_size = 32
        win._project.synthetic_flat.smoothing_sigma = 3.5
        win._sync_project_to_model()
        assert win._pipeline.synthetic_flat_enabled is True
        assert win._pipeline.synthetic_flat_tile_size == 32
        assert win._pipeline.synthetic_flat_smoothing_sigma == pytest.approx(3.5)

    def test_sync_project_to_model_drizzle(self, win: MainWindow) -> None:
        win._project.drizzle.enabled = True
        win._project.drizzle.drop_size = 0.5
        win._sync_project_to_model()
        assert win._pipeline.drizzle_enabled is True
        assert win._pipeline.drizzle_drop_size == pytest.approx(0.5)

    def test_sync_roundtrip_preserves_starless(self, win: MainWindow) -> None:
        win._pipeline.starless_enabled = True
        win._pipeline.starless_strength = 0.7
        win._pipeline.starless_format = "fits"
        win._sync_model_to_project()
        win._pipeline.starless_enabled = False
        win._sync_project_to_model()
        assert win._pipeline.starless_enabled is True
        assert win._pipeline.starless_strength == pytest.approx(0.7)
        assert win._pipeline.starless_format == "fits"

    def test_wcs_adapter_initially_none(self, win: MainWindow) -> None:
        assert win._wcs_adapter is None

    def test_pixel_hovered_no_wcs_shows_pixel_coords(self, win: MainWindow) -> None:
        win._on_image_loaded(
            __import__("numpy").ones((16, 16), dtype=__import__("numpy").float32),
            "test.fits"
        )
        win._on_pixel_hovered(10, 20, 0.5)
        msg = win._status_bar.currentMessage()
        assert "(10, 20)" in msg
        assert "0.5" in msg

    def test_pixel_hovered_with_wcs_shows_radec(self, win: MainWindow) -> None:
        class _FakeWcs:
            def world_to_pixel(self, ra: float, dec: float) -> tuple[float, float] | None:
                return (0.0, 0.0)
            def pixel_to_world(self, x: float, y: float) -> tuple[float, float] | None:
                return (83.822, -5.391)
            def image_size(self) -> tuple[int, int]:
                return (100, 100)

        win._wcs_adapter = _FakeWcs()
        win._on_pixel_hovered(5, 5, 0.3)
        msg = win._status_bar.currentMessage()
        assert "83.82200" in msg or "83.822" in msg
        assert "RA" in msg

    def test_pixel_hovered_wcs_none_result_no_crash(self, win: MainWindow) -> None:
        class _NoneWcs:
            def world_to_pixel(self, ra: float, dec: float) -> tuple[float, float] | None:
                return None
            def pixel_to_world(self, x: float, y: float) -> tuple[float, float] | None:
                return None
            def image_size(self) -> tuple[int, int]:
                return (100, 100)

        win._wcs_adapter = _NoneWcs()
        win._on_pixel_hovered(5, 5, 0.3)
        msg = win._status_bar.currentMessage()
        assert "(5, 5)" in msg

    def test_set_wcs_solution_none_clears_adapter(self, win: MainWindow) -> None:
        win._wcs_adapter = object()
        win.set_wcs_solution(None)
        assert win._wcs_adapter is None

    def test_sync_curves_roundtrip(self, win: MainWindow) -> None:
        win._pipeline.curves_enabled = True
        win._pipeline.curves_rgb_points = [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]
        win._sync_model_to_project()
        win._pipeline.curves_enabled = False
        win._pipeline.curves_rgb_points = [(0.0, 0.0), (1.0, 1.0)]
        win._sync_project_to_model()
        assert win._pipeline.curves_enabled is True
        pts = win._pipeline.curves_rgb_points
        assert len(pts) == 3
        assert abs(pts[1][1] - 0.7) < 1e-6


class TestMainWindowCompareView:
    def test_compare_act_disabled_initially(self, win: MainWindow) -> None:
        assert not win._compare_act.isEnabled()

    def test_compare_act_unchecked_initially(self, win: MainWindow) -> None:
        assert not win._compare_act.isChecked()

    def test_view_stack_shows_viewer_initially(self, win: MainWindow) -> None:
        assert win._view_stack.currentIndex() == 0

    def test_before_image_none_initially(self, win: MainWindow) -> None:
        assert win._before_image is None

    def test_toggle_compare_switches_to_index_1(self, win: MainWindow) -> None:
        win._compare_act.setEnabled(True)
        win._on_toggle_compare(True)
        assert win._view_stack.currentIndex() == 1

    def test_toggle_compare_off_switches_to_index_0(self, win: MainWindow) -> None:
        win._view_stack.setCurrentIndex(1)
        win._on_toggle_compare(False)
        assert win._view_stack.currentIndex() == 0

    def test_reset_compare_state_clears_all(self, win: MainWindow) -> None:
        win._before_image = np.ones((4, 4), dtype=np.float32)
        win._compare_act.setEnabled(True)
        win._compare_act.setChecked(True)
        win._view_stack.setCurrentIndex(1)
        win._reset_compare_state()
        assert win._before_image is None
        assert not win._compare_act.isEnabled()
        assert not win._compare_act.isChecked()
        assert win._view_stack.currentIndex() == 0

    def test_pipeline_finished_with_before_image_enables_compare(
        self, win: MainWindow
    ) -> None:
        from astroai.core.pipeline.base import PipelineContext
        win._before_image = np.zeros((4, 4), dtype=np.float32)
        ctx = PipelineContext()
        ctx.result = np.ones((4, 4), dtype=np.float32)
        win._on_pipeline_finished(ctx)
        assert win._compare_act.isEnabled()
        assert win._compare_act.isChecked()
        assert win._view_stack.currentIndex() == 1

    def test_pipeline_finished_without_before_image_no_compare(
        self, win: MainWindow
    ) -> None:
        from astroai.core.pipeline.base import PipelineContext
        win._before_image = None
        ctx = PipelineContext()
        ctx.result = np.ones((4, 4), dtype=np.float32)
        win._on_pipeline_finished(ctx)
        assert not win._compare_act.isEnabled()

    def test_new_project_resets_compare(self, win: MainWindow) -> None:
        win._compare_act.setEnabled(True)
        win._compare_act.setChecked(True)
        win._before_image = np.zeros((4, 4), dtype=np.float32)
        win._on_new_project()
        assert not win._compare_act.isEnabled()
        assert win._before_image is None


class TestMainWindowFrameSelection:
    def test_on_frame_selection_changed_updates_project(
        self, win: MainWindow
    ) -> None:
        from astroai.project.project_file import FrameEntry
        win._project.input_frames = [
            FrameEntry(path="/a.fits", selected=True),
            FrameEntry(path="/b.fits", selected=True),
        ]
        win._on_frame_selection_changed(0, False)
        assert win._project.input_frames[0].selected is False

    def test_on_frame_selection_changed_enables_run_when_selected(
        self, win: MainWindow
    ) -> None:
        from astroai.project.project_file import FrameEntry
        win._project.input_frames = [FrameEntry(path="/a.fits", selected=False)]
        win._on_frame_selection_changed(0, True)
        assert win._stack_run_act.isEnabled()

    def test_on_frame_selection_changed_disables_run_when_none_selected(
        self, win: MainWindow
    ) -> None:
        from astroai.project.project_file import FrameEntry
        win._project.input_frames = [FrameEntry(path="/a.fits", selected=True)]
        win._on_frame_selection_changed(0, False)
        assert not win._stack_run_act.isEnabled()

    def test_on_frame_selection_changed_out_of_range_no_crash(
        self, win: MainWindow
    ) -> None:
        win._project.input_frames = []
        win._on_frame_selection_changed(99, False)  # must not raise

    def test_on_frames_remove_requested_removes_entries(
        self, win: MainWindow
    ) -> None:
        from astroai.project.project_file import FrameEntry
        win._project.input_frames = [
            FrameEntry(path="/a.fits"),
            FrameEntry(path="/b.fits"),
            FrameEntry(path="/c.fits"),
        ]
        win._on_frames_remove_requested([0, 2])
        assert len(win._project.input_frames) == 1
        assert win._project.input_frames[0].path == "/b.fits"

    def test_on_frames_remove_requested_refreshes_panel(
        self, win: MainWindow
    ) -> None:
        from astroai.project.project_file import FrameEntry
        win._project.input_frames = [FrameEntry(path="/a.fits")]
        win._on_frames_remove_requested([0])
        assert win._frame_list_panel._table.rowCount() == 0

    def test_on_frames_remove_requested_out_of_range_no_crash(
        self, win: MainWindow
    ) -> None:
        win._project.input_frames = []
        win._on_frames_remove_requested([99])  # must not raise


class TestMainWindowProjectDirty:
    def test_new_project_title_has_no_asterisk(self, win: MainWindow) -> None:
        assert not win._project.is_dirty
        assert "*" not in win.windowTitle() or not win.isWindowModified()

    def test_touch_marks_project_dirty(self, win: MainWindow) -> None:
        win._project.touch()
        assert win._project.is_dirty

    def test_update_title_sets_window_modified(self, win: MainWindow) -> None:
        win._project.touch()
        win._update_title()
        assert win.isWindowModified()

    def test_save_project_marks_clean(self, win: MainWindow, tmp_path) -> None:
        from unittest.mock import patch
        win._project.touch()
        assert win._project.is_dirty
        path = tmp_path / "test.astroai"
        with patch.object(win, "_on_save_project_as"):
            win._save_project(path)
        assert not win._project.is_dirty

    def test_save_project_clears_window_modified(self, win: MainWindow, tmp_path) -> None:
        from unittest.mock import patch
        win._project.touch()
        win._update_title()
        path = tmp_path / "test.astroai"
        with patch.object(win, "_on_save_project_as"):
            win._save_project(path)
        assert not win.isWindowModified()

    def test_frame_selection_change_marks_dirty(self, win: MainWindow) -> None:
        from astroai.project.project_file import FrameEntry
        win._project.input_frames = [FrameEntry(path="/a.fits", selected=True)]
        win._on_frame_selection_changed(0, False)
        assert win._project.is_dirty

    def test_frames_remove_marks_dirty(self, win: MainWindow) -> None:
        from astroai.project.project_file import FrameEntry
        win._project.input_frames = [FrameEntry(path="/a.fits")]
        win._on_frames_remove_requested([0])
        assert win._project.is_dirty

    def test_session_notes_change_marks_dirty(self, win: MainWindow) -> None:
        win._on_session_notes_changed("neue Notiz")
        assert win._project.is_dirty

    def test_close_event_accepted_when_clean(self, win: MainWindow) -> None:
        from PySide6.QtGui import QCloseEvent
        event = QCloseEvent()
        win.closeEvent(event)
        assert event.isAccepted()

    def test_maybe_discard_returns_true_when_clean(self, win: MainWindow) -> None:
        assert win._maybe_discard_changes() is True


class TestAutoStretch:
    def test_apply_auto_stretch_returns_float32(self) -> None:
        import numpy as np
        from astroai.ui.main.app import _apply_auto_stretch
        data = np.random.rand(64, 64).astype(np.float32) * 0.01  # dark image
        result = _apply_auto_stretch(data)
        assert result.dtype == np.float32

    def test_apply_auto_stretch_range_zero_to_one(self) -> None:
        import numpy as np
        from astroai.ui.main.app import _apply_auto_stretch
        data = np.random.rand(64, 64).astype(np.float32) * 0.01
        result = _apply_auto_stretch(data)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_apply_auto_stretch_does_not_modify_source(self) -> None:
        import numpy as np
        from astroai.ui.main.app import _apply_auto_stretch
        data = np.random.rand(32, 32).astype(np.float32) * 0.01
        original = data.copy()
        _apply_auto_stretch(data)
        np.testing.assert_array_equal(data, original)

    def test_apply_auto_stretch_3d_array(self) -> None:
        import numpy as np
        from astroai.ui.main.app import _apply_auto_stretch
        data = np.random.rand(3, 64, 64).astype(np.float32) * 0.01
        result = _apply_auto_stretch(data)
        assert result.shape == data.shape
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_apply_auto_stretch_constant_array_no_crash(self) -> None:
        import numpy as np
        from astroai.ui.main.app import _apply_auto_stretch
        data = np.full((16, 16), 0.5, dtype=np.float32)
        result = _apply_auto_stretch(data)  # hi == lo case
        assert result is not None

    def test_toggle_auto_stretch_sets_state(self, win: MainWindow) -> None:
        win._on_toggle_auto_stretch(True)
        assert win._auto_stretch is True
        win._on_toggle_auto_stretch(False)
        assert win._auto_stretch is False

    def test_auto_stretch_action_initially_disabled(self, win: MainWindow) -> None:
        assert not win._auto_stretch_act.isEnabled()

    def test_auto_stretch_action_enabled_after_image_load(
        self, win: MainWindow
    ) -> None:
        import numpy as np
        img = np.random.rand(64, 64).astype(np.float32)
        win._on_image_loaded(img, "test.fits")
        assert win._auto_stretch_act.isEnabled()

    def test_toggle_with_image_updates_viewer(self, win: MainWindow) -> None:
        import numpy as np
        img = np.random.rand(64, 64).astype(np.float32) * 0.01
        win._on_image_loaded(img, "test.fits")
        win._on_toggle_auto_stretch(True)
        # Viewer should have data (not None)
        assert win._viewer._raw_data is not None


class TestCopyImageToClipboard:
    def test_copy_action_initially_disabled(self, win: MainWindow) -> None:
        assert not win._copy_image_act.isEnabled()

    def test_copy_action_enabled_after_image_load(
        self, win: MainWindow
    ) -> None:
        import numpy as np
        img = np.random.rand(64, 64).astype(np.float32)
        win._on_image_loaded(img, "test.fits")
        assert win._copy_image_act.isEnabled()

    def test_on_copy_image_no_crash_without_image(
        self, win: MainWindow
    ) -> None:
        win._on_copy_image()  # must not raise (viewer has no data)

    def test_on_copy_image_updates_status(self, win: MainWindow) -> None:
        import numpy as np
        img = np.random.rand(32, 32).astype(np.float32)
        win._on_image_loaded(img, "test.fits")
        win._on_copy_image()
        assert "Zwischenablage" in win._status_bar.currentMessage()

    def test_on_copy_image_puts_image_on_clipboard(
        self, win: MainWindow
    ) -> None:
        import numpy as np
        from PySide6.QtWidgets import QApplication
        img = np.random.rand(32, 32).astype(np.float32)
        win._on_image_loaded(img, "test.fits")
        win._on_copy_image()
        cb = QApplication.clipboard().image()
        assert not cb.isNull()
        assert cb.width() == 32
        assert cb.height() == 32


class TestSavePreviewImage:
    def test_save_preview_action_initially_disabled(self, win: MainWindow) -> None:
        assert not win._save_preview_act.isEnabled()

    def test_save_preview_enabled_after_image_load(self, win: MainWindow) -> None:
        import numpy as np
        img = np.random.rand(32, 32).astype(np.float32)
        win._on_image_loaded(img, "test.fits")
        assert win._save_preview_act.isEnabled()

    def test_on_save_preview_no_crash_without_image(self, win: MainWindow, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: ("", ""))
        win._on_save_preview_image()  # must not raise

    def test_on_save_preview_writes_file(self, win: MainWindow, tmp_path, monkeypatch) -> None:
        import numpy as np
        from PySide6.QtWidgets import QFileDialog

        out_path = str(tmp_path / "preview.png")
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: (out_path, "PNG-Bild (*.png)"),
        )
        img = np.random.rand(32, 32).astype(np.float32)
        win._on_image_loaded(img, "test.fits")
        win._on_save_preview_image()
        assert (tmp_path / "preview.png").exists()

    def test_on_save_preview_updates_status(self, win: MainWindow, tmp_path, monkeypatch) -> None:
        import numpy as np
        from PySide6.QtWidgets import QFileDialog

        out_path = str(tmp_path / "out.png")
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: (out_path, "PNG-Bild (*.png)"),
        )
        img = np.random.rand(32, 32).astype(np.float32)
        win._on_image_loaded(img, "test.fits")
        win._on_save_preview_image()
        assert "out.png" in win._status_bar.currentMessage()

    def test_on_save_preview_no_dialog_if_no_image(self, win: MainWindow, monkeypatch) -> None:
        called = []
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            lambda *a, **kw: called.append(True) or ("", ""),
        )
        win._on_save_preview_image()
        assert called == []  # dialog not opened when no image loaded


class TestCalibStatusLabel:
    def test_calib_status_label_exists(self, win: MainWindow) -> None:
        assert hasattr(win, "_calib_status_label")

    def test_initial_label_is_dash(self, win: MainWindow) -> None:
        assert "—" in win._calib_status_label.text()

    def test_refresh_with_darks(self, win: MainWindow) -> None:
        win._project.calibration.dark_frames = ["/tmp/d1.fits", "/tmp/d2.fits"]
        win._refresh_calib_status()
        assert "2D" in win._calib_status_label.text()

    def test_refresh_with_flats(self, win: MainWindow) -> None:
        win._project.calibration.flat_frames = ["/tmp/f1.fits"]
        win._refresh_calib_status()
        assert "1F" in win._calib_status_label.text()

    def test_refresh_with_darks_and_flats(self, win: MainWindow) -> None:
        win._project.calibration.dark_frames = ["/d.fits"]
        win._project.calibration.flat_frames = ["/f1.fits", "/f2.fits"]
        win._refresh_calib_status()
        text = win._calib_status_label.text()
        assert "1D" in text
        assert "2F" in text

    def test_refresh_empty_config_shows_dash(self, win: MainWindow) -> None:
        win._project.calibration.dark_frames = []
        win._project.calibration.flat_frames = []
        win._refresh_calib_status()
        assert "—" in win._calib_status_label.text()
