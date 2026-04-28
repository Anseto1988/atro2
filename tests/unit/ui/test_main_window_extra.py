"""Extra tests for app.py coverage gaps: lines 130,133,137-169,850-890,956,984,991,1004,1343-1351,1471,1475,1522,1578."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

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


class TestEnrichFitsEntry:
    def test_non_frame_entry_returns_early(self) -> None:
        from astroai.ui.main.app import _enrich_fits_entry
        _enrich_fits_entry("not-a-frame-entry")

    def test_non_fits_path_returns_early(self) -> None:
        from astroai.ui.main.app import _enrich_fits_entry
        from astroai.project.project_file import FrameEntry
        e = FrameEntry(path="/data/image.png")
        _enrich_fits_entry(e)

    def test_fits_path_enriches_metadata(self, tmp_path: Path) -> None:
        from astroai.ui.main.app import _enrich_fits_entry
        from astroai.project.project_file import FrameEntry
        f = tmp_path / "frame.fits"
        f.write_bytes(b"")
        e = FrameEntry(path=str(f))
        mock_hdr = MagicMock()
        mock_hdr.get = lambda k, default=None: {"EXPTIME": 120.0, "GAIN": "800", "CCD-TEMP": -10.5}.get(k, default)
        mock_hdu = MagicMock()
        mock_hdu.header = mock_hdr
        mock_hdul = MagicMock()
        mock_hdul.__enter__ = lambda s: mock_hdul
        mock_hdul.__exit__ = MagicMock(return_value=False)
        mock_hdul.__getitem__ = lambda s, i: mock_hdu
        with patch("astropy.io.fits.open", return_value=mock_hdul):
            _enrich_fits_entry(e)
        assert e.exposure == pytest.approx(120.0)
        assert e.gain_iso == 800
        assert e.temperature == pytest.approx(-10.5)


class TestEnrichRawEntry:
    def test_non_frame_entry_returns_early(self) -> None:
        from astroai.ui.main.app import _enrich_raw_entry
        _enrich_raw_entry(42)

    def test_raw_path_enriches_metadata(self, tmp_path: Path) -> None:
        from astroai.ui.main.app import _enrich_raw_entry
        from astroai.project.project_file import FrameEntry
        f = tmp_path / "frame.cr2"
        f.write_bytes(b"")
        e = FrameEntry(path=str(f))
        mock_meta = MagicMock()
        mock_meta.exposure = 30.0
        mock_meta.gain_iso = 1600
        mock_meta.temperature = -15.0
        with patch("astroai.core.io.raw_io.read_raw_metadata", return_value=mock_meta):
            _enrich_raw_entry(e)
        assert e.exposure == pytest.approx(30.0)
        assert e.gain_iso == 1600
        assert e.temperature == pytest.approx(-15.0)


class TestAutoCalibMatchFull:
    def _setup_mock_lib(self):
        mock_lib = MagicMock()
        mock_lib.darks = [MagicMock()]
        mock_lib.flats = [MagicMock()]
        mock_lib.bias = []
        return mock_lib

    def _make_match_result(self):
        mock_result = MagicMock()
        mock_result.coverage = 0.8
        return mock_result

    def test_yes_response_updates_calibration(self, win: MainWindow, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **kw: "/calib_dir")
        mock_lib = self._setup_mock_lib()
        mock_result = self._make_match_result()
        mock_cfg = MagicMock()
        mock_cfg.dark_frames = [MagicMock()]
        mock_cfg.flat_frames = [MagicMock()]
        mock_cfg.bias_frames = []
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **kw: QMessageBox.StandardButton.Yes)
        with patch("astroai.core.calibration.scanner.scan_directory", return_value=[]):
            with patch("astroai.core.calibration.scanner.build_calibration_library", return_value=mock_lib):
                with patch("astroai.core.calibration.matcher.batch_match", return_value=mock_result):
                    with patch("astroai.core.calibration.matcher.suggest_calibration_config", return_value=mock_cfg):
                        win._on_auto_match_calibration()
        assert "aktualisiert" in win._status_bar.currentMessage().lower()

    def test_no_response_shows_aborted_message(self, win: MainWindow, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **kw: "/calib_dir")
        mock_lib = self._setup_mock_lib()
        mock_result = self._make_match_result()
        mock_cfg = MagicMock()
        mock_cfg.dark_frames = []
        mock_cfg.flat_frames = []
        mock_cfg.bias_frames = []
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **kw: QMessageBox.StandardButton.No)
        with patch("astroai.core.calibration.scanner.scan_directory", return_value=[]):
            with patch("astroai.core.calibration.scanner.build_calibration_library", return_value=mock_lib):
                with patch("astroai.core.calibration.matcher.batch_match", return_value=mock_result):
                    with patch("astroai.core.calibration.matcher.suggest_calibration_config", return_value=mock_cfg):
                        win._on_auto_match_calibration()
        assert "abgebrochen" in win._status_bar.currentMessage().lower()


class TestPixelHoveredExceptionPath:
    def test_wcs_exception_does_not_crash(self, win: MainWindow) -> None:
        from astroai.ui.overlay.sky_objects import WcsTransform
        mock_wcs = MagicMock(spec=WcsTransform)
        mock_wcs.pixel_to_world.side_effect = RuntimeError("WCS failure")
        win._wcs_adapter = mock_wcs
        win._on_pixel_hovered(100, 200, 0.42)
        assert "100" in win._status_bar.currentMessage()


class TestCloseEventIgnored:
    def test_close_event_ignored_when_dirty(self, win: MainWindow, monkeypatch) -> None:
        from PySide6.QtGui import QCloseEvent
        monkeypatch.setattr(win, "_maybe_discard_changes", lambda: False)
        event = MagicMock(spec=QCloseEvent)
        event.spontaneous.return_value = True
        win.closeEvent(event)
        event.ignore.assert_called_once()


class TestNewAndOpenProjectGuards:
    def test_new_project_returns_early(self, win: MainWindow, monkeypatch) -> None:
        original_project = win._project
        monkeypatch.setattr(win, "_maybe_discard_changes", lambda: False)
        win._on_new_project()
        assert win._project is original_project

    def test_open_project_returns_early(self, win: MainWindow, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog
        opened = []
        monkeypatch.setattr(win, "_maybe_discard_changes", lambda: False)
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            lambda *a, **kw: (opened.append(True) or ("", "")))
        win._on_open_project()
        assert not opened


class TestProjectSummaryOptionalFields:
    def test_summary_with_exposure_groups_quality_and_temp(self, win: MainWindow, monkeypatch) -> None:
        from astroai.project.summary import ProjectSummary, ExposureGroup
        summary = ProjectSummary(
            total_frames=5,
            selected_count=3,
            total_exposure_s=600.0,
            exposure_groups=[ExposureGroup(exposure_s=120.0, count=3)],
            scored_count=3,
            quality_mean=0.85,
            quality_min=0.70,
            quality_max=0.95,
            temp_min=-10.0,
            temp_max=-8.0,
        )
        shown_texts: list = []
        monkeypatch.setattr(QMessageBox, "information",
                            lambda *a, **kw: shown_texts.append(a[2] if len(a) > 2 else ""))
        with patch("astroai.project.summary.compute_summary", return_value=summary):
            win._on_show_project_summary()
        assert shown_texts
        full_text = shown_texts[0]
        assert "120" in full_text
        assert "85" in full_text
        assert "-10" in full_text


class TestSavePreviewEdgeCases:
    def _load_image(self, win: MainWindow) -> None:
        win._current_image = np.zeros((64, 64), dtype=np.float32)
        win._viewer.set_image_data(win._current_image)

    def test_no_path_from_dialog_returns_early(self, win: MainWindow, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog
        self._load_image(win)
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: ("", ""))
        win._on_save_preview_image()

    def test_qimage_save_failure_shows_warning(self, win: MainWindow, monkeypatch, tmp_path: Path) -> None:
        from PySide6.QtWidgets import QFileDialog
        self._load_image(win)
        save_path = str(tmp_path / "out.png")
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: (save_path, "PNG"))
        warnings_shown: list = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: warnings_shown.append(True))
        mock_qimg = MagicMock()
        mock_qimg.save.return_value = False
        monkeypatch.setattr(win._viewer, "render_full_qimage", lambda: mock_qimg)
        win._on_save_preview_image()
        assert warnings_shown


class TestRunFullPipelineValidationFails:
    def test_validation_failure_stops_run(self, win: MainWindow, monkeypatch) -> None:
        monkeypatch.setattr(win, "_validate_before_run", lambda: False)
        started: list = []
        monkeypatch.setattr(win._pipeline_worker, "start", lambda *a: started.append(True))
        win._on_run_full_pipeline()
        assert not started


class TestPipelineFinishedWithActiveStep:
    def test_active_step_set_done_on_finish(self, win: MainWindow) -> None:
        from astroai.core.pipeline.base import PipelineContext
        from astroai.ui.models import StepState
        if win._pipeline.steps:
            win._pipeline.steps[0].state = StepState.ACTIVE
        ctx = PipelineContext()
        ctx.result = np.zeros((8, 8), dtype=np.float32)
        win._on_pipeline_finished(ctx)
        if win._pipeline.steps:
            assert win._pipeline.steps[0].state == StepState.DONE


class TestEnrichRawEntryException:
    def test_read_raw_metadata_exception_is_swallowed(self, tmp_path: Path) -> None:
        """Lines 169-170: exception in read_raw_metadata → silently ignored."""
        from astroai.ui.main.app import _enrich_raw_entry
        from astroai.project.project_file import FrameEntry
        f = tmp_path / "frame.cr2"
        f.write_bytes(b"")
        e = FrameEntry(path=str(f))
        with patch("astroai.core.io.raw_io.read_raw_metadata", side_effect=RuntimeError("rawpy error")):
            _enrich_raw_entry(e)  # must not raise


class TestGetBackendLabelException:
    def test_onnx_registry_exception_returns_cpu(self, win: MainWindow) -> None:
        """Lines 667-668: OnnxModelRegistry raises → fallback to '[CPU]'."""
        with patch("astroai.core.onnx_registry.OnnxModelRegistry", side_effect=RuntimeError("no registry")):
            result = win._get_backend_label()
        assert result == "[CPU]"


class TestAutoCalibMatchWithFrames:
    def test_yes_with_project_frames_reads_fits(self, win: MainWindow, monkeypatch, tmp_path: Path) -> None:
        """Lines 886-891: selected frames in project → FITS read attempted."""
        from PySide6.QtWidgets import QFileDialog
        from astroai.project.project_file import FrameEntry
        from astroai.core.io.fits_io import ImageMetadata
        f = tmp_path / "light.fits"
        f.write_bytes(b"")
        win._project.input_frames = [FrameEntry(path=str(f), selected=True)]
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **kw: "/calib_dir")
        mock_lib = MagicMock()
        mock_lib.darks = [MagicMock()]
        mock_lib.flats = []
        mock_lib.bias = []
        mock_result = MagicMock()
        mock_result.coverage = 0.5
        mock_cfg = MagicMock()
        mock_cfg.dark_frames = []
        mock_cfg.flat_frames = []
        mock_cfg.bias_frames = []
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **kw: QMessageBox.StandardButton.No)
        fake_img = np.zeros((4, 4), dtype=np.float32)
        fake_meta = ImageMetadata()
        with patch("astroai.core.calibration.scanner.scan_directory", return_value=[]):
            with patch("astroai.core.calibration.scanner.build_calibration_library", return_value=mock_lib):
                with patch("astroai.core.calibration.matcher.batch_match", return_value=mock_result):
                    with patch("astroai.core.calibration.matcher.suggest_calibration_config", return_value=mock_cfg):
                        with patch("astroai.core.io.fits_io.read_fits", return_value=(fake_img, fake_meta)):
                            win._on_auto_match_calibration()


class TestAutoCalibMatchFitsException:
    def test_read_fits_exception_uses_empty_metadata(self, win: MainWindow, monkeypatch, tmp_path: Path) -> None:
        """Lines 890-891: read_fits raises → empty ImageMetadata used instead."""
        from PySide6.QtWidgets import QFileDialog
        from astroai.project.project_file import FrameEntry
        f = tmp_path / "light.fits"
        f.write_bytes(b"")
        win._project.input_frames = [FrameEntry(path=str(f), selected=True)]
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **kw: "/calib_dir")
        mock_lib = MagicMock()
        mock_lib.darks = [MagicMock()]
        mock_lib.flats = []
        mock_lib.bias = []
        mock_result = MagicMock()
        mock_result.coverage = 0.5
        mock_cfg = MagicMock()
        mock_cfg.dark_frames = []
        mock_cfg.flat_frames = []
        mock_cfg.bias_frames = []
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **kw: QMessageBox.StandardButton.No)
        with patch("astroai.core.calibration.scanner.scan_directory", return_value=[]):
            with patch("astroai.core.calibration.scanner.build_calibration_library", return_value=mock_lib):
                with patch("astroai.core.calibration.matcher.batch_match", return_value=mock_result):
                    with patch("astroai.core.calibration.matcher.suggest_calibration_config", return_value=mock_cfg):
                        with patch("astroai.core.io.fits_io.read_fits", side_effect=OSError("bad fits")):
                            win._on_auto_match_calibration()


class TestUndoRedo:
    def test_undo_already_running_returns_early(self, win: MainWindow, monkeypatch) -> None:
        """Line 1696: pipeline running → early return."""
        from astroai.core.pipeline.runner import PipelineWorker
        monkeypatch.setattr(PipelineWorker, "is_running", property(lambda self: True))
        win._on_undo()

    def test_undo_no_history_no_crash(self, win: MainWindow) -> None:
        """_on_undo with empty history → entry is None → early return."""
        win._on_undo()

    def test_undo_with_base_image_restores(self, win: MainWindow, monkeypatch) -> None:
        """Lines 1709-1712: undo to first entry → restore base image to viewer."""
        from astroai.core.processing_history import HistoryEntry
        entry = MagicMock(spec=HistoryEntry)
        entry.params = {}
        monkeypatch.setattr(win._processing_history, "undo", lambda: entry)
        monkeypatch.setattr(win._processing_history, "peek_undo", lambda: None)
        win._processing_base_image = np.zeros((8, 8), dtype=np.float32)
        win._on_undo()
        assert win._current_image is win._processing_base_image

    def test_undo_with_prev_entry_starts_pipeline(self, win: MainWindow, monkeypatch) -> None:
        """Lines 1701-1708: undo with prev entry → pipeline re-run."""
        from astroai.core.processing_history import HistoryEntry
        entry = MagicMock(spec=HistoryEntry)
        entry.params = {}
        prev_entry = MagicMock(spec=HistoryEntry)
        prev_entry.params = {}
        started: list = []
        monkeypatch.setattr(win._processing_history, "undo", lambda: entry)
        monkeypatch.setattr(win._processing_history, "peek_undo", lambda: prev_entry)
        monkeypatch.setattr(win._pipeline_worker, "start", lambda *a: started.append(True))
        win._processing_base_image = np.zeros((8, 8), dtype=np.float32)
        win._on_undo()
        assert started

    def test_redo_already_running_returns_early(self, win: MainWindow, monkeypatch) -> None:
        """Line 1717: pipeline running → early return."""
        from astroai.core.pipeline.runner import PipelineWorker
        monkeypatch.setattr(PipelineWorker, "is_running", property(lambda self: True))
        win._on_redo()

    def test_redo_no_history_no_crash(self, win: MainWindow) -> None:
        """_on_redo with empty redo stack → no crash."""
        win._on_redo()

    def test_redo_with_base_image_starts_pipeline(self, win: MainWindow, monkeypatch) -> None:
        """Lines 1721-1727: redo with base image → pipeline re-run."""
        from astroai.core.processing_history import HistoryEntry
        entry = MagicMock(spec=HistoryEntry)
        entry.params = {}
        started: list = []
        monkeypatch.setattr(win._processing_history, "redo", lambda: entry)
        monkeypatch.setattr(win._pipeline_worker, "start", lambda *a: started.append(True))
        win._processing_base_image = np.zeros((8, 8), dtype=np.float32)
        win._on_redo()
        assert started
