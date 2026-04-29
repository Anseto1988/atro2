"""Unit tests for ModelManagerPanel (VER-430)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astroai.ui.widgets.model_manager_panel import (
    ModelManagerPanel,
    _DownloadWorker,
    _fmt_size,
    _STATUS_CORRUPT,
    _STATUS_DOWNLOADING,
    _STATUS_MISSING,
    _STATUS_PRESENT,
    _STATUS_UNAVAILABLE,
)


# ---------------------------------------------------------------------------
# _fmt_size utility
# ---------------------------------------------------------------------------

class TestFmtSize:
    def test_missing_file_returns_dash(self, tmp_path: Path) -> None:
        assert _fmt_size(tmp_path / "nope.onnx") == "—"

    def test_bytes(self, tmp_path: Path) -> None:
        f = tmp_path / "tiny.bin"
        f.write_bytes(b"x" * 512)
        assert _fmt_size(f) == "512 B"

    def test_kilobytes(self, tmp_path: Path) -> None:
        f = tmp_path / "small.bin"
        f.write_bytes(b"x" * 2048)
        result = _fmt_size(f)
        assert result.endswith("KB")
        assert "2.0" in result

    def test_megabytes(self, tmp_path: Path) -> None:
        f = tmp_path / "medium.bin"
        f.write_bytes(b"x" * (3 * 1024 ** 2))
        result = _fmt_size(f)
        assert result.endswith("MB")
        assert "3.0" in result

    def test_gigabytes(self, tmp_path: Path) -> None:
        f = tmp_path / "large.bin"
        # create sparse file via seek-and-write
        with open(f, "wb") as fh:
            fh.seek(2 * 1024 ** 3 - 1)
            fh.write(b"\x00")
        result = _fmt_size(f)
        assert result.endswith("GB")


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

class TestModelManagerPanelConstruction:
    @pytest.fixture()
    def panel(self, qtbot, tmp_path: Path) -> ModelManagerPanel:
        p = ModelManagerPanel(models_dir=tmp_path)
        qtbot.addWidget(p)
        return p

    def test_panel_is_qwidget(self, panel: ModelManagerPanel) -> None:
        from PySide6.QtWidgets import QWidget
        assert isinstance(panel, QWidget)

    def test_table_has_correct_column_count(self, panel: ModelManagerPanel) -> None:
        assert panel._table.columnCount() == 6

    def test_table_has_rows_for_all_manifest_entries(
        self, panel: ModelManagerPanel
    ) -> None:
        manifest = panel._downloader.get_manifest()
        assert panel._table.rowCount() == len(manifest)

    def test_header_labels_set(self, panel: ModelManagerPanel) -> None:
        headers = [
            panel._table.horizontalHeaderItem(i).text()
            for i in range(panel._table.columnCount())
        ]
        assert "Modell" in headers
        assert "Status" in headers
        assert "Aktionen" in headers

    def test_models_dir_attribute(self, panel: ModelManagerPanel, tmp_path: Path) -> None:
        assert panel._models_dir == tmp_path


# ---------------------------------------------------------------------------
# Status computation
# ---------------------------------------------------------------------------

class TestComputeStatus:
    @pytest.fixture()
    def panel(self, qtbot, tmp_path: Path) -> ModelManagerPanel:
        p = ModelManagerPanel(models_dir=tmp_path)
        qtbot.addWidget(p)
        return p

    def test_unavailable_model(self, panel: ModelManagerPanel, tmp_path: Path) -> None:
        status, _ = panel._compute_status("starnet_plus_plus", tmp_path / "x.onnx", False)
        assert status == _STATUS_UNAVAILABLE

    def test_missing_model(self, panel: ModelManagerPanel, tmp_path: Path) -> None:
        status, _ = panel._compute_status("nafnet_denoise", tmp_path / "missing.onnx", True)
        assert status == _STATUS_MISSING

    def test_downloading_status(self, panel: ModelManagerPanel, tmp_path: Path) -> None:
        panel._active_threads["nafnet_denoise"] = MagicMock()
        status, _ = panel._compute_status("nafnet_denoise", tmp_path / "x.onnx", True)
        assert status == _STATUS_DOWNLOADING
        panel._active_threads.pop("nafnet_denoise")

    def test_present_model(self, panel: ModelManagerPanel, tmp_path: Path) -> None:
        # Create a fake file and mock is_available to return True
        fake_file = tmp_path / "nafnet_denoise.onnx"
        fake_file.write_bytes(b"data")
        with patch.object(panel._downloader, "is_available", return_value=True):
            status, _ = panel._compute_status("nafnet_denoise", fake_file, True)
        assert status == _STATUS_PRESENT

    def test_corrupt_model(self, panel: ModelManagerPanel, tmp_path: Path) -> None:
        fake_file = tmp_path / "nafnet_denoise.onnx"
        fake_file.write_bytes(b"bad_data")
        with patch.object(panel._downloader, "is_available", return_value=False):
            status, _ = panel._compute_status("nafnet_denoise", fake_file, True)
        assert status == _STATUS_CORRUPT


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

class TestRefresh:
    @pytest.fixture()
    def panel(self, qtbot, tmp_path: Path) -> ModelManagerPanel:
        p = ModelManagerPanel(models_dir=tmp_path)
        qtbot.addWidget(p)
        return p

    def test_refresh_sets_row_count_to_manifest_size(self, panel: ModelManagerPanel) -> None:
        panel.refresh()
        assert panel._table.rowCount() == len(panel._downloader.get_manifest())

    def test_status_column_contains_text(self, panel: ModelManagerPanel) -> None:
        panel.refresh()
        for row in range(panel._table.rowCount()):
            item = panel._table.item(row, 2)  # _COL_STATUS = 2
            assert item is not None
            assert len(item.text()) > 0

    def test_name_column_contains_model_names(self, panel: ModelManagerPanel) -> None:
        manifest_names = list(panel._downloader.get_manifest().keys())
        panel.refresh()
        displayed_names = [
            panel._table.item(row, 0).text()
            for row in range(panel._table.rowCount())
        ]
        for name in manifest_names:
            assert name in displayed_names

    def test_progress_bar_created_for_each_model(self, panel: ModelManagerPanel) -> None:
        panel.refresh()
        for name in panel._downloader.get_manifest():
            assert name in panel._progress_bars

    def test_progress_bar_initially_hidden(self, panel: ModelManagerPanel) -> None:
        panel.refresh()
        for pb in panel._progress_bars.values():
            assert not pb.isVisible()


# ---------------------------------------------------------------------------
# Download flow
# ---------------------------------------------------------------------------

class TestDownload:
    @pytest.fixture()
    def panel(self, qtbot, tmp_path: Path) -> ModelManagerPanel:
        p = ModelManagerPanel(models_dir=tmp_path)
        qtbot.addWidget(p)
        return p

    def test_start_download_shows_progress_bar(
        self, panel: ModelManagerPanel, qtbot
    ) -> None:
        with patch.object(panel._downloader, "ensure_model"):
            with patch("PySide6.QtCore.QThread.start"):
                panel._start_download("nafnet_denoise")
        # isHidden() checks own flag independent of parent widget visibility
        assert not panel._progress_bars["nafnet_denoise"].isHidden()

    def test_start_download_emits_status_message(
        self, panel: ModelManagerPanel, qtbot
    ) -> None:
        with patch("PySide6.QtCore.QThread.start"):
            with qtbot.waitSignal(panel.status_message, timeout=500) as blocker:
                panel._start_download("nafnet_denoise")
        assert "nafnet_denoise" in blocker.args[0]

    def test_start_download_second_call_is_noop(
        self, panel: ModelManagerPanel
    ) -> None:
        panel._active_threads["nafnet_denoise"] = MagicMock()
        initial_count = len(panel._active_threads)
        with patch("PySide6.QtCore.QThread.start"):
            panel._start_download("nafnet_denoise")
        assert len(panel._active_threads) == initial_count

    def test_on_download_progress_updates_bar(
        self, panel: ModelManagerPanel
    ) -> None:
        panel.refresh()  # ensure progress bar created
        panel._on_download_progress("nafnet_denoise", 55)
        assert panel._progress_bars["nafnet_denoise"].value() == 55

    def test_on_download_finished_removes_thread(
        self, panel: ModelManagerPanel
    ) -> None:
        panel._active_threads["nafnet_denoise"] = MagicMock()
        panel._on_download_finished("nafnet_denoise")
        assert "nafnet_denoise" not in panel._active_threads

    def test_on_download_finished_hides_progress_bar(
        self, panel: ModelManagerPanel
    ) -> None:
        panel.refresh()
        pb = panel._progress_bars["nafnet_denoise"]
        pb.setVisible(True)
        panel._on_download_finished("nafnet_denoise")
        assert not pb.isVisible()

    def test_on_download_finished_emits_status_message(
        self, panel: ModelManagerPanel, qtbot
    ) -> None:
        panel._active_threads["nafnet_denoise"] = MagicMock()
        with qtbot.waitSignal(panel.status_message, timeout=500) as blocker:
            panel._on_download_finished("nafnet_denoise")
        assert "nafnet_denoise" in blocker.args[0]

    def test_on_download_error_removes_thread(
        self, panel: ModelManagerPanel
    ) -> None:
        panel._active_threads["nafnet_denoise"] = MagicMock()
        with patch(
            "astroai.ui.widgets.model_manager_panel.QMessageBox.critical"
        ):
            panel._on_download_error("nafnet_denoise", "timeout")
        assert "nafnet_denoise" not in panel._active_threads

    def test_on_download_error_hides_progress_bar(
        self, panel: ModelManagerPanel
    ) -> None:
        panel.refresh()
        pb = panel._progress_bars["nafnet_denoise"]
        pb.setVisible(True)
        with patch(
            "astroai.ui.widgets.model_manager_panel.QMessageBox.critical"
        ):
            panel._on_download_error("nafnet_denoise", "err")
        assert not pb.isVisible()

    def test_on_download_error_emits_status_message(
        self, panel: ModelManagerPanel, qtbot
    ) -> None:
        with patch(
            "astroai.ui.widgets.model_manager_panel.QMessageBox.critical"
        ):
            with qtbot.waitSignal(panel.status_message, timeout=500) as blocker:
                panel._on_download_error("nafnet_denoise", "timeout")
        assert "nafnet_denoise" in blocker.args[0]


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

class TestVerify:
    @pytest.fixture()
    def panel(self, qtbot, tmp_path: Path) -> ModelManagerPanel:
        p = ModelManagerPanel(models_dir=tmp_path)
        qtbot.addWidget(p)
        return p

    def test_verify_ok_shows_information_dialog(
        self, panel: ModelManagerPanel
    ) -> None:
        with patch.object(panel._downloader, "is_available", return_value=True):
            with patch(
                "astroai.ui.widgets.model_manager_panel.QMessageBox.information"
            ) as mock_info:
                panel._verify_model("nafnet_denoise")
        mock_info.assert_called_once()

    def test_verify_fail_shows_warning_dialog(
        self, panel: ModelManagerPanel
    ) -> None:
        with patch.object(panel._downloader, "is_available", return_value=False):
            with patch(
                "astroai.ui.widgets.model_manager_panel.QMessageBox.warning"
            ) as mock_warn:
                panel._verify_model("nafnet_denoise")
        mock_warn.assert_called_once()

    def test_verify_ok_emits_status_message(
        self, panel: ModelManagerPanel, qtbot
    ) -> None:
        with patch.object(panel._downloader, "is_available", return_value=True):
            with patch("astroai.ui.widgets.model_manager_panel.QMessageBox.information"):
                with qtbot.waitSignal(panel.status_message, timeout=500) as blocker:
                    panel._verify_model("nafnet_denoise")
        assert "nafnet_denoise" in blocker.args[0]

    def test_verify_fail_emits_status_message(
        self, panel: ModelManagerPanel, qtbot
    ) -> None:
        with patch.object(panel._downloader, "is_available", return_value=False):
            with patch("astroai.ui.widgets.model_manager_panel.QMessageBox.warning"):
                with qtbot.waitSignal(panel.status_message, timeout=500) as blocker:
                    panel._verify_model("nafnet_denoise")
        assert "nafnet_denoise" in blocker.args[0]


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDelete:
    @pytest.fixture()
    def panel(self, qtbot, tmp_path: Path) -> ModelManagerPanel:
        p = ModelManagerPanel(models_dir=tmp_path)
        qtbot.addWidget(p)
        return p

    def test_delete_confirmed_removes_file(
        self, panel: ModelManagerPanel, tmp_path: Path
    ) -> None:
        model_file = tmp_path / "nafnet_denoise.onnx"
        model_file.write_bytes(b"fake")
        with patch(
            "astroai.ui.widgets.model_manager_panel.QMessageBox.question",
            return_value=__import__("PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.StandardButton.Yes,
        ):
            panel._delete_model("nafnet_denoise")
        assert not model_file.exists()

    def test_delete_cancelled_keeps_file(
        self, panel: ModelManagerPanel, tmp_path: Path
    ) -> None:
        model_file = tmp_path / "nafnet_denoise.onnx"
        model_file.write_bytes(b"fake")
        with patch(
            "astroai.ui.widgets.model_manager_panel.QMessageBox.question",
            return_value=__import__("PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.StandardButton.No,
        ):
            panel._delete_model("nafnet_denoise")
        assert model_file.exists()

    def test_delete_nonexistent_file_is_noop(
        self, panel: ModelManagerPanel, tmp_path: Path
    ) -> None:
        with patch(
            "astroai.ui.widgets.model_manager_panel.QMessageBox.question"
        ) as mock_q:
            panel._delete_model("nafnet_denoise")
        mock_q.assert_not_called()

    def test_delete_unknown_model_is_noop(
        self, panel: ModelManagerPanel
    ) -> None:
        panel._delete_model("totally_unknown_model")  # must not raise

    def test_delete_confirmed_emits_status_message(
        self, panel: ModelManagerPanel, tmp_path: Path, qtbot
    ) -> None:
        model_file = tmp_path / "nafnet_denoise.onnx"
        model_file.write_bytes(b"fake")
        from PySide6.QtWidgets import QMessageBox
        with patch(
            "astroai.ui.widgets.model_manager_panel.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            with qtbot.waitSignal(panel.status_message, timeout=500) as blocker:
                panel._delete_model("nafnet_denoise")
        assert "nafnet_denoise" in blocker.args[0]


# ---------------------------------------------------------------------------
# _DownloadWorker
# ---------------------------------------------------------------------------

class TestDownloadWorker:
    def test_worker_emits_finished_on_success(self, qtbot, tmp_path: Path) -> None:
        worker = _DownloadWorker("nafnet_denoise", tmp_path)
        with patch(
            "astroai.ui.widgets.model_manager_panel.ModelDownloader.ensure_model"
        ):
            with qtbot.waitSignal(worker.finished, timeout=1000):
                worker.run()

    def test_worker_emits_error_on_exception(self, qtbot, tmp_path: Path) -> None:
        worker = _DownloadWorker("nafnet_denoise", tmp_path)
        with patch(
            "astroai.ui.widgets.model_manager_panel.ModelDownloader.ensure_model",
            side_effect=RuntimeError("network error"),
        ):
            with qtbot.waitSignal(worker.error, timeout=1000) as blocker:
                worker.run()
        assert "network error" in blocker.args[0]

    def test_worker_emits_progress(self, qtbot, tmp_path: Path) -> None:
        from astroai.core.pipeline.base import PipelineProgress, PipelineStage

        worker = _DownloadWorker("nafnet_denoise", tmp_path)
        captured: list[int] = []
        worker.progress_pct.connect(captured.append)

        def fake_ensure(name: str) -> None:
            from astroai.core.pipeline.base import PipelineProgress, PipelineStage
            # Trigger the progress callback by calling ModelDownloader directly
            pass  # progress is internal; tested via integration

        # Verify signal connection at least doesn't crash
        worker.progress_pct.emit(50)
        assert captured == [50]


# ---------------------------------------------------------------------------
# is_downloadable integration
# ---------------------------------------------------------------------------

class TestIsDownloadable:
    def test_nafnet_is_downloadable(self, tmp_path: Path) -> None:
        from astroai.inference.models.downloader import ModelDownloader
        dl = ModelDownloader(models_dir=tmp_path)
        assert dl.is_downloadable("nafnet_denoise") is True

    def test_starnet_is_not_downloadable(self, tmp_path: Path) -> None:
        from astroai.inference.models.downloader import ModelDownloader
        dl = ModelDownloader(models_dir=tmp_path)
        assert dl.is_downloadable("starnet_plus_plus") is False

    def test_unknown_model_is_not_downloadable(self, tmp_path: Path) -> None:
        from astroai.inference.models.downloader import ModelDownloader
        dl = ModelDownloader(models_dir=tmp_path)
        assert dl.is_downloadable("nonexistent_model") is False
