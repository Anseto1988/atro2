"""Tests for PhotometryPanel widget."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from astroai.engine.photometry.models import PhotometryResult, StarMeasurement
from astroai.ui.widgets.photometry_panel import PhotometryPanel


def _make_result(n: int = 3) -> PhotometryResult:
    stars = [
        StarMeasurement(
            star_id=i,
            ra=10.0 + i * 0.1,
            dec=20.0 + i * 0.1,
            x_pixel=100.0 + i,
            y_pixel=200.0 + i,
            instr_mag=-5.0 + i * 0.5,
            catalog_mag=-4.8 + i * 0.5,
            cal_mag=-4.9 + i * 0.5,
            residual=0.1 * i,
        )
        for i in range(n)
    ]
    return PhotometryResult(stars=stars, r_squared=0.9876, n_matched=n)


class TestPhotometryPanel:
    @pytest.fixture()
    def panel(self, qtbot) -> PhotometryPanel:
        w = PhotometryPanel()
        qtbot.addWidget(w)
        return w

    def test_initial_state_empty(self, panel: PhotometryPanel) -> None:
        assert panel._result is None
        assert panel._table.rowCount() == 0
        assert "Keine Photometrie-Daten" in panel._summary_label.text()
        assert not panel._csv_btn.isEnabled()
        assert not panel._fits_btn.isEnabled()

    def test_set_result_populates_table(self, panel: PhotometryPanel) -> None:
        result = _make_result(3)
        panel.set_result(result)
        assert panel._table.rowCount() == 3
        assert panel._csv_btn.isEnabled()
        assert panel._fits_btn.isEnabled()

    def test_set_result_updates_summary(self, panel: PhotometryPanel) -> None:
        result = _make_result(5)
        panel.set_result(result)
        assert "5" in panel._summary_label.text()
        assert "0.9876" in panel._summary_label.text()

    def test_set_result_star_values(self, panel: PhotometryPanel) -> None:
        result = _make_result(1)
        panel.set_result(result)
        star = result.stars[0]
        item_id = panel._table.item(0, 0)
        assert item_id is not None
        assert item_id.data(2) == star.star_id  # DisplayRole == 2

    def test_set_result_none_clears(self, panel: PhotometryPanel) -> None:
        panel.set_result(_make_result(2))
        panel.set_result(None)
        assert panel._table.rowCount() == 0
        assert not panel._csv_btn.isEnabled()
        assert not panel._fits_btn.isEnabled()
        assert "Keine Photometrie-Daten" in panel._summary_label.text()

    def test_clear(self, panel: PhotometryPanel) -> None:
        panel.set_result(_make_result(4))
        panel.clear()
        assert panel._result is None
        assert panel._table.rowCount() == 0
        assert not panel._csv_btn.isEnabled()

    def test_set_result_empty_stars_shows_no_data(self, panel: PhotometryPanel) -> None:
        result = PhotometryResult(stars=[], r_squared=0.0, n_matched=0)
        panel.set_result(result)
        assert panel._table.rowCount() == 0
        assert "Keine Photometrie-Daten" in panel._summary_label.text()

    def test_table_not_editable(self, panel: PhotometryPanel) -> None:
        from PySide6.QtWidgets import QTableWidget
        assert panel._table.editTriggers() == QTableWidget.EditTrigger.NoEditTriggers

    def test_table_selects_rows(self, panel: PhotometryPanel) -> None:
        from PySide6.QtWidgets import QTableWidget
        assert panel._table.selectionBehavior() == QTableWidget.SelectionBehavior.SelectRows

    def test_accessible_names(self, panel: PhotometryPanel) -> None:
        assert panel.accessibleName() == "Photometrie-Ergebnisse"
        assert panel._csv_btn.accessibleName()
        assert panel._fits_btn.accessibleName()

    @patch("astroai.ui.widgets.photometry_panel.QFileDialog.getSaveFileName")
    def test_export_csv_triggers_dialog(
        self, mock_dialog: MagicMock, panel: PhotometryPanel, tmp_path
    ) -> None:
        csv_path = str(tmp_path / "test.csv")
        mock_dialog.return_value = (csv_path, "")
        panel.set_result(_make_result(2))
        panel._export_csv()
        mock_dialog.assert_called_once()
        assert (tmp_path / "test.csv").exists()

    @patch("astroai.ui.widgets.photometry_panel.QFileDialog.getSaveFileName")
    def test_export_fits_triggers_dialog(
        self, mock_dialog: MagicMock, panel: PhotometryPanel, tmp_path
    ) -> None:
        fits_path = str(tmp_path / "test.fits")
        mock_dialog.return_value = (fits_path, "")
        panel.set_result(_make_result(2))
        panel._export_fits()
        mock_dialog.assert_called_once()
        assert (tmp_path / "test.fits").exists()

    @patch("astroai.ui.widgets.photometry_panel.QFileDialog.getSaveFileName")
    def test_export_csv_cancelled(
        self, mock_dialog: MagicMock, panel: PhotometryPanel
    ) -> None:
        mock_dialog.return_value = ("", "")
        panel.set_result(_make_result(1))
        panel._export_csv()
        mock_dialog.assert_called_once()

    @patch("astroai.ui.widgets.photometry_panel.QFileDialog.getSaveFileName")
    def test_export_csv_no_result(
        self, mock_dialog: MagicMock, panel: PhotometryPanel
    ) -> None:
        panel._export_csv()
        mock_dialog.assert_not_called()

    def test_columns_count(self, panel: PhotometryPanel) -> None:
        assert panel._table.columnCount() == 7
