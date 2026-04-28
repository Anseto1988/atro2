"""Tests for PhotometryPanel widget."""
from __future__ import annotations

import pytest

from astroai.engine.photometry.models import PhotometryResult, StarMeasurement
from astroai.ui.widgets.photometry_panel import PhotometryPanel, _COLUMNS


def _star(star_id=1, ra=83.8, dec=-5.4, instr_mag=-10.5, cal_mag=6.2,
          catalog_mag=6.1, residual=0.01) -> StarMeasurement:
    return StarMeasurement(
        star_id=star_id, ra=ra, dec=dec,
        x_pixel=100.0, y_pixel=200.0,
        instr_mag=instr_mag, cal_mag=cal_mag,
        catalog_mag=catalog_mag, residual=residual,
    )


def _result(n=3, r_sq=0.998) -> PhotometryResult:
    stars = [_star(star_id=i) for i in range(n)]
    return PhotometryResult(stars=stars, r_squared=r_sq, n_matched=n)


@pytest.fixture()
def panel(qtbot):
    w = PhotometryPanel()
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_summary_label_default_text(self, panel):
        assert "Keine" in panel._summary_label.text()

    def test_table_has_correct_columns(self, panel):
        assert panel._table.columnCount() == len(_COLUMNS)

    def test_csv_btn_disabled_initially(self, panel):
        assert not panel._csv_btn.isEnabled()

    def test_fits_btn_disabled_initially(self, panel):
        assert not panel._fits_btn.isEnabled()

    def test_table_empty_initially(self, panel):
        assert panel._table.rowCount() == 0


class TestSetResult:
    def test_set_none_shows_no_data_message(self, panel):
        panel.set_result(None)
        assert "Keine" in panel._summary_label.text()

    def test_set_none_disables_export_buttons(self, panel):
        panel.set_result(_result())
        panel.set_result(None)
        assert not panel._csv_btn.isEnabled()
        assert not panel._fits_btn.isEnabled()

    def test_set_result_populates_table(self, panel):
        panel.set_result(_result(n=3))
        assert panel._table.rowCount() == 3

    def test_set_result_enables_export_buttons(self, panel):
        panel.set_result(_result(n=1))
        assert panel._csv_btn.isEnabled()
        assert panel._fits_btn.isEnabled()

    def test_set_result_shows_summary(self, panel):
        panel.set_result(_result(n=5, r_sq=0.999))
        assert "5" in panel._summary_label.text()
        assert "0.9990" in panel._summary_label.text()

    def test_set_result_empty_stars_shows_no_data(self, panel):
        panel.set_result(PhotometryResult(stars=[], r_squared=0.0, n_matched=0))
        assert "Keine" in panel._summary_label.text()

    def test_set_result_second_call_clears_old_rows(self, panel):
        panel.set_result(_result(n=5))
        panel.set_result(_result(n=2))
        assert panel._table.rowCount() == 2

    def test_row_contains_star_id(self, panel):
        star = _star(star_id=42)
        result = PhotometryResult(stars=[star], r_squared=0.9, n_matched=1)
        panel.set_result(result)
        item = panel._table.item(0, 0)
        assert item is not None
        assert item.data(2) == 42  # Qt.ItemDataRole.DisplayRole = 2


class TestClear:
    def test_clear_resets_table(self, panel):
        panel.set_result(_result(n=3))
        panel.clear()
        assert panel._table.rowCount() == 0

    def test_clear_disables_buttons(self, panel):
        panel.set_result(_result(n=3))
        panel.clear()
        assert not panel._csv_btn.isEnabled()
        assert not panel._fits_btn.isEnabled()


class TestExport:
    def test_export_csv_no_result_no_crash(self, panel):
        panel._result = None
        panel._export_csv()

    def test_export_fits_no_result_no_crash(self, panel):
        panel._result = None
        panel._export_fits()

    def test_export_csv_cancelled_no_crash(self, panel, monkeypatch):
        panel.set_result(_result(n=1))
        monkeypatch.setattr(
            "astroai.ui.widgets.photometry_panel.QFileDialog.getSaveFileName",
            lambda *a, **kw: ("", ""),
        )
        panel._export_csv()

    def test_export_fits_cancelled_no_crash(self, panel, monkeypatch):
        panel.set_result(_result(n=1))
        monkeypatch.setattr(
            "astroai.ui.widgets.photometry_panel.QFileDialog.getSaveFileName",
            lambda *a, **kw: ("", ""),
        )
        panel._export_fits()

    def test_export_csv_with_path_calls_exporter(self, panel, monkeypatch):
        panel.set_result(_result(n=1))
        calls = []

        class FakeExporter:
            def to_csv(self, result, path):
                calls.append(("csv", path))

        monkeypatch.setattr(
            "astroai.ui.widgets.photometry_panel.QFileDialog.getSaveFileName",
            lambda *a, **kw: ("/tmp/out.csv", ""),
        )
        import astroai.engine.photometry.export as exp_mod
        monkeypatch.setattr(exp_mod, "PhotometryExporter", FakeExporter)
        panel._export_csv()
        assert len(calls) == 1

    def test_export_fits_with_path_calls_exporter(self, panel, monkeypatch):
        panel.set_result(_result(n=1))
        calls = []

        class FakeExporter:
            def to_fits(self, result, path):
                calls.append(("fits", path))

        monkeypatch.setattr(
            "astroai.ui.widgets.photometry_panel.QFileDialog.getSaveFileName",
            lambda *a, **kw: ("/tmp/out.fits", ""),
        )
        import astroai.engine.photometry.export as exp_mod
        monkeypatch.setattr(exp_mod, "PhotometryExporter", FakeExporter)
        panel._export_fits()
        assert len(calls) == 1
