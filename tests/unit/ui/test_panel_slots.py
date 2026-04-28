"""Targeted tests to close coverage gaps in UI panel slot/handler methods.

Panels covered:
- photometry_panel  (lines 128, 133)
- drizzle_panel     (lines 136-137, 141-143, 147, 151-153)
- color_calibration_panel (lines 104-105, 109-111, 115)
- deconvolution_panel (lines 93-94, 98, 102)
- log_widget        (lines 133-140)
- mosaic_panel      (lines 169-176, 180-182)
- starless_panel    (lines 115-116, 120-121, 125-127, 131)
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from astroai.ui.models import PipelineModel


# ---------------------------------------------------------------------------
# photometry_panel — lines 128, 133
# ---------------------------------------------------------------------------

class TestPhotometryPanelExportFitsGaps:
    """Cover the two early-return branches in _export_fits."""

    @pytest.fixture()
    def panel(self, qtbot):
        from astroai.ui.widgets.photometry_panel import PhotometryPanel
        w = PhotometryPanel()
        qtbot.addWidget(w)
        return w

    @patch("astroai.ui.widgets.photometry_panel.QFileDialog.getSaveFileName")
    def test_export_fits_no_result_returns_early(self, mock_dialog, panel) -> None:
        """Line 128: _export_fits returns immediately when _result is None."""
        panel._result = None
        panel._export_fits()
        mock_dialog.assert_not_called()

    @patch("astroai.ui.widgets.photometry_panel.QFileDialog.getSaveFileName")
    def test_export_fits_cancelled_returns_early(self, mock_dialog, panel) -> None:
        """Line 133: _export_fits returns when dialog returns empty string."""
        from astroai.engine.photometry.models import PhotometryResult, StarMeasurement
        stars = [
            StarMeasurement(
                star_id=0, ra=10.0, dec=20.0, x_pixel=100.0, y_pixel=200.0,
                instr_mag=-5.0, catalog_mag=-4.8, cal_mag=-4.9, residual=0.1,
            )
        ]
        result = PhotometryResult(stars=stars, r_squared=0.99, n_matched=1)
        panel.set_result(result)
        mock_dialog.return_value = ("", "")
        panel._export_fits()
        mock_dialog.assert_called_once()


# ---------------------------------------------------------------------------
# drizzle_panel — lines 136-137, 141-143, 147, 151-153
# ---------------------------------------------------------------------------

class TestDrizzlePanelSlots:
    """Cover all four slot handler methods in DrizzlePanel."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, model: PipelineModel, qtbot):
        from astroai.ui.widgets.drizzle_panel import DrizzlePanel
        w = DrizzlePanel(model)
        qtbot.addWidget(w)
        return w

    def test_on_enabled_changed_true_updates_model(self, panel, model) -> None:
        """Lines 136-137: enabling sets model and enables settings group."""
        panel._enabled_cb.setChecked(True)
        assert model.drizzle_enabled is True
        assert panel._settings_group.isEnabled()

    def test_on_enabled_changed_false_updates_model(self, panel, model) -> None:
        """Lines 136-137: disabling sets model and disables settings group."""
        model._drizzle_enabled = True
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert model.drizzle_enabled is False
        assert not panel._settings_group.isEnabled()

    def test_on_drop_size_changed_via_radio_button(self, panel, model) -> None:
        """Lines 141-143: clicking a radio button updates model drop size."""
        # Select the 0.5 button (index 0)
        panel._drop_buttons[0].setChecked(True)
        panel._on_drop_size_changed()
        assert model.drizzle_drop_size == pytest.approx(0.5)

    def test_on_drop_size_changed_1_0(self, panel, model) -> None:
        """Lines 141-143: selecting 1.0 drop size."""
        panel._drop_buttons[2].setChecked(True)
        panel._on_drop_size_changed()
        assert model.drizzle_drop_size == pytest.approx(1.0)

    def test_on_drop_size_changed_no_checked_button_is_noop(self, panel, model) -> None:
        """Lines 141-143: no checked button means model stays unchanged."""
        original = model.drizzle_drop_size
        with patch.object(panel._drop_group, "checkedButton", return_value=None):
            panel._on_drop_size_changed()
        assert model.drizzle_drop_size == original

    def test_on_scale_changed_updates_model(self, panel, model) -> None:
        """Line 147: changing scale spinbox updates model."""
        panel._scale_spin.setValue(2.0)
        assert model.drizzle_scale == pytest.approx(2.0)

    def test_on_pixfrac_changed_updates_model_and_label(self, panel, model) -> None:
        """Lines 151-153: slider change updates model and label text."""
        panel._pixfrac_slider.setValue(75)
        assert model.drizzle_pixfrac == pytest.approx(0.75)
        assert panel._pixfrac_value.text() == "0.8"  # nearest .1 of 0.75

    def test_on_pixfrac_changed_100_shows_1_0(self, panel, model) -> None:
        """Lines 151-153: slider at 100 → label '1.0'."""
        # Move away first so valueChanged fires
        panel._pixfrac_slider.setValue(50)
        panel._pixfrac_slider.setValue(100)
        assert model.drizzle_pixfrac == pytest.approx(1.0)
        assert panel._pixfrac_value.text() == "1.0"

    def test_on_pixfrac_changed_10_shows_0_1(self, panel, model) -> None:
        """Lines 151-153: slider at 10 → label '0.1'."""
        panel._pixfrac_slider.setValue(10)
        assert model.drizzle_pixfrac == pytest.approx(0.1)
        assert panel._pixfrac_value.text() == "0.1"

    def test_signal_emitted_on_enabled(self, panel, model, qtbot) -> None:
        """Setting enabled emits drizzle_config_changed."""
        with qtbot.waitSignal(model.drizzle_config_changed, timeout=500):
            panel._enabled_cb.setChecked(True)


# ---------------------------------------------------------------------------
# color_calibration_panel — lines 104-105, 109-111, 115
# ---------------------------------------------------------------------------

class TestColorCalibrationPanelSlots:
    """Cover the three slot handlers in ColorCalibrationPanel."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, model: PipelineModel, qtbot):
        from astroai.ui.widgets.color_calibration_panel import ColorCalibrationPanel
        w = ColorCalibrationPanel(model)
        qtbot.addWidget(w)
        return w

    def test_on_enabled_changed_true_updates_model_and_group(self, panel, model) -> None:
        """Lines 104-105: toggling on sets model and enables settings group."""
        panel._enabled_cb.setChecked(True)
        assert model.color_calibration_enabled is True
        assert panel._settings_group.isEnabled()

    def test_on_enabled_changed_false_disables_group(self, panel, model) -> None:
        """Lines 104-105: toggling off disables settings group."""
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert model.color_calibration_enabled is False
        assert not panel._settings_group.isEnabled()

    def test_on_catalog_changed_to_2mass(self, panel, model) -> None:
        """Lines 109-111: selecting 2MASS updates model catalog."""
        panel._catalog_combo.setCurrentIndex(1)  # 2MASS
        assert model.color_calibration_catalog == "2mass"

    def test_on_catalog_changed_to_gaia_dr3(self, panel, model) -> None:
        """Lines 109-111: selecting GAIA DR3 updates model catalog."""
        # switch to 2mass first, then back
        panel._catalog_combo.setCurrentIndex(1)
        panel._catalog_combo.setCurrentIndex(0)  # GAIA DR3
        assert model.color_calibration_catalog == "gaia_dr3"

    def test_on_radius_changed_updates_model(self, panel, model) -> None:
        """Line 115: changing radius spinbox updates model."""
        panel._radius_spin.setValue(12)
        assert model.color_calibration_sample_radius == 12

    def test_signal_emitted_on_enabled(self, panel, model, qtbot) -> None:
        """Enabling emits color_calibration_config_changed."""
        with qtbot.waitSignal(model.color_calibration_config_changed, timeout=500):
            panel._enabled_cb.setChecked(True)

    def test_on_catalog_changed_with_invalid_data_is_noop(self, panel, model) -> None:
        """Lines 109-111: if itemData returns falsy, model is unchanged."""
        original = model.color_calibration_catalog
        with patch.object(panel._catalog_combo, "itemData", return_value=None):
            panel._on_catalog_changed(0)
        assert model.color_calibration_catalog == original


# ---------------------------------------------------------------------------
# deconvolution_panel — lines 93-94, 98, 102
# ---------------------------------------------------------------------------

class TestDeconvolutionPanelSlots:
    """Cover the three slot handlers in DeconvolutionPanel."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, model: PipelineModel, qtbot):
        from astroai.ui.widgets.deconvolution_panel import DeconvolutionPanel
        w = DeconvolutionPanel(model)
        qtbot.addWidget(w)
        return w

    def test_on_enabled_changed_true(self, panel, model) -> None:
        """Lines 93-94: enabling sets model and enables settings group."""
        panel._enabled_cb.setChecked(True)
        assert model.deconvolution_enabled is True
        assert panel._settings_group.isEnabled()

    def test_on_enabled_changed_false(self, panel, model) -> None:
        """Lines 93-94: disabling sets model and disables settings group."""
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert model.deconvolution_enabled is False
        assert not panel._settings_group.isEnabled()

    def test_on_iterations_changed_updates_model(self, panel, model) -> None:
        """Line 98: changing iterations spinbox updates model."""
        panel._iter_spin.setValue(25)
        assert model.deconvolution_iterations == 25

    def test_on_sigma_changed_updates_model(self, panel, model) -> None:
        """Line 102: changing sigma spinbox updates model."""
        panel._sigma_spin.setValue(3.5)
        assert model.deconvolution_psf_sigma == pytest.approx(3.5)

    def test_signal_emitted_on_enabled(self, panel, model, qtbot) -> None:
        """Enabling emits deconvolution_config_changed."""
        with qtbot.waitSignal(model.deconvolution_config_changed, timeout=500):
            panel._enabled_cb.setChecked(True)

    def test_pipeline_reset_syncs_panel(self, panel, model) -> None:
        """_sync_from_model responds to pipeline_reset signal."""
        model._deconvolution_enabled = True
        model._deconvolution_iterations = 42
        model._deconvolution_psf_sigma = 2.5
        model.pipeline_reset.emit()
        assert panel._enabled_cb.isChecked()
        assert panel._iter_spin.value() == 42
        assert panel._sigma_spin.value() == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# log_widget — lines 133-140
# ---------------------------------------------------------------------------

class TestLogWidgetExportWithContent:
    """Cover _on_export when text is present and dialog returns a path."""

    @pytest.fixture()
    def widget(self, qtbot):
        from astroai.ui.widgets.log_widget import LogWidget
        w = LogWidget()
        qtbot.addWidget(w)
        return w

    def test_export_writes_file_when_content_present(self, widget, tmp_path) -> None:
        """Lines 133-140: export writes log text to the chosen file."""
        widget.append_message("Exporttest-Nachricht", logging.INFO)
        out_file = str(tmp_path / "log_export.txt")
        with patch(
            "astroai.ui.widgets.log_widget.QFileDialog.getSaveFileName",
            return_value=(out_file, ""),
        ):
            widget._on_export()
        assert (tmp_path / "log_export.txt").exists()
        content = (tmp_path / "log_export.txt").read_text(encoding="utf-8")
        assert "Exporttest-Nachricht" in content

    def test_export_empty_content_no_dialog(self, widget) -> None:
        """Lines 129-132: if text is blank, dialog is never opened."""
        with patch(
            "astroai.ui.widgets.log_widget.QFileDialog.getSaveFileName"
        ) as mock_dialog:
            widget._on_export()
        mock_dialog.assert_not_called()

    def test_export_cancelled_does_not_crash(self, widget) -> None:
        """Lines 138-140 branch: dialog cancelled → no file written, no crash."""
        widget.append_message("Einige Daten", logging.WARNING)
        with patch(
            "astroai.ui.widgets.log_widget.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ):
            widget._on_export()  # must not raise

    def test_export_default_name_contains_timestamp(self, widget, tmp_path) -> None:
        """Line 134: default_name contains 'astroai_log_' prefix."""
        widget.append_message("Zeitstempel-Test", logging.DEBUG)
        captured_args: list = []

        def capture_dialog(parent, title, default, ffilter):
            captured_args.extend([default])
            return ("", "")

        with patch(
            "astroai.ui.widgets.log_widget.QFileDialog.getSaveFileName",
            side_effect=capture_dialog,
        ):
            widget._on_export()
        assert captured_args
        assert captured_args[0].startswith("astroai_log_")
        assert captured_args[0].endswith(".txt")


# ---------------------------------------------------------------------------
# mosaic_panel — lines 169-176, 180-182
# ---------------------------------------------------------------------------

class TestMosaicPanelDialogSlots:
    """Cover _on_add_panels (file dialog) and _on_remove_panel."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, model: PipelineModel, qtbot):
        from astroai.ui.widgets.mosaic_panel import MosaicPanel
        w = MosaicPanel(model)
        qtbot.addWidget(w)
        return w

    @patch("astroai.ui.widgets.mosaic_panel.QFileDialog.getOpenFileNames")
    def test_on_add_panels_adds_to_model(self, mock_dialog, panel, model) -> None:
        """Lines 169-176: add dialog result paths are forwarded to model."""
        mock_dialog.return_value = (["/a.fits", "/b.fits"], "")
        panel._on_add_panels()
        assert model.mosaic_panels == ["/a.fits", "/b.fits"]

    @patch("astroai.ui.widgets.mosaic_panel.QFileDialog.getOpenFileNames")
    def test_on_add_panels_cancelled_adds_nothing(self, mock_dialog, panel, model) -> None:
        """Lines 169-176: cancelled dialog adds no panels."""
        mock_dialog.return_value = ([], "")
        panel._on_add_panels()
        assert model.mosaic_panels == []

    @patch("astroai.ui.widgets.mosaic_panel.QFileDialog.getOpenFileNames")
    def test_on_add_panels_updates_list_widget(self, mock_dialog, panel) -> None:
        """Lines 169-176: list widget shows added files."""
        mock_dialog.return_value = (["/img1.fits"], "")
        panel._on_add_panels()
        assert panel._panel_list.count() == 1
        assert panel._panel_list.item(0).text() == "/img1.fits"

    def test_on_remove_panel_removes_selected(self, panel, model) -> None:
        """Lines 180-182: clicking remove with a selection removes from model."""
        model.mosaic_panels = ["/keep.fits", "/remove.fits"]
        # select the second item
        panel._panel_list.setCurrentRow(1)
        panel._on_remove_panel()
        assert model.mosaic_panels == ["/keep.fits"]

    def test_on_remove_panel_no_selection_is_noop(self, panel, model) -> None:
        """Lines 180-182: no selected item means model is unchanged."""
        model.mosaic_panels = ["/a.fits"]
        panel._panel_list.clearSelection()
        panel._panel_list.setCurrentItem(None)
        panel._on_remove_panel()
        assert model.mosaic_panels == ["/a.fits"]


# ---------------------------------------------------------------------------
# starless_panel — lines 115-116, 120-121, 125-127, 131
# ---------------------------------------------------------------------------

class TestStarlessPanelSlots:
    """Cover all four slot handlers in StarlessPanel."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, model: PipelineModel, qtbot):
        from astroai.ui.widgets.starless_panel import StarlessPanel
        w = StarlessPanel(model)
        qtbot.addWidget(w)
        return w

    def test_on_enabled_changed_true(self, panel, model) -> None:
        """Lines 115-116: enabling sets model and enables settings group."""
        panel._enabled_cb.setChecked(True)
        assert model.starless_enabled is True
        assert panel._settings_group.isEnabled()

    def test_on_enabled_changed_false(self, panel, model) -> None:
        """Lines 115-116: disabling sets model and disables settings group."""
        panel._enabled_cb.setChecked(True)
        panel._enabled_cb.setChecked(False)
        assert model.starless_enabled is False
        assert not panel._settings_group.isEnabled()

    def test_on_strength_changed_updates_model(self, panel, model) -> None:
        """Lines 120-121: slider change updates model strength."""
        panel._strength_slider.setValue(60)
        assert model.starless_strength == pytest.approx(0.60)

    def test_on_strength_changed_updates_label(self, panel) -> None:
        """Lines 120-121: slider change updates percentage label."""
        panel._strength_slider.setValue(45)
        assert panel._strength_value.text() == "45%"

    def test_on_strength_changed_zero(self, panel, model) -> None:
        """Lines 120-121: slider at 0 → strength 0.0, label '0%'."""
        panel._strength_slider.setValue(0)
        assert model.starless_strength == pytest.approx(0.0)
        assert panel._strength_value.text() == "0%"

    def test_on_format_changed_tiff(self, panel, model) -> None:
        """Lines 125-127: selecting TIFF format updates model."""
        # _FORMAT_OPTIONS = [("xisf", ...), ("tiff", ...), ("fits", ...)]
        panel._format_combo.setCurrentIndex(1)  # tiff
        assert model.starless_format == "tiff"

    def test_on_format_changed_fits(self, panel, model) -> None:
        """Lines 125-127: selecting FITS format updates model."""
        panel._format_combo.setCurrentIndex(2)  # fits
        assert model.starless_format == "fits"

    def test_on_format_changed_with_no_data_is_noop(self, panel, model) -> None:
        """Lines 125-127: if currentData() returns falsy, model is unchanged."""
        original = model.starless_format
        with patch.object(panel._format_combo, "currentData", return_value=None):
            panel._on_format_changed(0)
        assert model.starless_format == original

    def test_on_mask_changed_false(self, panel, model) -> None:
        """Line 131: unchecking mask updates model save_star_mask."""
        panel._mask_cb.setChecked(False)
        assert model.save_star_mask is False

    def test_on_mask_changed_true(self, panel, model) -> None:
        """Line 131: checking mask updates model save_star_mask."""
        panel._mask_cb.setChecked(False)
        panel._mask_cb.setChecked(True)
        assert model.save_star_mask is True

    def test_signal_emitted_on_enabled(self, panel, model, qtbot) -> None:
        """Enabling emits starless_config_changed."""
        with qtbot.waitSignal(model.starless_config_changed, timeout=500):
            panel._enabled_cb.setChecked(True)

    def test_pipeline_reset_syncs_panel(self, panel, model) -> None:
        """_sync_from_model responds to pipeline_reset signal."""
        model._starless_enabled = True
        model._starless_strength = 0.75
        model._starless_format = "fits"
        model._save_star_mask = False
        model.pipeline_reset.emit()
        assert panel._enabled_cb.isChecked()
        assert panel._strength_slider.value() == 75
        assert panel._strength_value.text() == "75%"
        assert panel._mask_cb.isChecked() is False


# ---------------------------------------------------------------------------
# annotation_panel — lines 83-95 (set_wcs_active)
# ---------------------------------------------------------------------------

class TestAnnotationPanelSetWcsActive:
    """Cover both branches of set_wcs_active (lines 83-95)."""

    @pytest.fixture()
    def panel(self, qtbot):
        from astroai.ui.widgets.annotation_panel import AnnotationPanel
        w = AnnotationPanel()
        qtbot.addWidget(w)
        return w

    def test_set_wcs_active_true_sets_label_and_enables_checkboxes(self, panel) -> None:
        """Lines 83-95: active=True sets 'aktiv' text and enables all checkboxes."""
        panel.set_wcs_active(True)
        assert "aktiv" in panel._status_label.text().lower()
        assert panel._status_label.objectName() == "annotationStatusActive"
        assert panel._dso_cb.isEnabled()
        assert panel._stars_cb.isEnabled()
        assert panel._boundaries_cb.isEnabled()
        assert panel._grid_cb.isEnabled()

    def test_set_wcs_active_false_sets_label_and_disables_checkboxes(self, panel) -> None:
        """Lines 83-95: active=False resets label text and disables all checkboxes."""
        # Activate first so we actually change state
        panel.set_wcs_active(True)
        panel.set_wcs_active(False)
        assert "Kein Plate Solve aktiv" in panel._status_label.text()
        assert panel._status_label.objectName() == "annotationStatusLabel"
        assert not panel._dso_cb.isEnabled()
        assert not panel._stars_cb.isEnabled()
        assert not panel._boundaries_cb.isEnabled()
        assert not panel._grid_cb.isEnabled()

    def test_set_wcs_active_roundtrip(self, panel) -> None:
        """Toggle active → inactive → active stays consistent."""
        panel.set_wcs_active(False)
        assert not panel._dso_cb.isEnabled()
        panel.set_wcs_active(True)
        assert panel._dso_cb.isEnabled()


# ---------------------------------------------------------------------------
# activation_dialog — lines 108-112 (_on_activate) and 116-122 (_on_deactivate)
# ---------------------------------------------------------------------------

class TestActivationDialogSlots:
    """Cover _on_activate and _on_deactivate slots (lines 108-122)."""

    @pytest.fixture()
    def adapter(self):
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.is_activated = False
        # Signal objects must support .connect()
        for sig_name in ("activation_started", "activation_succeeded", "activation_failed"):
            sig = MagicMock()
            sig.connect = MagicMock()
            setattr(mock, sig_name, sig)
        return mock

    @pytest.fixture()
    def dialog(self, qtbot, adapter):
        from astroai.ui.widgets.activation_dialog import ActivationDialog
        dlg = ActivationDialog(adapter)
        qtbot.addWidget(dlg)
        return dlg

    def test_on_activate_with_key_emits_signal_and_calls_adapter(
        self, dialog, adapter, qtbot
    ) -> None:
        """Lines 108-112: with non-empty key, emits activation_requested and calls adapter."""
        dialog._key_input.setText("ASTRO-1234-5678-ABCD")
        with qtbot.waitSignal(dialog.activation_requested, timeout=500) as blocker:
            dialog._on_activate()
        assert blocker.args == ["ASTRO-1234-5678-ABCD"]
        adapter.activate_async.assert_called_once_with("ASTRO-1234-5678-ABCD")

    def test_on_activate_with_empty_key_does_nothing(self, dialog, adapter) -> None:
        """Lines 108-110: empty key causes early return — adapter never called."""
        dialog._key_input.setText("   ")
        dialog._on_activate()
        adapter.activate_async.assert_not_called()

    def test_on_deactivate_calls_adapter_and_resets_ui(self, dialog, adapter) -> None:
        """Lines 116-122: deactivate calls adapter and resets all UI elements."""
        # Put dialog in an 'activated' visual state first
        dialog._deactivate_btn.setVisible(True)
        dialog._success_label.show()
        dialog._error_label.show()
        dialog._key_input.setText("ASTRO-OLD-KEY-XXXX")
        dialog._key_input.setEnabled(False)
        dialog._activate_btn.setEnabled(True)

        dialog._on_deactivate()

        adapter.deactivate.assert_called_once()
        assert not dialog._deactivate_btn.isVisible()
        assert dialog._success_label.isHidden()
        assert dialog._error_label.isHidden()
        assert dialog._key_input.text() == ""
        assert dialog._key_input.isEnabled()
        assert not dialog._activate_btn.isEnabled()

    def test_on_activate_strips_whitespace(self, dialog, adapter, qtbot) -> None:
        """Lines 108-112: leading/trailing whitespace is stripped from key."""
        dialog._key_input.setText("  ASTRO-TRIM-TRIM-TEST  ")
        with qtbot.waitSignal(dialog.activation_requested, timeout=500) as blocker:
            dialog._on_activate()
        assert blocker.args == ["ASTRO-TRIM-TRIM-TEST"]
        adapter.activate_async.assert_called_once_with("ASTRO-TRIM-TRIM-TEST")


# ---------------------------------------------------------------------------
# channel_panel — lines 121-170 (_on_combine early returns) and 176-181 (_on_browse)
# ---------------------------------------------------------------------------

class TestChannelCombinerPanelSlots:
    """Cover early-return branches in _on_combine and _on_browse (lines 119-181)."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    @pytest.fixture()
    def panel(self, model: PipelineModel, qtbot):
        from astroai.ui.widgets.channel_panel import ChannelCombinerPanel
        w = ChannelCombinerPanel(model)
        qtbot.addWidget(w)
        return w

    def test_on_combine_no_channels_sets_status(self, panel) -> None:
        """Lines 146-148: when no channel paths are filled, status label shows message."""
        # Ensure all edits are empty
        for edit in panel._channel_edits.values():
            edit.clear()
        panel._on_combine()
        assert panel._status_label.text() == "Keine Kanäle geladen."

    def test_on_combine_read_fits_error_sets_status_and_returns(self, panel) -> None:
        """Lines 136-144: read error sets status label and returns without combining."""
        panel._rb_lrgb.setChecked(True)
        panel._channel_edits["L"].setText("/fake/luminance.fits")

        with patch(
            "astroai.core.io.read_fits",
            side_effect=OSError("Datei nicht gefunden"),
        ):
            panel._on_combine()

        assert "Lesefehler L:" in panel._status_label.text()
        assert "Datei nicht gefunden" in panel._status_label.text()

    def test_on_combine_read_tiff_error_sets_status_and_returns(self, panel) -> None:
        """Lines 136-144: TIFF read error sets status and returns."""
        panel._rb_lrgb.setChecked(True)
        panel._channel_edits["R"].setText("/fake/red.tiff")

        with patch(
            "astroai.core.io.read_tiff",
            side_effect=ValueError("Ungültiges TIFF"),
        ):
            panel._on_combine()

        assert "Lesefehler R:" in panel._status_label.text()
        assert "Ungültiges TIFF" in panel._status_label.text()

    @patch("astroai.ui.widgets.channel_panel.QFileDialog.getOpenFileName")
    def test_on_browse_sets_edit_text_when_file_chosen(self, mock_dialog, panel) -> None:
        """Lines 176-181: _pick_file sets line edit text when dialog returns a path."""
        mock_dialog.return_value = ("/chosen/path/image.fits", "")
        # Directly invoke _pick_file with the L channel edit
        edit = panel._channel_edits["L"]
        panel._pick_file(edit, "L")
        assert edit.text() == "/chosen/path/image.fits"

    @patch("astroai.ui.widgets.channel_panel.QFileDialog.getOpenFileName")
    def test_on_browse_cancelled_leaves_edit_empty(self, mock_dialog, panel) -> None:
        """Lines 176-181: cancelled dialog leaves line edit unchanged."""
        mock_dialog.return_value = ("", "")
        edit = panel._channel_edits["L"]
        edit.clear()
        panel._pick_file(edit, "L")
        assert edit.text() == ""

    # -----------------------------------------------------------------------
    # Success paths — lines 141, 150-168
    # -----------------------------------------------------------------------

    def test_on_combine_lrgb_success_sets_ok_status(self, panel) -> None:
        """Lines 151-157, 167: LRGB combine success sets status label to 'OK …'."""
        import numpy as np

        arr = np.ones((8, 8), dtype=np.float32)
        panel._rb_lrgb.setChecked(True)
        panel._channel_edits["L"].setText("/fake/lum.fits")

        with patch("astroai.core.io.read_fits", return_value=(arr, None)):
            panel._on_combine()

        assert panel._status_label.text().startswith("OK")
        assert "Shape" in panel._status_label.text()

    def test_on_combine_lrgb_success_calls_model_result_ready(self, panel) -> None:
        """Line 168: model_result_ready is called with the combined ndarray."""
        import numpy as np

        arr = np.ones((8, 8), dtype=np.float32)
        panel._rb_lrgb.setChecked(True)
        panel._channel_edits["R"].setText("/fake/red.fits")

        received: list = []
        panel.model_result_ready = lambda r: received.append(r)

        with patch("astroai.core.io.read_fits", return_value=(arr, None)):
            panel._on_combine()

        assert len(received) == 1
        assert isinstance(received[0], np.ndarray)

    def test_on_combine_narrowband_success_sets_ok_status(self, panel) -> None:
        """Lines 158-167: narrowband combine success sets status label to 'OK …'."""
        import numpy as np

        arr = np.ones((8, 8), dtype=np.float32)
        panel._rb_nb.setChecked(True)
        panel._channel_edits["Ha"].setText("/fake/ha.fits")

        with patch("astroai.core.io.read_fits", return_value=(arr, None)):
            panel._on_combine()

        assert panel._status_label.text().startswith("OK")

    def test_on_combine_narrowband_success_calls_model_result_ready(self, panel) -> None:
        """Line 168: model_result_ready called with narrowband result."""
        import numpy as np

        arr = np.ones((8, 8), dtype=np.float32)
        panel._rb_nb.setChecked(True)
        # Select HOO palette so the combo is valid
        panel._palette_combo.setCurrentIndex(1)  # "HOO"
        panel._channel_edits["Ha"].setText("/fake/ha.fits")
        panel._channel_edits["OIII"].setText("/fake/oiii.fits")

        received: list = []
        panel.model_result_ready = lambda r: received.append(r)

        with patch("astroai.core.io.read_fits", return_value=(arr, None)):
            panel._on_combine()

        assert len(received) == 1

    def test_on_combine_combine_exception_sets_fehler_status(self, panel) -> None:
        """Lines 169-170: exception inside combine sets status to 'Fehler: …'."""
        import numpy as np

        arr = np.ones((8, 8), dtype=np.float32)
        panel._rb_lrgb.setChecked(True)
        panel._channel_edits["L"].setText("/fake/lum.fits")

        with patch("astroai.core.io.read_fits", return_value=(arr, None)):
            with patch(
                "astroai.processing.channels.ChannelCombiner.combine_lrgb",
                side_effect=RuntimeError("combine fehler"),
            ):
                panel._on_combine()

        assert panel._status_label.text().startswith("Fehler:")
        assert "combine fehler" in panel._status_label.text()

    def test_on_combine_line141_successful_read_stores_channel(self, panel) -> None:
        """Line 141: channels[key] = data is executed when read_fits succeeds."""
        import numpy as np

        arr = np.ones((8, 8), dtype=np.float32) * 0.5
        panel._rb_lrgb.setChecked(True)
        # Provide only the G channel so we can verify it was stored and used
        panel._channel_edits["G"].setText("/fake/green.fits")

        received: list = []
        panel.model_result_ready = lambda r: received.append(r)

        with patch("astroai.core.io.read_fits", return_value=(arr, None)):
            panel._on_combine()

        # If line 141 ran correctly the combine should succeed and
        # model_result_ready is called with the result
        assert len(received) == 1
