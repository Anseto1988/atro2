"""Tests for AnnotationPanel widget."""
from __future__ import annotations

import pytest

from astroai.ui.widgets.annotation_panel import AnnotationPanel


@pytest.fixture()
def panel(qtbot):
    w = AnnotationPanel()
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_creates_without_error(self, panel):
        assert panel is not None

    def test_dso_checked_initially(self, panel):
        assert panel._dso_cb.isChecked()

    def test_stars_checked_initially(self, panel):
        assert panel._stars_cb.isChecked()

    def test_boundaries_unchecked_initially(self, panel):
        assert not panel._boundaries_cb.isChecked()

    def test_grid_unchecked_initially(self, panel):
        assert not panel._grid_cb.isChecked()

    def test_status_label_default_text(self, panel):
        assert "Kein Plate Solve" in panel._status_label.text()

    def test_accessible_name(self, panel):
        assert panel.accessibleName() == "Annotations-Steuerung"


class TestSetWcsActive:
    def test_active_true_enables_checkboxes(self, panel):
        panel.set_wcs_active(True)
        assert panel._dso_cb.isEnabled()
        assert panel._stars_cb.isEnabled()
        assert panel._boundaries_cb.isEnabled()
        assert panel._grid_cb.isEnabled()

    def test_active_false_disables_checkboxes(self, panel):
        panel.set_wcs_active(True)
        panel.set_wcs_active(False)
        assert not panel._dso_cb.isEnabled()
        assert not panel._stars_cb.isEnabled()
        assert not panel._boundaries_cb.isEnabled()
        assert not panel._grid_cb.isEnabled()

    def test_active_true_updates_status_text(self, panel):
        panel.set_wcs_active(True)
        assert "aktiv" in panel._status_label.text().lower()

    def test_active_false_updates_status_text(self, panel):
        panel.set_wcs_active(True)
        panel.set_wcs_active(False)
        assert "Kein Plate Solve" in panel._status_label.text()

    def test_active_true_sets_object_name(self, panel):
        panel.set_wcs_active(True)
        assert panel._status_label.objectName() == "annotationStatusActive"

    def test_active_false_sets_object_name(self, panel):
        panel.set_wcs_active(True)
        panel.set_wcs_active(False)
        assert panel._status_label.objectName() == "annotationStatusLabel"


class TestSignals:
    def test_dso_toggle_emits_signal(self, panel, qtbot):
        panel._dso_cb.setChecked(True)
        with qtbot.waitSignal(panel.dso_toggled, timeout=500) as blocker:
            panel._dso_cb.setChecked(False)
        assert blocker.args == [False]

    def test_stars_toggle_emits_signal(self, panel, qtbot):
        with qtbot.waitSignal(panel.stars_toggled, timeout=500) as blocker:
            panel._stars_cb.setChecked(False)
        assert blocker.args == [False]

    def test_boundaries_toggle_emits_signal(self, panel, qtbot):
        with qtbot.waitSignal(panel.boundaries_toggled, timeout=500) as blocker:
            panel._boundaries_cb.setChecked(True)
        assert blocker.args == [True]

    def test_grid_toggle_emits_signal(self, panel, qtbot):
        with qtbot.waitSignal(panel.grid_toggled, timeout=500) as blocker:
            panel._grid_cb.setChecked(True)
        assert blocker.args == [True]
