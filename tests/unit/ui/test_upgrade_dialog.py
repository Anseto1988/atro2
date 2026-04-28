"""Tests for UpgradeDialog widget."""
from __future__ import annotations

import pytest

from astroai.licensing.models import LicenseTier
from astroai.ui.widgets.upgrade_dialog import UpgradeDialog, _TIER_LABELS


@pytest.fixture()
def dialog_free(qtbot):
    dlg = UpgradeDialog("AI-Sternausrichtung", LicenseTier.PRO_MONTHLY, LicenseTier.FREE)
    qtbot.addWidget(dlg)
    return dlg


@pytest.fixture()
def dialog_pro(qtbot):
    dlg = UpgradeDialog("Mosaic", LicenseTier.FOUNDING_MEMBER, LicenseTier.PRO_MONTHLY)
    qtbot.addWidget(dlg)
    return dlg


class TestInitialState:
    def test_creates_without_error(self, dialog_free):
        assert dialog_free is not None

    def test_window_title(self, dialog_free):
        assert dialog_free.windowTitle() == "Upgrade erforderlich"

    def test_minimum_width(self, dialog_free):
        assert dialog_free.minimumWidth() == 380

    def test_is_modal(self, dialog_free):
        assert dialog_free.isModal()


class TestActivateButtonVisibility:
    def test_activate_button_shown_for_free_tier(self, dialog_free):
        btn_texts = [
            dialog_free.layout().itemAt(i).widget().text()
            for i in range(dialog_free.layout().count())
            if dialog_free.layout().itemAt(i).widget()
        ]
        # Free tier shows "Lizenz eingeben" button — find it in children
        buttons = dialog_free.findChildren(
            __import__("PySide6.QtWidgets", fromlist=["QPushButton"]).QPushButton
        )
        btn_labels = [b.text() for b in buttons]
        assert "Lizenz eingeben" in btn_labels

    def test_activate_button_absent_for_non_free_tier(self, dialog_pro):
        from PySide6.QtWidgets import QPushButton
        buttons = dialog_pro.findChildren(QPushButton)
        btn_labels = [b.text() for b in buttons]
        assert "Lizenz eingeben" not in btn_labels

    def test_upgrade_button_always_shown(self, dialog_free):
        from PySide6.QtWidgets import QPushButton
        buttons = dialog_free.findChildren(QPushButton)
        btn_labels = [b.text() for b in buttons]
        assert "Upgrade-Info" in btn_labels

    def test_close_button_always_shown(self, dialog_free):
        from PySide6.QtWidgets import QPushButton
        buttons = dialog_free.findChildren(QPushButton)
        btn_labels = [b.text() for b in buttons]
        assert "Schließen" in btn_labels


class TestSignals:
    def test_upgrade_requested_signal_emitted(self, dialog_free, qtbot):
        with qtbot.waitSignal(dialog_free.upgrade_requested, timeout=500):
            dialog_free._on_upgrade()

    def test_activate_requested_signal_emitted(self, dialog_free, qtbot):
        with qtbot.waitSignal(dialog_free.activate_requested, timeout=500):
            dialog_free._on_activate()

    def test_on_activate_calls_accept(self, dialog_free, qtbot):
        results = []
        dialog_free.accepted.connect(lambda: results.append(True))
        dialog_free._on_activate()
        assert len(results) == 1


class TestTierLabels:
    def test_all_tiers_have_labels(self):
        for tier in LicenseTier:
            assert tier in _TIER_LABELS

    def test_free_tier_label(self):
        assert _TIER_LABELS[LicenseTier.FREE] == "Free"
