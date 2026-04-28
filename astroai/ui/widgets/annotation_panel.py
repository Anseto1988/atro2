"""Annotation toggle panel for controlling overlay visibility."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

__all__ = ["AnnotationPanel"]


class AnnotationPanel(QWidget):
    """Dock-panel with toggles for each annotation layer."""

    dso_toggled = Signal(bool)
    stars_toggled = Signal(bool)
    boundaries_toggled = Signal(bool)
    grid_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Annotations-Steuerung")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Annotationen")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        self._status_label = QLabel("Kein Plate Solve aktiv")
        self._status_label.setObjectName("annotationStatusLabel")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        layout.addSpacing(4)

        self._dso_cb = self._make_checkbox(
            "Deep-Sky-Objekte (Messier/NGC)",
            "Messier- und NGC-Objekte anzeigen",
            checked=True,
        )
        self._dso_cb.toggled.connect(self.dso_toggled)
        layout.addWidget(self._dso_cb)

        self._stars_cb = self._make_checkbox(
            "Benannte Sterne",
            "Helle Sterne mit Namen anzeigen",
            checked=True,
        )
        self._stars_cb.toggled.connect(self.stars_toggled)
        layout.addWidget(self._stars_cb)

        self._boundaries_cb = self._make_checkbox(
            "Konstellationsgrenzen",
            "IAU-Konstellationsgrenzen anzeigen",
            checked=False,
        )
        self._boundaries_cb.toggled.connect(self.boundaries_toggled)
        layout.addWidget(self._boundaries_cb)

        self._grid_cb = self._make_checkbox(
            "RA/Dec-Gitter",
            "Koordinatengitter einblenden",
            checked=False,
        )
        self._grid_cb.toggled.connect(self.grid_toggled)
        layout.addWidget(self._grid_cb)

        layout.addStretch()

    def set_wcs_active(self, active: bool) -> None:
        if active:
            self._status_label.setText("Plate Solve aktiv — Annotationen verfuegbar")
            self._status_label.setObjectName("annotationStatusActive")
        else:
            self._status_label.setText("Kein Plate Solve aktiv")
            self._status_label.setObjectName("annotationStatusLabel")
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)

        self._dso_cb.setEnabled(active)
        self._stars_cb.setEnabled(active)
        self._boundaries_cb.setEnabled(active)
        self._grid_cb.setEnabled(active)

    @staticmethod
    def _make_checkbox(text: str, tooltip: str, *, checked: bool) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(checked)
        cb.setToolTip(tooltip)
        cb.setAccessibleName(text)
        return cb
