"""AstroAI application entry point and MainWindow."""
from __future__ import annotations

import sys
from typing import cast
from pathlib import Path

from PySide6.QtCore import QFile, Qt, QTextStream, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from astroai import __version__
from astroai.project import AstroProject, ProjectSerializer
from astroai.project.project_file import PROJECT_EXTENSION
from astroai.project.recent_files import RecentProjects
from astroai.project.serializer import ProjectSerializerError
from astroai.ui.license_adapter import QLicenseAdapter
from astroai.ui.main.loader import FileLoader
from astroai.ui.models import PipelineModel
from astroai.ui.widgets.activation_dialog import ActivationDialog
from astroai.ui.widgets.histogram_widget import HistogramWidget
from astroai.ui.widgets.image_viewer import ImageViewer
from astroai.ui.widgets.license_badge import LicenseBadge
from astroai.ui.widgets.offline_banner import OfflineBanner
from astroai.ui.widgets.log_widget import LogWidget
from astroai.core.calibration.worker import CalibrationWorker
from astroai.ui.widgets.calibration_benchmark import CalibrationBenchmarkWidget
from astroai.ui.widgets.progress_widget import ProgressWidget
from astroai.ui.widgets.upgrade_dialog import UpgradeDialog
from astroai.ui.widgets.annotation_panel import AnnotationPanel
from astroai.ui.widgets.channel_panel import ChannelCombinerPanel
from astroai.ui.widgets.comet_stack_panel import CometStackPanel
from astroai.ui.widgets.deconvolution_panel import DeconvolutionPanel
from astroai.ui.widgets.drizzle_panel import DrizzlePanel
from astroai.ui.widgets.mosaic_panel import MosaicPanel
from astroai.ui.widgets.color_calibration_panel import ColorCalibrationPanel
from astroai.ui.widgets.starless_panel import StarlessPanel
from astroai.ui.widgets.workflow_graph import WorkflowGraph
from astroai.ui.overlay.annotation_overlay import AnnotationOverlay
from astroai.ui.widgets.sky_overlay import SkyOverlay
from astroai.licensing.models import LicenseTier

__all__ = ["MainWindow", "main"]

def _resources_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "astroai" / "ui" / "resources"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent / "resources"

_RESOURCES = _resources_dir()


def _load_stylesheet() -> str:
    qss_path = _RESOURCES / "dark_theme.qss"
    if not qss_path.exists():
        return ""
    f = QFile(str(qss_path))
    if not f.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
        return ""
    stream = QTextStream(f)
    text = stream.readAll()
    f.close()
    return text


class MainWindow(QMainWindow):
    """Primary application window with dock-based layout."""

    def __init__(self, license_adapter: QLicenseAdapter | None = None) -> None:
        super().__init__()
        self.setWindowTitle("AstroAI Suite")
        self.setMinimumSize(960, 640)

        self._license = license_adapter or QLicenseAdapter(self)
        self._pipeline = PipelineModel(self)
        self._calibration_worker = CalibrationWorker(self)
        self._file_loader = FileLoader(self)
        self._project = AstroProject()
        self._project_path: Path | None = None
        self._recent = RecentProjects()
        self._setup_central()
        self._setup_docks()
        self._setup_menus()
        self._setup_statusbar()
        self._connect_signals()
        self._license.verify()

    def _setup_central(self) -> None:
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self._offline_banner = OfflineBanner()
        central_layout.addWidget(self._offline_banner)

        viewer_container = QWidget()
        stack = QStackedLayout(viewer_container)
        stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        stack.setContentsMargins(0, 0, 0, 0)

        self._viewer = ImageViewer()
        self._sky_overlay = SkyOverlay(self._viewer)
        self._annotation_overlay = AnnotationOverlay(self._viewer)

        stack.addWidget(self._viewer)
        stack.addWidget(self._sky_overlay)
        stack.addWidget(self._annotation_overlay)

        central_layout.addWidget(viewer_container, stretch=1)
        self.setCentralWidget(central)

    def _setup_docks(self) -> None:
        self._histogram = HistogramWidget()
        hist_dock = QDockWidget("Histogramm", self)
        hist_dock.setWidget(self._histogram)
        hist_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, hist_dock)

        workflow_container = QWidget()
        wf_layout = QVBoxLayout(workflow_container)
        wf_layout.setContentsMargins(0, 0, 0, 0)
        self._workflow = WorkflowGraph(self._pipeline)
        wf_layout.addWidget(self._workflow)
        wf_dock = QDockWidget("Pipeline", self)
        wf_dock.setWidget(workflow_container)
        wf_dock.setAllowedAreas(
            Qt.DockWidgetArea.TopDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, wf_dock)

        self._comet_stack_panel = CometStackPanel(self._pipeline)
        comet_dock = QDockWidget("Komet-Stacking", self)
        comet_dock.setWidget(self._comet_stack_panel)
        comet_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, comet_dock)

        self._drizzle_panel = DrizzlePanel(self._pipeline)
        drizzle_dock = QDockWidget("Drizzle", self)
        drizzle_dock.setWidget(self._drizzle_panel)
        drizzle_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, drizzle_dock)

        self._mosaic_panel = MosaicPanel(self._pipeline)
        mosaic_dock = QDockWidget("Mosaic", self)
        mosaic_dock.setWidget(self._mosaic_panel)
        mosaic_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, mosaic_dock)

        self._deconvolution_panel = DeconvolutionPanel(self._pipeline)
        deconv_dock = QDockWidget("Deconvolution", self)
        deconv_dock.setWidget(self._deconvolution_panel)
        deconv_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, deconv_dock)

        self._starless_panel = StarlessPanel(self._pipeline)
        starless_dock = QDockWidget("Starless", self)
        starless_dock.setWidget(self._starless_panel)
        starless_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, starless_dock)

        self._channel_panel = ChannelCombinerPanel(self._pipeline)
        channel_dock = QDockWidget("Channel Combiner", self)
        channel_dock.setWidget(self._channel_panel)
        channel_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, channel_dock)

        self._color_calibration_panel = ColorCalibrationPanel(self._pipeline)
        color_cal_dock = QDockWidget("Farbkalibrierung", self)
        color_cal_dock.setWidget(self._color_calibration_panel)
        color_cal_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, color_cal_dock)

        self._annotation_panel = AnnotationPanel()
        annot_dock = QDockWidget("Annotationen", self)
        annot_dock.setWidget(self._annotation_panel)
        annot_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, annot_dock)

        self._progress = ProgressWidget()
        prog_dock = QDockWidget("Fortschritt", self)
        prog_dock.setWidget(self._progress)
        prog_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, prog_dock)

        self._benchmark = CalibrationBenchmarkWidget()
        bench_dock = QDockWidget("Kalibrierungs-Benchmark", self)
        bench_dock.setWidget(self._benchmark)
        bench_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bench_dock)

        self._log_widget = LogWidget()
        self._log_dock = QDockWidget("Fehler-Log", self)
        self._log_dock.setWidget(self._log_widget)
        self._log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)
        self._log_widget.install_root_handler()

    def _setup_menus(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&Datei")

        new_proj_act = QAction("&Neues Projekt", self)
        new_proj_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_N))  # type: ignore[operator]
        new_proj_act.triggered.connect(self._on_new_project)
        file_menu.addAction(new_proj_act)

        open_proj_act = QAction("Projekt &oeffnen...", self)
        open_proj_act.setShortcut(QKeySequence.StandardKey.Open)
        open_proj_act.triggered.connect(self._on_open_project)
        file_menu.addAction(open_proj_act)

        self._recent_menu = QMenu("Zuletzt geoeffnet", self)
        self._rebuild_recent_menu()
        file_menu.addMenu(self._recent_menu)

        file_menu.addSeparator()

        save_proj_act = QAction("Projekt &speichern", self)
        save_proj_act.setShortcut(QKeySequence.StandardKey.Save)
        save_proj_act.triggered.connect(self._on_save_project)
        file_menu.addAction(save_proj_act)

        save_as_act = QAction("Projekt speichern &unter...", self)
        save_as_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_S))  # type: ignore[operator]
        save_as_act.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_act)

        file_menu.addSeparator()

        open_img_act = QAction("&Bild oeffnen...", self)
        open_img_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_O))  # type: ignore[operator]
        open_img_act.triggered.connect(self._on_open_image)
        file_menu.addAction(open_img_act)

        file_menu.addSeparator()

        import_menu = QMenu("Frames &importieren", self)

        import_lights_act = QAction("&Light-Frames importieren...", self)
        import_lights_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_L))  # type: ignore[operator]
        import_lights_act.triggered.connect(self._on_import_lights)
        import_menu.addAction(import_lights_act)

        import_darks_act = QAction("&Dark-Frames importieren...", self)
        import_darks_act.triggered.connect(self._on_import_darks)
        import_menu.addAction(import_darks_act)

        import_flats_act = QAction("&Flat-Frames importieren...", self)
        import_flats_act.triggered.connect(self._on_import_flats)
        import_menu.addAction(import_flats_act)

        import_bias_act = QAction("&Bias-Frames importieren...", self)
        import_bias_act.triggered.connect(self._on_import_bias)
        import_menu.addAction(import_bias_act)

        file_menu.addMenu(import_menu)

        file_menu.addSeparator()
        quit_act = QAction("&Beenden", self)
        quit_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Q))  # type: ignore[operator]
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = menu_bar.addMenu("&Ansicht")
        fit_act = QAction("An Fenster &anpassen", self)
        fit_act.setShortcut(QKeySequence(Qt.Key.Key_F))
        fit_act.triggered.connect(self._viewer.fit_to_view)
        view_menu.addAction(fit_act)

        help_menu = menu_bar.addMenu("&Hilfe")
        license_act = QAction("&Lizenz verwalten...", self)
        license_act.triggered.connect(self._on_manage_license)
        help_menu.addAction(license_act)

    def _setup_statusbar(self) -> None:
        self._status_bar = self.statusBar()
        self._status_bar.showMessage("Bereit")
        self._license_badge = LicenseBadge()
        self._status_bar.addPermanentWidget(self._license_badge)

    def _connect_signals(self) -> None:
        self._viewer.zoom_changed.connect(self._on_zoom_changed)
        self._viewer.pixel_hovered.connect(self._on_pixel_hovered)
        self._file_loader.image_loaded.connect(self._on_image_loaded)
        self._file_loader.load_error.connect(self._on_load_error)
        self._file_loader.load_status.connect(self._on_load_status)
        self._license.status_changed.connect(self._license_badge.on_status_changed)
        self._license.status_changed.connect(self._offline_banner.on_status_changed)

        self._annotation_panel.dso_toggled.connect(self._annotation_overlay.set_show_dso)
        self._annotation_panel.stars_toggled.connect(self._annotation_overlay.set_show_stars)
        self._annotation_panel.boundaries_toggled.connect(self._annotation_overlay.set_show_boundaries)
        self._annotation_panel.grid_toggled.connect(self._annotation_overlay.set_show_grid)

        self._calibration_worker.metrics.connect(self._benchmark.update_metrics)
        self._calibration_worker.finished.connect(self._on_calibration_finished)
        self._calibration_worker.error.connect(self._on_calibration_error)

        self._pipeline.comet_preview_changed.connect(self._on_comet_preview_changed)

    @Slot()
    def _on_open_image(self) -> None:
        if self._file_loader.is_loading:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Bild oeffnen",
            "",
            "FITS (*.fits *.fit *.fts);;TIFF (*.tif *.tiff);;PNG (*.png);;Alle (*)",
        )
        if not path:
            return
        self._progress.set_indeterminate()
        self._progress.set_status("Lade...")
        self._file_loader.load(Path(path))

    _FITS_FILTER = "FITS (*.fits *.fit *.fts);;TIFF (*.tif *.tiff);;PNG (*.png);;Alle (*)"

    def _pick_files(self, title: str) -> list[str]:
        paths, _ = QFileDialog.getOpenFileNames(self, title, "", self._FITS_FILTER)
        return paths

    @Slot()
    def _on_import_lights(self) -> None:
        from astroai.project.project_file import FrameEntry

        paths = self._pick_files("Light-Frames importieren")
        if not paths:
            return
        existing = {e.path for e in self._project.input_frames}
        added = 0
        for p in paths:
            if p not in existing:
                self._project.input_frames.append(FrameEntry(path=p))
                existing.add(p)
                added += 1
        self._project.touch()
        self._status_bar.showMessage(f"{added} Light-Frame(s) zum Projekt hinzugefuegt")

    @Slot()
    def _on_import_darks(self) -> None:
        paths = self._pick_files("Dark-Frames importieren")
        if not paths:
            return
        existing = set(self._project.calibration.dark_frames)
        added = sum(1 for p in paths if p not in existing)
        self._project.calibration.dark_frames = list(existing | set(paths))
        self._project.touch()
        self._status_bar.showMessage(f"{added} Dark-Frame(s) zum Projekt hinzugefuegt")

    @Slot()
    def _on_import_flats(self) -> None:
        paths = self._pick_files("Flat-Frames importieren")
        if not paths:
            return
        existing = set(self._project.calibration.flat_frames)
        added = sum(1 for p in paths if p not in existing)
        self._project.calibration.flat_frames = list(existing | set(paths))
        self._project.touch()
        self._status_bar.showMessage(f"{added} Flat-Frame(s) zum Projekt hinzugefuegt")

    @Slot()
    def _on_import_bias(self) -> None:
        paths = self._pick_files("Bias-Frames importieren")
        if not paths:
            return
        existing = set(self._project.calibration.bias_frames)
        added = sum(1 for p in paths if p not in existing)
        self._project.calibration.bias_frames = list(existing | set(paths))
        self._project.touch()
        self._status_bar.showMessage(f"{added} Bias-Frame(s) zum Projekt hinzugefuegt")

    @Slot(object, str)
    def _on_image_loaded(self, data: object, name: str) -> None:
        import numpy as np

        img = data
        assert isinstance(img, np.ndarray)
        self._viewer.set_image_data(img)
        self._histogram.set_image_data(img)
        self._progress.reset()
        self._status_bar.showMessage(f"{name}  ({img.shape[1]}x{img.shape[0]})")

    @Slot(str)
    def _on_load_error(self, msg: str) -> None:
        import logging

        self._progress.reset()
        self._status_bar.showMessage(f"Fehler: {msg}")
        logging.getLogger("astroai.pipeline").error(msg)
        self._log_dock.setVisible(True)

    @Slot(str)
    def _on_load_status(self, msg: str) -> None:
        self._progress.set_status(msg)

    @Slot(float)
    def _on_zoom_changed(self, zoom: float) -> None:
        pct = zoom * 100
        msg = self._status_bar.currentMessage().split("|")[0].strip()
        self._status_bar.showMessage(f"{msg} | Zoom: {pct:.0f}%")

    @Slot(int, int, float)
    def _on_pixel_hovered(self, x: int, y: int, value: float) -> None:
        msg = self._status_bar.currentMessage().split("|")[0].strip()
        self._status_bar.showMessage(
            f"{msg} | ({x}, {y}) = {value:.2f}"
        )

    # -- project actions ------------------------------------------------

    @Slot()
    def _on_new_project(self) -> None:
        self._project = AstroProject()
        self._project_path = None
        self._pipeline.reset()
        self._update_title()
        self._status_bar.showMessage("Neues Projekt erstellt")

    @Slot()
    def _on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Projekt oeffnen", "",
            f"AstroAI Projekt (*{PROJECT_EXTENSION});;Alle (*)",
        )
        if path:
            self._load_project(Path(path))

    @Slot()
    def _on_save_project(self) -> None:
        if self._project_path:
            self._save_project(self._project_path)
        else:
            self._on_save_project_as()

    @Slot()
    def _on_save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Projekt speichern unter", "",
            f"AstroAI Projekt (*{PROJECT_EXTENSION})",
        )
        if path:
            p = Path(path)
            if p.suffix != PROJECT_EXTENSION:
                p = p.with_suffix(PROJECT_EXTENSION)
            self._save_project(p)

    def _save_project(self, path: Path) -> None:
        try:
            ProjectSerializer.save(self._project, path)
            self._project_path = path
            self._recent.add(path)
            self._rebuild_recent_menu()
            self._update_title()
            self._status_bar.showMessage(f"Projekt gespeichert: {path.name}")
        except ProjectSerializerError as exc:
            QMessageBox.critical(self, "Speicherfehler", str(exc))

    def _load_project(self, path: Path) -> None:
        try:
            self._project = ProjectSerializer.load(path)
            self._project_path = path
            self._recent.add(path)
            self._rebuild_recent_menu()
            self._pipeline.reset()
            self._update_title()
            self._status_bar.showMessage(f"Projekt geladen: {path.name}")
        except ProjectSerializerError as exc:
            QMessageBox.warning(self, "Ladefehler", str(exc))

    def _update_title(self) -> None:
        name = self._project.metadata.name
        if self._project_path:
            name = self._project_path.stem
        self.setWindowTitle(f"{name} - AstroAI Suite")

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        entries = self._recent.entries
        if not entries:
            no_act = self._recent_menu.addAction("(keine)")
            no_act.setEnabled(False)
            return
        for entry in entries:
            act = self._recent_menu.addAction(entry)
            act.triggered.connect(lambda checked=False, p=entry: self._load_project(Path(p)))

    @Slot()
    def _on_manage_license(self) -> None:
        dlg = ActivationDialog(self._license, self)
        dlg.exec()

    def set_wcs_solution(
        self,
        wcs: object | None,
        image_shape: tuple[int, int] | None = None,
    ) -> None:
        """Activate annotation overlay from plate-solve result.

        Accepts:
          - engine AnnotationOverlay (from platesolving.annotation)
          - SolveResult (requires image_shape)
          - astropy WCS (requires image_shape)
          - WcsTransform protocol instance
          - None to clear
        """
        from astroai.ui.overlay.sky_objects import WcsTransform
        from astroai.ui.overlay.wcs_adapter import WcsAdapter

        adapter: WcsTransform | None = None

        if wcs is None:
            pass
        elif isinstance(wcs, WcsTransform):
            adapter = wcs
        else:
            try:
                from astroai.engine.platesolving.annotation import (
                    AnnotationOverlay as EngineOverlay,
                )

                if isinstance(wcs, EngineOverlay):
                    adapter = WcsAdapter.from_engine_overlay(wcs)
            except ImportError:
                pass

            if adapter is None and image_shape is not None:
                try:
                    from astroai.engine.platesolving.solver import SolveResult

                    if isinstance(wcs, SolveResult):
                        adapter = WcsAdapter.from_solve_result(wcs, image_shape)
                except ImportError:
                    pass

            if adapter is None and image_shape is not None:
                try:
                    from astropy.wcs import WCS

                    if isinstance(wcs, WCS):
                        h, w = image_shape
                        adapter = WcsAdapter(wcs, w, h)
                except ImportError:
                    pass

        self._annotation_overlay.set_wcs(adapter)
        self._annotation_panel.set_wcs_active(adapter is not None)

        wcs_solution = None
        if adapter is not None:
            try:
                from astroai.astrometry.catalog import WcsSolution

                if isinstance(wcs, WcsSolution):
                    wcs_solution = wcs
            except ImportError:
                pass
        self._sky_overlay.set_solution(wcs_solution)

    @Slot()
    def _on_comet_preview_changed(self) -> None:
        preview = self._pipeline.comet_preview_image
        if preview is not None:
            self._viewer.set_image_data(preview)
            self._histogram.set_image_data(preview)
            mode_labels = {"stars": "Sterne", "comet": "Kometenkopf", "blend": "Blend"}
            label = mode_labels.get(self._pipeline.comet_tracking_mode, "")
            self._status_bar.showMessage(f"Komet-Vorschau: {label}")

    @Slot(object)
    def _on_calibration_finished(self, _results: object) -> None:
        self._benchmark.finish()

    @Slot(str)
    def _on_calibration_error(self, msg: str) -> None:
        import logging

        self._benchmark.reset()
        logging.getLogger("astroai.pipeline").error("Kalibrierung fehlgeschlagen: %s", msg)

    def require_tier(self, feature_name: str, required: LicenseTier) -> bool:
        """Check tier access; shows UpgradeDialog if insufficient. Returns True if allowed."""
        current = self._license.tier
        tier_order = [LicenseTier.FREE, LicenseTier.PRO_MONTHLY, LicenseTier.PRO_ANNUAL, LicenseTier.FOUNDING_MEMBER]
        if tier_order.index(current) >= tier_order.index(required):
            return True

        dlg = UpgradeDialog(feature_name, required, current, self)
        dlg.activate_requested.connect(self._on_manage_license)
        dlg.exec()
        return False


def main() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication(sys.argv))
    app.setApplicationName("AstroAI Suite")
    app.setApplicationVersion(__version__)
    app.setStyleSheet(_load_stylesheet())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
