"""AstroAI application entry point and MainWindow."""
from __future__ import annotations

import sys
from typing import cast
from pathlib import Path

import numpy as np

from PySide6.QtCore import QFile, Qt, QTextStream, Slot
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStackedLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from astroai import __version__
from astroai.project import AstroProject, ProjectSerializer
from astroai.project.project_file import PROJECT_EXTENSION
from astroai.project.recent_files import RecentProjects
from astroai.core.pipeline.presets import PresetManager
from astroai.project.serializer import ProjectSerializerError
from astroai.ui.license_adapter import QLicenseAdapter
from astroai.ui.main.loader import FileLoader
from astroai.ui.models import PipelineModel
from astroai.ui.widgets.activation_dialog import ActivationDialog
from astroai.ui.widgets.live_histogram_view import HistogramView
from astroai.ui.widgets.image_viewer import ImageViewer
from astroai.ui.widgets.license_badge import LicenseBadge
from astroai.ui.widgets.offline_banner import OfflineBanner
from astroai.ui.widgets.log_widget import LogWidget
from astroai.core.calibration.worker import CalibrationWorker
from astroai.core.pipeline.builder import PipelineBuilder
from astroai.core.pipeline.runner import PipelineWorker
from astroai.core.pipeline.base import PipelineContext
from astroai.core.processing_history import ProcessingHistory
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
from astroai.ui.widgets.shortcuts_dialog import ShortcutsDialog
from astroai.ui.widgets.smart_calib_panel import SmartCalibPanel
from astroai.ui.widgets.synthetic_flat_panel import SyntheticFlatPanel
from astroai.ui.widgets.frame_quality_dashboard import FrameQualityDashboard
from astroai.ui.widgets.frame_selection_panel import FrameSelectionPanel
from astroai.ui.widgets.registration_panel import RegistrationPanel
from astroai.ui.widgets.background_removal_panel import BackgroundRemovalPanel
from astroai.ui.widgets.denoise_panel import DenoisePanel
from astroai.ui.widgets.model_manager_panel import ModelManagerPanel
from astroai.ui.widgets.curves_panel import CurvesPanel
from astroai.ui.widgets.stretch_panel import StretchPanel
from astroai.ui.widgets.starless_panel import StarlessPanel
from astroai.ui.widgets.star_processing_panel import StarProcessingPanel
from astroai.ui.widgets.stacking_panel import StackingPanel
from astroai.ui.widgets.export_panel import ExportPanel
from astroai.ui.widgets.frame_list_panel import FrameListPanel
from astroai.ui.widgets.session_notes_panel import SessionNotesPanel
from astroai.ui.widgets.image_stats_widget import ImageStatsWidget
from astroai.ui.widgets.split_compare_view import SplitCompareView
from astroai.ui.widgets.fits_metadata_panel import FITSMetadataPanel
from astroai.ui.widgets.workflow_graph import WorkflowGraph
from astroai.ui.overlay.annotation_overlay import AnnotationOverlay
from astroai.ui.widgets.sky_overlay import SkyOverlay
from astroai.licensing.models import LicenseTier

__all__ = ["MainWindow", "main"]

# Maps PipelineStage.name → PipelineModel step key for WorkflowGraph tracking
_STAGE_TO_STEP_KEY: dict[str, str] = {
    "CALIBRATION": "calibrate",
    "REGISTRATION": "register",
    "STACKING": "stack",
    "COMET_STACKING": "comet_stacking",
    "DRIZZLE": "drizzle",
    "MOSAIC": "mosaic",
    "ASTROMETRY": "export",      # plate solving maps closest to export stage
    "PHOTOMETRY": "color_calibration",
    "PROCESSING": "stretch",
    "SAVING": "export",
}


def _resources_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "astroai" / "ui" / "resources"  # type: ignore[attr-defined]  # pragma: no cover
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


def _apply_auto_stretch(data: np.ndarray) -> np.ndarray:
    """Percentile auto-stretch for preview display. Does not modify source data."""
    arr = np.asarray(data, dtype=np.float32)
    lo = float(np.percentile(arr, 0.5))
    hi = float(np.percentile(arr, 99.5))
    if hi <= lo:
        return cast(np.ndarray, arr.copy())
    return cast(np.ndarray, np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32))


def _enrich_fits_entry(entry: object) -> None:
    """Populate FrameEntry metadata from FITS header. Silent no-op on failure."""
    from astroai.project.project_file import FrameEntry
    if not isinstance(entry, FrameEntry):
        return
    p = entry.path
    if not any(p.lower().endswith(s) for s in (".fits", ".fit", ".fts")):
        return
    try:
        from astropy.io import fits as _fits
        with _fits.open(p, memmap=False) as hdul:
            hdr = hdul[0].header
            raw_exp = hdr.get("EXPTIME") or hdr.get("EXPOSURE")
            if raw_exp is not None:
                entry.exposure = float(raw_exp)
            raw_gain = hdr.get("GAIN")
            if raw_gain is not None:
                entry.gain_iso = int(float(raw_gain))
            raw_temp = hdr.get("CCD-TEMP") or hdr.get("CCD_TEMP")
            if raw_temp is not None:
                entry.temperature = float(raw_temp)
    except Exception:
        pass


def _enrich_raw_entry(entry: object) -> None:
    """Populate FrameEntry metadata from RAW EXIF. Silent no-op on failure."""
    from astroai.project.project_file import FrameEntry
    from astroai.core.io.raw_io import RAW_EXTENSIONS
    if not isinstance(entry, FrameEntry):
        return
    if Path(entry.path).suffix.lower() not in RAW_EXTENSIONS:
        return
    try:
        from astroai.core.io.raw_io import read_raw_metadata
        meta = read_raw_metadata(Path(entry.path))
        if meta.exposure is not None:
            entry.exposure = meta.exposure
        if meta.gain_iso is not None:
            entry.gain_iso = meta.gain_iso
        if meta.temperature is not None:
            entry.temperature = meta.temperature
    except Exception:
        pass


class MainWindow(QMainWindow):
    """Primary application window with dock-based layout."""

    def __init__(self, license_adapter: QLicenseAdapter | None = None) -> None:
        super().__init__()
        self.setWindowTitle("AstroAI Suite")
        self.setMinimumSize(960, 640)

        self._license = license_adapter or QLicenseAdapter(self)
        self._pipeline = PipelineModel(self)
        self._calibration_worker = CalibrationWorker(self)
        self._pipeline_worker = PipelineWorker(self)
        self._pipeline_builder = PipelineBuilder()
        self._processing_history = ProcessingHistory(parent=self)
        self._current_image: object = None
        self._before_image: object = None  # image snapshot before pipeline run
        self._processing_base_image: object = None  # base image for undo re-apply
        self._auto_stretch: bool = False
        self._wcs_adapter: object = None  # WcsTransform | None
        self._file_loader = FileLoader(self)
        self._project = AstroProject()
        self._project_path: Path | None = None
        self._recent = RecentProjects()
        self._preset_manager = PresetManager()
        self._setup_presets()
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

        # Page 0: normal viewer with overlays
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

        # Page 1: split before/after comparison
        self._compare_view = SplitCompareView()

        self._view_stack = QStackedWidget()
        self._view_stack.addWidget(viewer_container)
        self._view_stack.addWidget(self._compare_view)

        central_layout.addWidget(self._view_stack, stretch=1)
        self.setCentralWidget(central)

    def _setup_docks(self) -> None:
        self._histogram = HistogramView()
        hist_dock = QDockWidget("Histogramm", self)
        hist_dock.setWidget(self._histogram)
        hist_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, hist_dock)

        self._image_stats = ImageStatsWidget()
        stats_dock = QDockWidget("Bildstatistik", self)
        stats_dock.setWidget(self._image_stats)
        stats_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, stats_dock)

        self._fits_metadata = FITSMetadataPanel()
        fits_dock = QDockWidget("FITS-Metadaten", self)
        fits_dock.setWidget(self._fits_metadata)
        fits_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, fits_dock)

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

        self._star_processing_panel = StarProcessingPanel(self._pipeline)
        star_proc_dock = QDockWidget("Sternverarbeitung", self)
        star_proc_dock.setWidget(self._star_processing_panel)
        star_proc_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, star_proc_dock)

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

        self._synthetic_flat_panel = SyntheticFlatPanel(self._pipeline)
        synth_flat_dock = QDockWidget("Synth. Flat", self)
        synth_flat_dock.setWidget(self._synthetic_flat_panel)
        synth_flat_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, synth_flat_dock)

        self._smart_calib_panel = SmartCalibPanel(lambda: self._project)
        smart_calib_dock = QDockWidget("Smart-Kalibrierung", self)
        smart_calib_dock.setWidget(self._smart_calib_panel)
        smart_calib_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, smart_calib_dock)

        self._stacking_panel = StackingPanel(self._pipeline)
        stacking_dock = QDockWidget("Stacking", self)
        stacking_dock.setWidget(self._stacking_panel)
        stacking_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, stacking_dock)

        self._frame_list_panel = FrameListPanel()
        frame_list_dock = QDockWidget("Light-Frames", self)
        frame_list_dock.setWidget(self._frame_list_panel)
        frame_list_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, frame_list_dock)

        self._frame_selection_panel = FrameSelectionPanel(self._pipeline)
        frame_sel_dock = QDockWidget("Frame-Selektion", self)
        frame_sel_dock.setWidget(self._frame_selection_panel)
        frame_sel_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, frame_sel_dock)

        self._frame_quality_dashboard = FrameQualityDashboard()
        fq_dock = QDockWidget("Frame-Qualitaets-Dashboard", self)
        fq_dock.setWidget(self._frame_quality_dashboard)
        fq_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, fq_dock)

        self._registration_panel = RegistrationPanel(self._pipeline)
        registration_dock = QDockWidget("Registrierung", self)
        registration_dock.setWidget(self._registration_panel)
        registration_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, registration_dock)

        self._background_removal_panel = BackgroundRemovalPanel(self._pipeline)
        bg_removal_dock = QDockWidget("Hintergrundentfernung", self)
        bg_removal_dock.setWidget(self._background_removal_panel)
        bg_removal_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, bg_removal_dock)

        self._denoise_panel = DenoisePanel(self._pipeline)
        denoise_dock = QDockWidget("Entrauschen", self)
        denoise_dock.setWidget(self._denoise_panel)
        denoise_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, denoise_dock)

        self._stretch_panel = StretchPanel(self._pipeline)
        stretch_dock = QDockWidget("Stretching", self)
        stretch_dock.setWidget(self._stretch_panel)
        stretch_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, stretch_dock)

        self._curves_panel = CurvesPanel(self._pipeline)
        curves_dock = QDockWidget("Kurven", self)
        curves_dock.setWidget(self._curves_panel)
        curves_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, curves_dock)

        self._export_panel = ExportPanel(self._pipeline)
        export_dock = QDockWidget("Export", self)
        export_dock.setWidget(self._export_panel)
        export_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, export_dock)

        self._session_notes_panel = SessionNotesPanel()
        notes_dock = QDockWidget("Session-Notizen", self)
        notes_dock.setWidget(self._session_notes_panel)
        notes_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, notes_dock)

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

        self._model_manager = ModelManagerPanel()
        self._model_manager_dock = QDockWidget("KI-Modelle", self)
        self._model_manager_dock.setWidget(self._model_manager)
        self._model_manager_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._model_manager_dock)
        self._model_manager_dock.setVisible(False)
        self._model_manager.status_message.connect(self._on_status_message)

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
        import_folder_act = QAction("&Ordner importieren...", self)
        import_folder_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_F))  # type: ignore[operator]
        import_folder_act.triggered.connect(self._on_import_from_folder)
        import_menu.addAction(import_folder_act)

        import_darks_act = QAction("&Dark-Frames importieren...", self)
        import_darks_act.triggered.connect(self._on_import_darks)
        import_menu.addAction(import_darks_act)

        import_flats_act = QAction("&Flat-Frames importieren...", self)
        import_flats_act.triggered.connect(self._on_import_flats)
        import_menu.addAction(import_flats_act)

        import_bias_act = QAction("&Bias-Frames importieren...", self)
        import_bias_act.triggered.connect(self._on_import_bias)
        import_menu.addAction(import_bias_act)

        import_menu.addSeparator()
        auto_match_act = QAction("&Kalibrierung automatisch zuordnen...", self)
        auto_match_act.setShortcut(
            QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_K)  # type: ignore[operator]
        )
        auto_match_act.triggered.connect(self._on_auto_match_calibration)
        import_menu.addAction(auto_match_act)

        file_menu.addMenu(import_menu)

        file_menu.addSeparator()
        export_stats_act = QAction("Frames-Statistik &exportieren...", self)
        export_stats_act.setShortcut(
            QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_E)  # type: ignore[operator]
        )
        export_stats_act.triggered.connect(self._on_export_frame_stats)
        file_menu.addAction(export_stats_act)

        file_menu.addSeparator()
        quit_act = QAction("&Beenden", self)
        quit_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Q))  # type: ignore[operator]
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        edit_menu = menu_bar.addMenu("&Bearbeiten")

        self._undo_act = QAction("&Rückgängig", self)
        self._undo_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Z))  # type: ignore[operator]
        self._undo_act.setEnabled(False)
        self._undo_act.triggered.connect(self._on_undo)
        edit_menu.addAction(self._undo_act)

        self._redo_act = QAction("&Wiederherstellen", self)
        self._redo_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Y))  # type: ignore[operator]
        self._redo_act.setEnabled(False)
        self._redo_act.triggered.connect(self._on_redo)
        edit_menu.addAction(self._redo_act)

        edit_menu.addSeparator()

        self._copy_image_act = QAction("Bild &kopieren", self)
        self._copy_image_act.setShortcut(
            QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_C)  # type: ignore[operator]
        )
        self._copy_image_act.setEnabled(False)
        self._copy_image_act.triggered.connect(self._on_copy_image)
        edit_menu.addAction(self._copy_image_act)
        self._save_preview_act = QAction("Vorschau-Bild &speichern...", self)
        self._save_preview_act.setShortcut(
            QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_P)  # type: ignore[operator]
        )
        self._save_preview_act.setEnabled(False)
        self._save_preview_act.triggered.connect(self._on_save_preview_image)
        edit_menu.addAction(self._save_preview_act)

        view_menu = menu_bar.addMenu("&Ansicht")
        fit_act = QAction("An Fenster &anpassen", self)
        fit_act.setShortcut(QKeySequence(Qt.Key.Key_F))
        fit_act.triggered.connect(self._on_fit_to_view)
        view_menu.addAction(fit_act)

        self._auto_stretch_act = QAction("Auto-&Stretch Vorschau", self)
        self._auto_stretch_act.setShortcut(
            QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_A)  # type: ignore[operator]
        )
        self._auto_stretch_act.setCheckable(True)
        self._auto_stretch_act.setEnabled(False)
        self._auto_stretch_act.triggered.connect(self._on_toggle_auto_stretch)
        view_menu.addAction(self._auto_stretch_act)

        view_menu.addSeparator()
        self._compare_act = QAction("&Vorher/Nachher Vergleich", self)
        self._compare_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_D))  # type: ignore[operator]
        self._compare_act.setCheckable(True)
        self._compare_act.setEnabled(False)
        self._compare_act.triggered.connect(self._on_toggle_compare)
        view_menu.addAction(self._compare_act)

        pipeline_menu = menu_bar.addMenu("&Pipeline")
        self._run_act = QAction("&Ausführen", self)
        self._run_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_R))  # type: ignore[operator]
        self._run_act.setEnabled(False)
        self._run_act.triggered.connect(self._on_run_pipeline)
        pipeline_menu.addAction(self._run_act)

        self._stack_run_act = QAction("&Stack && Process", self)
        self._stack_run_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_R))  # type: ignore[operator]
        self._stack_run_act.setEnabled(False)
        self._stack_run_act.triggered.connect(self._on_run_full_pipeline)
        pipeline_menu.addAction(self._stack_run_act)

        self._cancel_act = QAction("A&bbrechen", self)
        self._cancel_act.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        self._cancel_act.setEnabled(False)
        self._cancel_act.triggered.connect(self._on_cancel_pipeline)
        pipeline_menu.addAction(self._cancel_act)

        pipeline_menu.addSeparator()
        save_preset_act = QAction("Preset &speichern...", self)
        save_preset_act.triggered.connect(self._on_save_preset)
        pipeline_menu.addAction(save_preset_act)

        self._preset_menu = QMenu("Preset &laden", self)
        self._rebuild_preset_menu()
        pipeline_menu.addMenu(self._preset_menu)

        extras_menu = menu_bar.addMenu("&Extras")
        model_mgr_act = QAction("KI-&Modelle verwalten...", self)
        model_mgr_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_M))  # type: ignore[operator]
        model_mgr_act.triggered.connect(self._on_show_model_manager)
        extras_menu.addAction(model_mgr_act)

        help_menu = menu_bar.addMenu("&Hilfe")
        summary_act = QAction("&Projektübersicht...", self)
        summary_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_I))  # type: ignore[operator]
        summary_act.triggered.connect(self._on_show_project_summary)
        help_menu.addAction(summary_act)
        shortcuts_act = QAction("&Tastaturkürzel...", self)
        shortcuts_act.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Question))  # type: ignore[operator]
        shortcuts_act.triggered.connect(self._on_show_shortcuts)
        help_menu.addAction(shortcuts_act)
        help_menu.addSeparator()
        license_act = QAction("&Lizenz verwalten...", self)
        license_act.triggered.connect(self._on_manage_license)
        help_menu.addAction(license_act)

    def _setup_statusbar(self) -> None:
        self._status_bar = self.statusBar()
        self._status_bar.showMessage("Bereit")
        self._backend_label = QLabel(self._get_backend_label())
        self._backend_label.setStyleSheet("color: #8cf; font-size: 11px; padding: 0 6px;")
        self._backend_label.setToolTip("Aktives Inference-Backend")
        self._status_bar.addPermanentWidget(self._backend_label)
        self._calib_status_label = QLabel("Kalib.: —")
        self._calib_status_label.setStyleSheet("color: #aaa; font-size: 11px; padding: 0 6px;")
        self._calib_status_label.setToolTip("Anzahl konfigurierter Kalibrierungs-Frames")
        self._status_bar.addPermanentWidget(self._calib_status_label)
        self._zoom_label = QLabel("Zoom: 100%")
        self._zoom_label.setStyleSheet("color: #aaa; font-size: 11px; padding: 0 6px;")
        self._status_bar.addPermanentWidget(self._zoom_label)
        self._license_badge = LicenseBadge()
        self._status_bar.addPermanentWidget(self._license_badge)

    @staticmethod
    def _get_backend_label() -> str:
        try:
            from astroai.core.onnx_registry import OnnxModelRegistry
            return OnnxModelRegistry().backend_label
        except Exception:
            return "[CPU]"

    def _refresh_calib_status(self) -> None:
        cfg = self._project.calibration
        parts: list[str] = []
        if cfg.dark_frames:
            parts.append(f"{len(cfg.dark_frames)}D")
        if cfg.flat_frames:
            parts.append(f"{len(cfg.flat_frames)}F")
        if cfg.bias_frames:
            parts.append(f"{len(cfg.bias_frames)}B")
        self._calib_status_label.setText(
            f"Kalib.: {' / '.join(parts)}" if parts else "Kalib.: —"
        )

    def _connect_signals(self) -> None:
        self._viewer.zoom_changed.connect(self._on_zoom_changed)
        self._compare_view.zoom_changed.connect(self._on_zoom_changed)
        self._viewer.pixel_hovered.connect(self._on_pixel_hovered)
        self._file_loader.image_loaded.connect(self._on_image_loaded)
        self._file_loader.header_loaded.connect(self._fits_metadata.set_header)
        self._file_loader.load_error.connect(self._on_load_error)
        self._file_loader.load_status.connect(self._on_load_status)
        self._license.status_changed.connect(self._license_badge.on_status_changed)
        self._license.status_changed.connect(self._offline_banner.on_status_changed)

        self._annotation_panel.dso_toggled.connect(self._annotation_overlay.set_show_dso)
        self._annotation_panel.stars_toggled.connect(self._annotation_overlay.set_show_stars)
        self._annotation_panel.boundaries_toggled.connect(self._annotation_overlay.set_show_boundaries)
        self._annotation_panel.grid_toggled.connect(self._annotation_overlay.set_show_grid)

        self._annotation_panel.dso_toggled.connect(
            lambda v: setattr(self._pipeline, "annotation_show_dso", v)
        )
        self._annotation_panel.stars_toggled.connect(
            lambda v: setattr(self._pipeline, "annotation_show_stars", v)
        )
        self._annotation_panel.boundaries_toggled.connect(
            lambda v: setattr(self._pipeline, "annotation_show_boundaries", v)
        )
        self._annotation_panel.grid_toggled.connect(
            lambda v: setattr(self._pipeline, "annotation_show_grid", v)
        )
        self._pipeline.annotation_config_changed.connect(self._sync_annotation_from_model)
        self._pipeline.histogram_changed.connect(self._histogram.set_image_data)

        self._processing_history.history_changed.connect(self._on_history_changed)

        self._calibration_worker.metrics.connect(self._benchmark.update_metrics)
        self._calibration_worker.finished.connect(self._on_calibration_finished)
        self._calibration_worker.error.connect(self._on_calibration_error)

        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.cancelled.connect(self._on_pipeline_cancelled)
        self._pipeline_worker.error.connect(self._on_pipeline_error)
        self._pipeline_worker.progress.connect(self._on_pipeline_progress)
        self._pipeline_worker.stage_active.connect(self._on_pipeline_stage_active)

        self._pipeline.comet_preview_changed.connect(self._on_comet_preview_changed)
        self._session_notes_panel.text_changed.connect(self._on_session_notes_changed)
        self._frame_list_panel.selection_changed.connect(self._on_frame_selection_changed)
        self._frame_list_panel.remove_requested.connect(self._on_frames_remove_requested)
        self._frame_list_panel.files_dropped.connect(self._on_frames_dropped)
        self._frame_list_panel.preview_requested.connect(self._on_frame_preview_requested)
        self._denoise_panel.download_model_requested.connect(self._on_download_model_requested)

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

    _FITS_FILTER = (
        "FITS (*.fits *.fit *.fts);;"
        "TIFF (*.tif *.tiff);;"
        "PNG (*.png);;"
        "RAW (*.cr2 *.cr3 *.nef *.arw *.dng *.orf *.rw2);;"
        "Alle (*)"
    )

    def _pick_files(self, title: str) -> list[str]:
        paths, _ = QFileDialog.getOpenFileNames(self, title, "", self._FITS_FILTER)
        return paths

    @Slot()
    def _on_import_lights(self) -> None:
        paths = self._pick_files("Light-Frames importieren")
        if not paths:
            return
        self._add_light_frames(paths)

    @Slot()
    def _on_import_from_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "FITS-Dateien aus Ordner importieren",
            "",
        )
        if not directory:
            return
        _FITS_SUFFIXES = {".fits", ".fit", ".fts"}
        paths = [
            str(p)
            for p in Path(directory).rglob("*")
            if p.suffix.lower() in _FITS_SUFFIXES
        ]
        if not paths:
            self._status_bar.showMessage(
                f"Keine FITS-Dateien gefunden in: {Path(directory).name}"
            )
            return
        self._add_light_frames(paths)

    @Slot(list)
    def _on_frames_dropped(self, paths: list[str]) -> None:
        self._add_light_frames(paths)

    @Slot(str)
    def _on_frame_preview_requested(self, path: str) -> None:
        self._file_loader.load(Path(path))

    def _add_light_frames(self, paths: list[str]) -> None:
        """Add new FrameEntry objects for *paths* not already in the project."""
        from astroai.project.project_file import FrameEntry

        existing = {e.path for e in self._project.input_frames}
        added = 0
        for p in paths:
            if p not in existing:
                entry = FrameEntry(path=p)
                _enrich_fits_entry(entry)
                _enrich_raw_entry(entry)
                self._project.input_frames.append(entry)
                existing.add(p)
                added += 1
        if added:
            self._project.touch()
            self._frame_list_panel.refresh(self._project.input_frames)
            self._stack_run_act.setEnabled(True)
            self._update_title()
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
        self._refresh_calib_status()
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
        self._refresh_calib_status()
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
        self._refresh_calib_status()
        self._status_bar.showMessage(f"{added} Bias-Frame(s) zum Projekt hinzugefuegt")

    @Slot()
    def _on_auto_match_calibration(self) -> None:
        from astroai.core.calibration.scanner import build_calibration_library, scan_directory
        from astroai.core.calibration.matcher import batch_match, suggest_calibration_config
        from astroai.core.io.fits_io import ImageMetadata

        directory = QFileDialog.getExistingDirectory(
            self, "Verzeichnis mit Kalibrierungsframes auswählen"
        )
        if not directory:
            return

        self._status_bar.showMessage("Scanne Verzeichnis nach Kalibrierungsframes…")
        frames = scan_directory(Path(directory), recursive=True)
        library = build_calibration_library(frames)

        total_cal = len(library.darks) + len(library.flats) + len(library.bias)
        if total_cal == 0:
            QMessageBox.information(
                self,
                "Kein Ergebnis",
                f"Keine Kalibrierungsframes im Verzeichnis gefunden:\n{directory}",
            )
            self._status_bar.showMessage("Automatisches Matching: keine Kalibrierungsframes gefunden")
            return

        # Build lights metadata list from project
        from astroai.core.io.fits_io import read_fits

        lights: list[tuple[Path, ImageMetadata]] = []
        for fe in self._project.input_frames:
            if fe.selected and Path(fe.path).exists():
                try:
                    _d, meta = read_fits(fe.path)
                    lights.append((Path(fe.path), meta))
                except Exception:
                    lights.append((Path(fe.path), ImageMetadata()))

        result = batch_match(lights, library)
        cfg = suggest_calibration_config(result)

        dark_count = len(library.darks)
        flat_count = len(library.flats)
        bias_count = len(library.bias)
        coverage_pct = int(result.coverage * 100)

        msg = (
            f"Gefunden: {dark_count} Dark(s), {flat_count} Flat(s), {bias_count} Bias\n"
            f"Abdeckung: {coverage_pct}% der ausgewählten Frames\n\n"
            "Kalibrierung im Projekt aktualisieren?"
        )
        reply = QMessageBox.question(
            self,
            "Automatisches Kalibrierungs-Matching",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._project.calibration = cfg
            self._project.touch()
            self._update_title()
            self._refresh_calib_status()
            self._status_bar.showMessage(
                f"Kalibrierung aktualisiert: {len(cfg.dark_frames)} Dark(s), "
                f"{len(cfg.flat_frames)} Flat(s), {len(cfg.bias_frames)} Bias"
            )
        else:
            self._status_bar.showMessage("Automatisches Matching abgebrochen")

    @Slot()
    def _on_export_frame_stats(self) -> None:
        from astroai.core.io.frame_stats_export import export_frame_stats

        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Frames-Statistik exportieren",
            "frame_stats.csv",
            "CSV-Dateien (*.csv)",
        )
        if not dest:
            return
        rows = export_frame_stats(self._project.input_frames, Path(dest))
        self._status_bar.showMessage(f"Frames-Statistik exportiert: {rows} Eintraege → {Path(dest).name}")

    def _display_image_data(self, img: np.ndarray) -> None:
        """Push image to viewer/histogram/stats, applying auto-stretch when active."""
        display = _apply_auto_stretch(img) if self._auto_stretch else img
        self._viewer.set_image_data(display)
        self._pipeline.histogram_changed.emit(display)
        self._image_stats.set_image_data(display)

    @Slot(object, str)
    def _on_image_loaded(self, data: object, name: str) -> None:
        img = data
        assert isinstance(img, np.ndarray)
        self._current_image = img
        self._run_act.setEnabled(True)
        self._auto_stretch_act.setEnabled(True)
        self._copy_image_act.setEnabled(True)
        self._save_preview_act.setEnabled(True)
        self._display_image_data(img)
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
        self._zoom_label.setText(f"Zoom: {zoom * 100:.0f}%")

    @Slot(int, int, float)
    def _on_pixel_hovered(self, x: int, y: int, value: float) -> None:
        msg = self._status_bar.currentMessage().split("|")[0].strip()
        coords = f"({x}, {y}) = {value:.4f}"
        if self._wcs_adapter is not None:
            try:
                from astroai.ui.overlay.sky_objects import WcsTransform
                if isinstance(self._wcs_adapter, WcsTransform):
                    world = self._wcs_adapter.pixel_to_world(float(x), float(y))
                    if world is not None:
                        ra, dec = world
                        coords += f"  RA {ra:.5f}° Dec {dec:+.5f}°"
            except Exception:
                pass
        self._status_bar.showMessage(f"{msg} | {coords}")

    # -- project actions ------------------------------------------------

    def _maybe_discard_changes(self) -> bool:
        """Return True if it is safe to abandon current project (clean or user confirmed)."""
        if not self._project.is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Nicht gespeicherte Änderungen",
            "Das Projekt hat nicht gespeicherte Änderungen.\nTrotzdem fortfahren?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self._on_save_project()
            return True
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return False

    def closeEvent(self, event: QCloseEvent) -> None:
        # Only prompt for unsaved changes on user-initiated close (not programmatic)
        if event.spontaneous() and not self._maybe_discard_changes():
            event.ignore()
        else:
            event.accept()

    @Slot()
    def _on_new_project(self) -> None:
        if not self._maybe_discard_changes():
            return
        self._project = AstroProject()
        self._project_path = None
        self._pipeline.reset()
        self._session_notes_panel.set_notes("")
        self._fits_metadata.clear()
        self._reset_compare_state()
        self._update_title()
        self._status_bar.showMessage("Neues Projekt erstellt")

    @Slot()
    def _on_open_project(self) -> None:
        if not self._maybe_discard_changes():
            return
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
        self._sync_model_to_project()
        try:
            ProjectSerializer.save(self._project, path)
            self._project_path = path
            self._recent.add(path)
            self._rebuild_recent_menu()
            self._project.mark_clean()
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
            self._sync_project_to_model()
            self._refresh_calib_status()
            self._session_notes_panel.set_notes(self._project.metadata.description)
            self._pipeline.reset()
            self._reset_compare_state()
            self._update_title()
            self._status_bar.showMessage(f"Projekt geladen: {path.name}")
        except ProjectSerializerError as exc:
            QMessageBox.warning(self, "Ladefehler", str(exc))

    def _sync_model_to_project(self) -> None:
        p = self._pipeline
        self._project.synthetic_flat.enabled = p.synthetic_flat_enabled
        self._project.synthetic_flat.tile_size = p.synthetic_flat_tile_size
        self._project.synthetic_flat.smoothing_sigma = p.synthetic_flat_smoothing_sigma
        self._project.comet_stack.enabled = p.comet_stack_enabled
        self._project.comet_stack.tracking_mode = p.comet_tracking_mode
        self._project.comet_stack.blend_factor = p.comet_blend_factor
        self._project.drizzle.enabled = p.drizzle_enabled
        self._project.drizzle.drop_size = p.drizzle_drop_size
        self._project.drizzle.scale = p.drizzle_scale
        self._project.drizzle.pixfrac = p.drizzle_pixfrac
        self._project.mosaic.enabled = p.mosaic_enabled
        self._project.mosaic.blend_mode = p.mosaic_blend_mode
        self._project.mosaic.gradient_correct = p.mosaic_gradient_correct
        self._project.mosaic.output_scale = p.mosaic_output_scale
        self._project.channel_combine.enabled = p.channel_combine_enabled
        self._project.channel_combine.mode = p.channel_combine_mode
        self._project.channel_combine.palette = p.channel_combine_palette
        self._project.color_calibration.enabled = p.color_calibration_enabled
        self._project.color_calibration.catalog = p.color_calibration_catalog
        self._project.color_calibration.sample_radius = p.color_calibration_sample_radius
        self._project.deconvolution.enabled = p.deconvolution_enabled
        self._project.deconvolution.iterations = p.deconvolution_iterations
        self._project.deconvolution.psf_sigma = p.deconvolution_psf_sigma
        self._project.starless.enabled = p.starless_enabled
        self._project.starless.strength = p.starless_strength
        self._project.starless.format = p.starless_format
        self._project.starless.save_star_mask = p.save_star_mask
        self._project.frame_selection.enabled = p.frame_selection_enabled
        self._project.frame_selection.min_score = p.frame_selection_min_score
        self._project.frame_selection.max_rejected_fraction = p.frame_selection_max_rejected_fraction
        self._project.registration.upsample_factor = p.registration_upsample_factor
        self._project.registration.reference_frame_index = p.registration_reference_frame_index
        self._project.background_removal.enabled = p.background_removal_enabled
        self._project.background_removal.tile_size = p.background_removal_tile_size
        self._project.background_removal.method = p.background_removal_method
        self._project.background_removal.preserve_median = p.background_removal_preserve_median
        self._project.denoise.strength = p.denoise_strength
        self._project.denoise.tile_size = p.denoise_tile_size
        self._project.denoise.tile_overlap = p.denoise_tile_overlap
        self._project.stretch.target_background = p.stretch_target_background
        self._project.stretch.shadow_clipping_sigmas = p.stretch_shadow_clipping_sigmas
        self._project.stretch.linked_channels = p.stretch_linked_channels
        self._project.star_processing.reduce_enabled = p.star_reduce_enabled
        self._project.star_processing.reduce_factor = p.star_reduce_factor
        self._project.star_processing.detection_sigma = p.star_detection_sigma
        self._project.star_processing.min_area = p.star_min_area
        self._project.star_processing.max_area = p.star_max_area
        self._project.star_processing.mask_dilation = p.star_mask_dilation
        self._project.stacking.method = p.stacking_method
        self._project.stacking.sigma_low = p.stacking_sigma_low
        self._project.stacking.sigma_high = p.stacking_sigma_high
        self._project.curves.enabled = p.curves_enabled
        self._project.curves.rgb_points = [list(pt) for pt in p.curves_rgb_points]
        self._project.curves.r_points = [list(pt) for pt in p.curves_r_points]
        self._project.curves.g_points = [list(pt) for pt in p.curves_g_points]
        self._project.curves.b_points = [list(pt) for pt in p.curves_b_points]
        self._project.annotation.show_dso = p.annotation_show_dso
        self._project.annotation.show_stars = p.annotation_show_stars
        self._project.annotation.show_boundaries = p.annotation_show_boundaries
        self._project.annotation.show_grid = p.annotation_show_grid
        self._project.output_path = p.output_path
        self._project.output_format = p.output_format
        self._project.metadata.description = self._session_notes_panel.notes

    def _sync_project_to_model(self) -> None:
        p = self._pipeline
        sf = self._project.synthetic_flat
        p.synthetic_flat_enabled = sf.enabled
        p.synthetic_flat_tile_size = sf.tile_size
        p.synthetic_flat_smoothing_sigma = sf.smoothing_sigma
        cs = self._project.comet_stack
        p.comet_stack_enabled = cs.enabled
        p.comet_tracking_mode = cs.tracking_mode
        p.comet_blend_factor = cs.blend_factor
        dr = self._project.drizzle
        p.drizzle_enabled = dr.enabled
        p.drizzle_drop_size = dr.drop_size
        p.drizzle_scale = dr.scale
        p.drizzle_pixfrac = dr.pixfrac
        mo = self._project.mosaic
        p.mosaic_enabled = mo.enabled
        p.mosaic_blend_mode = mo.blend_mode
        p.mosaic_gradient_correct = mo.gradient_correct
        p.mosaic_output_scale = mo.output_scale
        cc = self._project.channel_combine
        p.channel_combine_enabled = cc.enabled
        p.channel_combine_mode = cc.mode
        p.channel_combine_palette = cc.palette
        cal = self._project.color_calibration
        p.color_calibration_enabled = cal.enabled
        p.color_calibration_catalog = cal.catalog
        p.color_calibration_sample_radius = cal.sample_radius
        dec = self._project.deconvolution
        p.deconvolution_enabled = dec.enabled
        p.deconvolution_iterations = dec.iterations
        p.deconvolution_psf_sigma = dec.psf_sigma
        sl = self._project.starless
        p.starless_enabled = sl.enabled
        p.starless_strength = sl.strength
        p.starless_format = sl.format
        p.save_star_mask = sl.save_star_mask
        fs = self._project.frame_selection
        p.frame_selection_enabled = fs.enabled
        p.frame_selection_min_score = fs.min_score
        p.frame_selection_max_rejected_fraction = fs.max_rejected_fraction
        reg = self._project.registration
        p.registration_upsample_factor = reg.upsample_factor
        p.registration_reference_frame_index = reg.reference_frame_index
        br = self._project.background_removal
        p.background_removal_enabled = br.enabled
        p.background_removal_tile_size = br.tile_size
        p.background_removal_method = br.method
        p.background_removal_preserve_median = br.preserve_median
        dn = self._project.denoise
        p.denoise_strength = dn.strength
        p.denoise_tile_size = dn.tile_size
        p.denoise_tile_overlap = dn.tile_overlap
        st = self._project.stretch
        p.stretch_target_background = st.target_background
        p.stretch_shadow_clipping_sigmas = st.shadow_clipping_sigmas
        p.stretch_linked_channels = st.linked_channels
        sp = self._project.star_processing
        p.star_reduce_enabled = sp.reduce_enabled
        p.star_reduce_factor = sp.reduce_factor
        p.star_detection_sigma = sp.detection_sigma
        p.star_min_area = sp.min_area
        p.star_max_area = sp.max_area
        p.star_mask_dilation = sp.mask_dilation
        sk = self._project.stacking
        p.stacking_method = sk.method
        p.stacking_sigma_low = sk.sigma_low
        p.stacking_sigma_high = sk.sigma_high
        cv = self._project.curves
        p.curves_enabled = cv.enabled
        p.curves_rgb_points = [tuple(pt) for pt in cv.rgb_points]  # type: ignore[misc]
        p.curves_r_points = [tuple(pt) for pt in cv.r_points]  # type: ignore[misc]
        p.curves_g_points = [tuple(pt) for pt in cv.g_points]  # type: ignore[misc]
        p.curves_b_points = [tuple(pt) for pt in cv.b_points]  # type: ignore[misc]
        an = self._project.annotation
        p.annotation_show_dso = an.show_dso
        p.annotation_show_stars = an.show_stars
        p.annotation_show_boundaries = an.show_boundaries
        p.annotation_show_grid = an.show_grid
        p.output_path = self._project.output_path
        p.output_format = self._project.output_format
        self._frame_list_panel.refresh(self._project.input_frames)

    @Slot(int, bool)
    def _on_frame_selection_changed(self, idx: int, selected: bool) -> None:
        # Panel mutates FrameEntry in-place; this is belt-and-suspenders
        if 0 <= idx < len(self._project.input_frames):
            self._project.input_frames[idx].selected = selected
        self._project.touch()
        self._update_title()
        if not self._pipeline_worker.is_running:
            has_selected = any(e.selected for e in self._project.input_frames)
            self._stack_run_act.setEnabled(has_selected)

    @Slot(list)
    def _on_frames_remove_requested(self, indices: list[int]) -> None:
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self._project.input_frames):
                del self._project.input_frames[idx]
        self._project.touch()
        self._frame_list_panel.refresh(self._project.input_frames)
        self._update_title()
        if not self._pipeline_worker.is_running:
            has_selected = any(e.selected for e in self._project.input_frames)
            self._stack_run_act.setEnabled(has_selected)

    @Slot()
    def _sync_annotation_from_model(self) -> None:
        p = self._pipeline
        self._annotation_panel._dso_cb.blockSignals(True)
        self._annotation_panel._dso_cb.setChecked(p.annotation_show_dso)
        self._annotation_panel._dso_cb.blockSignals(False)

        self._annotation_panel._stars_cb.blockSignals(True)
        self._annotation_panel._stars_cb.setChecked(p.annotation_show_stars)
        self._annotation_panel._stars_cb.blockSignals(False)

        self._annotation_panel._boundaries_cb.blockSignals(True)
        self._annotation_panel._boundaries_cb.setChecked(p.annotation_show_boundaries)
        self._annotation_panel._boundaries_cb.blockSignals(False)

        self._annotation_panel._grid_cb.blockSignals(True)
        self._annotation_panel._grid_cb.setChecked(p.annotation_show_grid)
        self._annotation_panel._grid_cb.blockSignals(False)

        self._annotation_overlay.set_show_dso(p.annotation_show_dso)
        self._annotation_overlay.set_show_stars(p.annotation_show_stars)
        self._annotation_overlay.set_show_boundaries(p.annotation_show_boundaries)
        self._annotation_overlay.set_show_grid(p.annotation_show_grid)

    def _update_title(self) -> None:
        name = self._project.metadata.name
        if self._project_path:
            name = self._project_path.stem
        self.setWindowModified(self._project.is_dirty)
        self.setWindowTitle(f"{name}[*] - AstroAI Suite")

    def _setup_presets(self) -> None:
        from astroai.core.pipeline.builtin_presets import install_builtin_presets
        install_builtin_presets(self._preset_manager)

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

    def _rebuild_preset_menu(self) -> None:
        from astroai.core.pipeline.builtin_presets import BUILTIN_PRESET_NAMES

        self._preset_menu.clear()
        names = self._preset_manager.list_names()
        if not names:
            no_act = self._preset_menu.addAction("(keine Presets)")
            no_act.setEnabled(False)
            return
        builtin_names = [n for n in names if n in BUILTIN_PRESET_NAMES]
        user_names = [n for n in names if n not in BUILTIN_PRESET_NAMES]
        for name in builtin_names:
            act = self._preset_menu.addAction(name)
            act.triggered.connect(lambda checked=False, n=name: self._on_load_preset(n))
        if builtin_names and user_names:
            self._preset_menu.addSeparator()
        for name in user_names:
            act = self._preset_menu.addAction(name)
            act.triggered.connect(lambda checked=False, n=name: self._on_load_preset(n))

    @Slot()
    def _on_save_preset(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Preset speichern", "Preset-Name:")
        if not ok or not name.strip():
            return
        preset = self._preset_manager.capture_from_model(name.strip(), self._pipeline)
        self._preset_manager.save(preset)
        self._rebuild_preset_menu()
        self._status_bar.showMessage(f"Preset gespeichert: {name.strip()!r}")

    def _on_load_preset(self, name: str) -> None:
        try:
            preset = self._preset_manager.load(name)
        except FileNotFoundError:
            QMessageBox.warning(self, "Preset laden", f"Preset nicht gefunden: {name!r}")
            self._rebuild_preset_menu()
            return
        self._preset_manager.apply_to_model(preset, self._pipeline)
        self._status_bar.showMessage(f"Preset geladen: {name!r}")

    @Slot(str)
    def _on_session_notes_changed(self, text: str) -> None:
        self._project.metadata.description = text
        self._project.touch()
        self._update_title()

    @Slot()
    def _on_manage_license(self) -> None:
        dlg = ActivationDialog(self._license, self)
        dlg.exec()

    @Slot()
    def _on_show_project_summary(self) -> None:
        from astroai.project.summary import compute_summary

        s = compute_summary(self._project)
        name = self._project.metadata.name
        lines = [
            f"<b>Projekt:</b> {name}",
            f"<b>Frames:</b> {s.selected_count} ausgewählt / {s.total_frames} gesamt",
            f"<b>Gesamtbelichtung:</b> {s.total_exposure_hms}",
        ]
        if s.exposure_groups:
            groups_str = ", ".join(f"{g.exposure_s:.0f}s×{g.count}" for g in s.exposure_groups)
            lines.append(f"<b>Belichtungsgruppen:</b> {groups_str}")
        if s.quality_mean is not None:
            lines.append(
                f"<b>Qualität:</b> Ø {s.quality_mean:.1%} "
                f"(min {s.quality_min:.1%} / max {s.quality_max:.1%})"
            )
        if s.temp_min is not None:
            lines.append(f"<b>Temperatur:</b> {s.temp_min:.1f}°C … {s.temp_max:.1f}°C")
        QMessageBox.information(self, "Projektübersicht", "<br>".join(lines))

    @Slot()
    def _on_show_shortcuts(self) -> None:
        dlg = ShortcutsDialog(self)
        dlg.show()

    @Slot()
    def _on_show_model_manager(self) -> None:
        self._model_manager_dock.setVisible(True)
        self._model_manager_dock.raise_()
        self._model_manager.refresh()

    @Slot(str)
    def _on_download_model_requested(self, model_name: str) -> None:
        self._model_manager_dock.setVisible(True)
        self._model_manager_dock.raise_()
        self._model_manager.start_download(model_name)

    @Slot(str)
    def _on_status_message(self, message: str) -> None:
        self._status_bar.showMessage(message, 5000)

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

        self._wcs_adapter = adapter
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
            self._image_stats.set_image_data(preview)
            mode_labels = {"stars": "Sterne", "comet": "Kometenkopf", "blend": "Blend"}
            label = mode_labels.get(self._pipeline.comet_tracking_mode, "")
            self._status_bar.showMessage(f"Komet-Vorschau: {label}")

    @Slot()
    def _on_fit_to_view(self) -> None:
        if self._view_stack.currentIndex() == 1:
            self._compare_view.fit_to_view()
        else:
            self._viewer.fit_to_view()

    @Slot(bool)
    def _on_toggle_auto_stretch(self, checked: bool) -> None:
        self._auto_stretch = checked
        if isinstance(self._current_image, np.ndarray):
            self._display_image_data(self._current_image)

    @Slot()
    def _on_copy_image(self) -> None:
        from PySide6.QtWidgets import QApplication
        qimg = self._viewer.render_full_qimage()
        if qimg is not None:
            QApplication.clipboard().setImage(qimg)
            self._status_bar.showMessage("Bild in Zwischenablage kopiert")

    @Slot()
    def _on_save_preview_image(self) -> None:
        qimg = self._viewer.render_full_qimage()
        if qimg is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Vorschau-Bild speichern",
            "",
            "PNG-Bild (*.png);;JPEG-Bild (*.jpg *.jpeg);;TIFF-Bild (*.tiff *.tif)",
        )
        if not path:
            return
        if qimg.save(path):
            self._status_bar.showMessage(f"Vorschau gespeichert: {Path(path).name}")
        else:
            QMessageBox.warning(self, "Speichern fehlgeschlagen", f"Datei konnte nicht gespeichert werden:\n{path}")

    @Slot(bool)
    def _on_toggle_compare(self, checked: bool) -> None:
        self._view_stack.setCurrentIndex(1 if checked else 0)

    def _reset_compare_state(self) -> None:
        self._before_image = None
        self._compare_view.clear()
        self._compare_act.setChecked(False)
        self._compare_act.setEnabled(False)
        self._view_stack.setCurrentIndex(0)

    @Slot()
    def _on_run_pipeline(self) -> None:
        if self._pipeline_worker.is_running:
            return
        if not isinstance(self._current_image, np.ndarray):
            return

        self._before_image = self._current_image
        if self._processing_base_image is None:
            self._processing_base_image = self._current_image
        self._processing_history.push(
            "Processing",
            self._pipeline.snapshot_processing_params(),
        )
        pipeline = self._pipeline_builder.build_processing_pipeline(self._pipeline)
        context = PipelineContext(images=[self._current_image])
        self._pipeline.reset()
        self._run_act.setEnabled(False)
        self._stack_run_act.setEnabled(False)
        self._cancel_act.setEnabled(True)
        self._progress.set_indeterminate()
        self._progress.set_status("Pipeline läuft…")
        self._pipeline_worker.start(pipeline, context)

    def _validate_before_run(self) -> bool:
        """Return True if the pipeline may start; show a warning dialog on errors."""
        from astroai.project.validator import validate_project

        result = validate_project(self._project)
        if not result.has_errors:
            return True
        lines = "\n".join(f"• {i.message}" for i in result.errors)
        QMessageBox.warning(self, "Projekt-Validierung fehlgeschlagen", lines)
        return False

    @Slot()
    def _on_run_full_pipeline(self) -> None:
        if self._pipeline_worker.is_running:
            return
        if not self._validate_before_run():
            return
        frame_paths = [
            Path(e.path)
            for e in self._project.input_frames
            if e.selected and Path(e.path).exists()
        ]
        if not frame_paths:
            self._status_bar.showMessage(
                "Keine ausgewählten Light-Frames — bitte Frames auswählen oder importieren"
            )
            return

        pipeline = self._pipeline_builder.build_full_pipeline(self._pipeline, frame_paths)
        if self._pipeline.output_path:
            export_step = self._pipeline_builder.build_export_step(
                self._pipeline,
                Path(self._pipeline.output_path),
                self._pipeline.output_filename,
            )
            pipeline._steps.append(export_step)
        context = PipelineContext()
        self._pipeline.reset()
        self._run_act.setEnabled(False)
        self._stack_run_act.setEnabled(False)
        self._cancel_act.setEnabled(True)
        self._progress.set_indeterminate()
        self._progress.set_status(f"Stack & Process: {len(frame_paths)} Frames…")
        self._pipeline_worker.start(pipeline, context)

    @Slot(object)
    def _on_pipeline_finished(self, context: object) -> None:
        from astroai.core.pipeline.base import PipelineContext as _PCtx

        assert isinstance(context, _PCtx)
        result = context.result
        if isinstance(result, np.ndarray):
            self._current_image = result
            self._display_image_data(result)
            if isinstance(self._before_image, np.ndarray):
                self._compare_view.set_before(self._before_image)
                self._compare_view.set_after(result)
                self._compare_act.setEnabled(True)
                self._compare_act.setChecked(True)
                self._view_stack.setCurrentIndex(1)

        # Write frame quality scores back to the project when available
        frame_scores: list[float] = context.metadata.get("frame_scores", [])
        rejected_indices: list[int] = context.metadata.get("frame_selection_rejected", [])
        if frame_scores and self._project.input_frames:
            for i, score in enumerate(frame_scores):
                if i < len(self._project.input_frames):
                    self._project.input_frames[i].quality_score = score
            self._frame_list_panel.refresh(self._project.input_frames)
            self._frame_quality_dashboard.set_scores(frame_scores, rejected_indices)
        elif not frame_scores:
            self._frame_quality_dashboard.clear()

        from astroai.ui.models import StepState
        active = self._pipeline.active_step()
        if active:
            self._pipeline.set_step_state(active.key, StepState.DONE)

        self._progress.reset()
        self._cancel_act.setEnabled(False)
        self._run_act.setEnabled(True)
        self._stack_run_act.setEnabled(bool(self._project.input_frames))
        self._status_bar.showMessage("Pipeline abgeschlossen")

    @Slot(float, str)
    def _on_pipeline_progress(self, fraction: float, message: str) -> None:
        self._progress.set_determinate()
        self._progress.set_progress(fraction)
        self._progress.set_status(message)

    @Slot(str)
    def _on_pipeline_stage_active(self, stage_name: str) -> None:
        step_key = _STAGE_TO_STEP_KEY.get(stage_name)
        if step_key:
            self._pipeline.advance_to(step_key)

    @Slot()
    def _on_cancel_pipeline(self) -> None:
        self._pipeline_worker.cancel()
        self._cancel_act.setEnabled(False)
        self._status_bar.showMessage("Abbruch angefordert…")

    @Slot()
    def _on_pipeline_cancelled(self) -> None:
        self._progress.reset()
        self._cancel_act.setEnabled(False)
        self._run_act.setEnabled(True)
        self._stack_run_act.setEnabled(bool(self._project.input_frames))
        self._status_bar.showMessage("Pipeline abgebrochen")

    @Slot(str)
    def _on_pipeline_error(self, msg: str) -> None:
        import logging

        from astroai.ui.models import StepState
        active = self._pipeline.active_step()
        if active:
            self._pipeline.set_step_state(active.key, StepState.ERROR)
        self._progress.reset()
        self._cancel_act.setEnabled(False)
        self._run_act.setEnabled(True)
        self._stack_run_act.setEnabled(bool(self._project.input_frames))
        self._status_bar.showMessage(f"Pipeline-Fehler: {msg}")
        logging.getLogger("astroai.pipeline").error("Pipeline-Fehler: %s", msg)

    @Slot(object)
    def _on_calibration_finished(self, _results: object) -> None:
        self._benchmark.finish()

    @Slot(str)
    def _on_calibration_error(self, msg: str) -> None:
        import logging

        self._benchmark.reset()
        logging.getLogger("astroai.pipeline").error("Kalibrierung fehlgeschlagen: %s", msg)

    # -- undo / redo handlers --------------------------------------------------

    @Slot()
    def _on_history_changed(self) -> None:
        can_undo = self._processing_history.can_undo
        can_redo = self._processing_history.can_redo
        undo_name = self._processing_history.undo_step_name
        redo_name = self._processing_history.redo_step_name
        self._undo_act.setEnabled(can_undo)
        self._redo_act.setEnabled(can_redo)
        self._undo_act.setText(
            f"Rückgängig: {undo_name}" if undo_name else "Rückgängig"
        )
        self._redo_act.setText(
            f"Wiederherstellen: {redo_name}" if redo_name else "Wiederherstellen"
        )

    @Slot()
    def _on_undo(self) -> None:
        if self._pipeline_worker.is_running:
            return
        entry = self._processing_history.undo()
        if entry is None:
            return
        prev = self._processing_history.peek_undo()
        if prev is not None:
            self._pipeline.restore_processing_params(prev.params)
            self._before_image = self._processing_base_image
            if isinstance(self._processing_base_image, np.ndarray):
                pipeline = self._pipeline_builder.build_processing_pipeline(self._pipeline)
                context = PipelineContext(images=[self._processing_base_image])
                self._pipeline.reset()
                self._pipeline_worker.start(pipeline, context)
        elif isinstance(self._processing_base_image, np.ndarray):
            self._current_image = self._processing_base_image
            self._display_image_data(self._processing_base_image)
            self._status_bar.showMessage("Rückgängig — Ausgangsbild wiederhergestellt")

    @Slot()
    def _on_redo(self) -> None:
        if self._pipeline_worker.is_running:
            return
        entry = self._processing_history.redo()
        if entry is None:
            return
        self._pipeline.restore_processing_params(entry.params)
        if isinstance(self._processing_base_image, np.ndarray):
            self._before_image = self._processing_base_image
            pipeline = self._pipeline_builder.build_processing_pipeline(self._pipeline)
            context = PipelineContext(images=[self._processing_base_image])
            self._pipeline.reset()
            self._pipeline_worker.start(pipeline, context)

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


if __name__ == "__main__":  # pragma: no cover
    main()
