"""Custom AstroAI widgets."""

from astroai.ui.widgets.calibration_benchmark import CalibrationBenchmarkWidget
from astroai.ui.widgets.activation_dialog import ActivationDialog
from astroai.ui.widgets.annotation_panel import AnnotationPanel
from astroai.ui.widgets.channel_panel import ChannelCombinerPanel
from astroai.ui.widgets.comet_stack_panel import CometStackPanel
from astroai.ui.widgets.deconvolution_panel import DeconvolutionPanel
from astroai.ui.widgets.histogram_widget import HistogramWidget
from astroai.ui.widgets.live_histogram_view import HistogramView, HistogramWorker
from astroai.ui.widgets.image_viewer import ImageViewer
from astroai.ui.widgets.license_badge import LicenseBadge
from astroai.ui.widgets.log_widget import LogWidget
from astroai.ui.widgets.mosaic_panel import MosaicPanel
from astroai.ui.widgets.offline_banner import OfflineBanner
from astroai.ui.widgets.photometry_panel import PhotometryPanel
from astroai.ui.widgets.progress_widget import ProgressWidget
from astroai.ui.widgets.fwhm_overlay import FWHMOverlay
from astroai.ui.widgets.star_analysis_panel import StarAnalysisPanel
from astroai.ui.widgets.starless_panel import StarlessPanel
from astroai.ui.widgets.upgrade_dialog import UpgradeDialog
from astroai.ui.widgets.pipeline_timeline_widget import PipelineTimelineWidget
from astroai.ui.widgets.workflow_graph import WorkflowGraph
from astroai.ui.widgets.channel_balance_panel import ChannelBalancePanel
from astroai.ui.widgets.frame_quality_dashboard import FrameQualityDashboard
from astroai.ui.widgets.model_manager_panel import ModelManagerPanel

__all__ = [
    "CalibrationBenchmarkWidget",
    "ActivationDialog",
    "AnnotationPanel",
    "ChannelCombinerPanel",
    "CometStackPanel",
    "DeconvolutionPanel",
    "HistogramWidget",
    "HistogramView",
    "HistogramWorker",
    "ImageViewer",
    "LicenseBadge",
    "LogWidget",
    "MosaicPanel",
    "OfflineBanner",
    "PhotometryPanel",
    "FWHMOverlay",
    "PipelineTimelineWidget",
    "ProgressWidget",
    "StarAnalysisPanel",
    "StarlessPanel",
    "UpgradeDialog",
    "WorkflowGraph",
    "ChannelBalancePanel",
    "FrameQualityDashboard",
    "ModelManagerPanel",
]
