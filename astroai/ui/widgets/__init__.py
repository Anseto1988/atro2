"""Custom AstroAI widgets."""

from astroai.ui.widgets.activation_dialog import ActivationDialog
from astroai.ui.widgets.annotation_panel import AnnotationPanel
from astroai.ui.widgets.channel_panel import ChannelCombinerPanel
from astroai.ui.widgets.deconvolution_panel import DeconvolutionPanel
from astroai.ui.widgets.histogram_widget import HistogramWidget
from astroai.ui.widgets.image_viewer import ImageViewer
from astroai.ui.widgets.license_badge import LicenseBadge
from astroai.ui.widgets.log_widget import LogWidget
from astroai.ui.widgets.offline_banner import OfflineBanner
from astroai.ui.widgets.progress_widget import ProgressWidget
from astroai.ui.widgets.starless_panel import StarlessPanel
from astroai.ui.widgets.upgrade_dialog import UpgradeDialog
from astroai.ui.widgets.workflow_graph import WorkflowGraph

__all__ = [
    "ActivationDialog",
    "AnnotationPanel",
    "ChannelCombinerPanel",
    "DeconvolutionPanel",
    "HistogramWidget",
    "ImageViewer",
    "LicenseBadge",
    "LogWidget",
    "OfflineBanner",
    "ProgressWidget",
    "StarlessPanel",
    "UpgradeDialog",
    "WorkflowGraph",
]
