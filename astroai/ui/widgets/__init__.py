"""Custom AstroAI widgets."""

from astroai.ui.widgets.activation_dialog import ActivationDialog
from astroai.ui.widgets.histogram_widget import HistogramWidget
from astroai.ui.widgets.image_viewer import ImageViewer
from astroai.ui.widgets.license_badge import LicenseBadge
from astroai.ui.widgets.offline_banner import OfflineBanner
from astroai.ui.widgets.progress_widget import ProgressWidget
from astroai.ui.widgets.upgrade_dialog import UpgradeDialog
from astroai.ui.widgets.workflow_graph import WorkflowGraph

__all__ = [
    "ActivationDialog",
    "HistogramWidget",
    "ImageViewer",
    "LicenseBadge",
    "OfflineBanner",
    "ProgressWidget",
    "UpgradeDialog",
    "WorkflowGraph",
]
