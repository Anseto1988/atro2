"""Annotation overlay for astrometric sky object visualization."""

from astroai.ui.overlay.sky_objects import (
    CatalogObject,
    ConstellationBoundarySegment,
    NamedStar,
    SkyObjectCatalog,
)
from astroai.ui.overlay.annotation_overlay import AnnotationOverlay

__all__ = [
    "AnnotationOverlay",
    "CatalogObject",
    "ConstellationBoundarySegment",
    "NamedStar",
    "SkyObjectCatalog",
]
