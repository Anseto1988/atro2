"""Project persistence – save/load AstroAI processing sessions."""
from __future__ import annotations

from astroai.project.project_file import (
    AstroProject,
    CalibrationConfig,
    DenoiseConfig,
    FrameEntry,
    ProjectMetadata,
    RegistrationConfig,
    StackingConfig,
    StarProcessingConfig,
    StretchConfig,
)
from astroai.project.serializer import ProjectSerializer
from astroai.project.summary import ExposureGroup, ProjectSummary, compute_summary
from astroai.project.validator import ValidationIssue, ValidationResult, validate_project

__all__ = [
    "AstroProject",
    "CalibrationConfig",
    "DenoiseConfig",
    "FrameEntry",
    "ProjectMetadata",
    "ProjectSerializer",
    "RegistrationConfig",
    "StackingConfig",
    "StarProcessingConfig",
    "StretchConfig",
    "ValidationIssue",
    "ValidationResult",
    "validate_project",
    "ExposureGroup",
    "ProjectSummary",
    "compute_summary",
]
