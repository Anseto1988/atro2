"""Validate an AstroProject before pipeline execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning" | "info"
    code: str   # machine-readable identifier
    message: str
    detail: str = ""


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def summary(self) -> str:
        if not self.issues:
            return "Projekt OK"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} Fehler")
        if self.warnings:
            parts.append(f"{len(self.warnings)} Warnung(en)")
        return ", ".join(parts)


def validate_project(project: object) -> ValidationResult:
    """Run all checks against *project* (an AstroProject instance).

    Returns a :class:`ValidationResult` with zero or more issues.
    Level semantics:
    - ``"error"`` — pipeline cannot run safely
    - ``"warning"`` — pipeline may run but result could be poor
    - ``"info"`` — informational, no action required
    """
    result = ValidationResult()

    # Lazy import to avoid circular imports
    from astroai.project.project_file import AstroProject, FrameEntry

    if not isinstance(project, AstroProject):
        result.issues.append(
            ValidationIssue("error", "INVALID_PROJECT", "Ungültiges Projektobjekt")
        )
        return result

    # ---- Light frames ----
    frames = project.input_frames
    if not frames:
        result.issues.append(
            ValidationIssue("error", "NO_FRAMES", "Keine Light-Frames im Projekt")
        )
    else:
        selected = [f for f in frames if isinstance(f, FrameEntry) and f.selected]
        if not selected:
            result.issues.append(
                ValidationIssue(
                    "error",
                    "NO_SELECTED_FRAMES",
                    "Alle Light-Frames deselektiert — mindestens einen Frame auswählen",
                )
            )
        else:
            missing = [f for f in selected if not Path(f.path).exists()]
            if missing:
                names = ", ".join(Path(f.path).name for f in missing[:3])
                suffix = f" (+{len(missing) - 3} weitere)" if len(missing) > 3 else ""
                result.issues.append(
                    ValidationIssue(
                        "warning",
                        "MISSING_LIGHT_FILES",
                        f"{len(missing)} Light-Frame(s) nicht gefunden",
                        detail=names + suffix,
                    )
                )

    # ---- Calibration frames ----
    cal = project.calibration
    for label, paths in (
        ("Dark", cal.dark_frames),
        ("Flat", cal.flat_frames),
        ("Bias", cal.bias_frames),
    ):
        missing_cal = [p for p in paths if not Path(p).exists()]
        if missing_cal:
            result.issues.append(
                ValidationIssue(
                    "warning",
                    f"MISSING_{label.upper()}_FILES",
                    f"{len(missing_cal)} {label}-Frame(s) nicht mehr vorhanden",
                    detail=Path(missing_cal[0]).name,
                )
            )

    # ---- Output path (if export step is expected) ----
    output_path = getattr(project, "output_path", None)
    if output_path and not Path(output_path).parent.exists():
        result.issues.append(
            ValidationIssue(
                "warning",
                "OUTPUT_DIR_MISSING",
                f"Ausgabeverzeichnis existiert nicht: {Path(output_path).parent}",
            )
        )

    return result
