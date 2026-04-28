"""Compute aggregate statistics for an AstroProject."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExposureGroup:
    exposure_s: float
    count: int


@dataclass
class ProjectSummary:
    total_frames: int = 0
    selected_count: int = 0

    # Exposure totals (only frames with exposure metadata)
    total_exposure_s: float = 0.0
    exposure_groups: list[ExposureGroup] = field(default_factory=list)

    # Quality score stats (only scored frames)
    scored_count: int = 0
    quality_mean: float | None = None
    quality_min: float | None = None
    quality_max: float | None = None

    # Temperature stats (only frames with temperature metadata)
    temp_min: float | None = None
    temp_max: float | None = None

    @property
    def unselected_count(self) -> int:
        return self.total_frames - self.selected_count

    @property
    def total_exposure_hms(self) -> str:
        """Human-readable total exposure, e.g. ``1h 23m 45s``."""
        secs = int(self.total_exposure_s)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        parts = []
        if h:
            parts.append(f"{h}h")
        if m or h:
            parts.append(f"{m}m")
        parts.append(f"{s}s")
        return " ".join(parts) if parts else "0s"


def compute_summary(project: Any) -> ProjectSummary:
    """Compute statistics for *project* (an :class:`AstroProject` instance)."""
    from astroai.project.project_file import FrameEntry

    frames = getattr(project, "input_frames", [])
    summary = ProjectSummary(total_frames=len(frames))

    selected = [f for f in frames if isinstance(f, FrameEntry) and f.selected]
    summary.selected_count = len(selected)

    # Exposure totals and groups
    exp_buckets: dict[float, int] = {}
    for f in selected:
        if f.exposure is not None:
            summary.total_exposure_s += f.exposure
            bucket = round(f.exposure, 2)
            exp_buckets[bucket] = exp_buckets.get(bucket, 0) + 1
    summary.exposure_groups = [
        ExposureGroup(exposure_s=k, count=v)
        for k, v in sorted(exp_buckets.items())
    ]

    # Quality stats
    scores = [f.quality_score for f in selected if f.quality_score is not None]
    if scores:
        summary.scored_count = len(scores)
        summary.quality_mean = sum(scores) / len(scores)
        summary.quality_min = min(scores)
        summary.quality_max = max(scores)

    # Temperature stats
    temps = [f.temperature for f in selected if f.temperature is not None]
    if temps:
        summary.temp_min = min(temps)
        summary.temp_max = max(temps)

    return summary
