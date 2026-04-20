"""Project data model for AstroAI pipeline persistence."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


PROJECT_FILE_VERSION = "1.0"
PROJECT_EXTENSION = ".astroai"


@dataclass
class ProjectMetadata:
    version: str = PROJECT_FILE_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    name: str = "Untitled"
    description: str = ""


@dataclass
class FrameEntry:
    path: str = ""
    exposure: float | None = None
    gain_iso: int | None = None
    temperature: float | None = None
    quality_score: float | None = None
    selected: bool = True


@dataclass
class CalibrationConfig:
    dark_frames: list[str] = field(default_factory=list)
    flat_frames: list[str] = field(default_factory=list)
    bias_frames: list[str] = field(default_factory=list)


@dataclass
class RegistrationConfig:
    enabled: bool = True
    reference_frame_index: int = 0
    upsample_factor: int = 10


@dataclass
class StackingConfig:
    method: str = "sigma_clip"
    sigma_low: float = 2.5
    sigma_high: float = 2.5


@dataclass
class StretchConfig:
    enabled: bool = True
    target_background: float = 0.25
    shadow_clipping_sigmas: float = -2.8
    linked_channels: bool = True


@dataclass
class DenoiseConfig:
    enabled: bool = True
    strength: float = 1.0
    tile_size: int = 512
    tile_overlap: int = 64
    model_type: str = "onnx"


@dataclass
class StarProcessingConfig:
    reduce_enabled: bool = False
    reduce_factor: float = 0.5
    detection_sigma: float = 4.0
    min_area: int = 3
    max_area: int = 5000
    mask_dilation: int = 3


@dataclass
class AstroProject:
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    input_frames: list[FrameEntry] = field(default_factory=list)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    stretch: StretchConfig = field(default_factory=StretchConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    star_processing: StarProcessingConfig = field(default_factory=StarProcessingConfig)
    output_path: str = ""
    output_format: str = "fits"

    def touch(self) -> None:
        self.metadata.modified_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AstroProject":
        return cls(
            metadata=ProjectMetadata(**data.get("metadata", {})),
            input_frames=[FrameEntry(**f) for f in data.get("input_frames", [])],
            calibration=CalibrationConfig(**data.get("calibration", {})),
            registration=RegistrationConfig(**data.get("registration", {})),
            stacking=StackingConfig(**data.get("stacking", {})),
            stretch=StretchConfig(**data.get("stretch", {})),
            denoise=DenoiseConfig(**data.get("denoise", {})),
            star_processing=StarProcessingConfig(**data.get("star_processing", {})),
            output_path=data.get("output_path", ""),
            output_format=data.get("output_format", "fits"),
        )
