from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from astroai.core.io.fits_io import ImageMetadata

if TYPE_CHECKING:
    from astroai.project.project_file import CalibrationConfig


@dataclass(frozen=True)
class CalibrationFrame:
    path: Path
    metadata: ImageMetadata
    data: NDArray[np.floating[Any]] | None = None


@dataclass
class CalibrationLibrary:
    darks: list[CalibrationFrame]
    flats: list[CalibrationFrame]
    bias: list[CalibrationFrame]

    @staticmethod
    def empty() -> CalibrationLibrary:
        return CalibrationLibrary(darks=[], flats=[], bias=[])

    @staticmethod
    def from_config(config: object, load_data: bool = False) -> CalibrationLibrary:
        """Build a CalibrationLibrary from a CalibrationConfig by loading FITS files."""
        from astroai.core.io.fits_io import read_fits

        def _load_frames(paths: list[str]) -> list[CalibrationFrame]:
            frames: list[CalibrationFrame] = []
            for p in paths:
                path = Path(p)
                if not path.exists():
                    continue
                try:
                    data, meta = read_fits(path)
                    frames.append(CalibrationFrame(path, meta, data if load_data else None))
                except Exception:
                    pass
            return frames

        return CalibrationLibrary(
            darks=_load_frames(getattr(config, "dark_frames", [])),
            flats=_load_frames(getattr(config, "flat_frames", [])),
            bias=_load_frames(getattr(config, "bias_frames", [])),
        )


def _match_score(light: ImageMetadata, cal: ImageMetadata) -> float:
    score = 0.0

    if light.exposure is not None and cal.exposure is not None:
        if abs(light.exposure - cal.exposure) < 0.01:
            score += 10.0
        else:
            score -= abs(light.exposure - cal.exposure)

    if light.gain_iso is not None and cal.gain_iso is not None:
        if light.gain_iso == cal.gain_iso:
            score += 5.0
        else:
            score -= 2.0

    if light.temperature is not None and cal.temperature is not None:
        delta = abs(light.temperature - cal.temperature)
        if delta < 2.0:
            score += 3.0
        else:
            score -= delta * 0.5

    if light.width > 0 and cal.width > 0:
        if light.width == cal.width and light.height == cal.height:
            score += 2.0
        else:
            return float("-inf")

    return score


def find_best_dark(
    light_meta: ImageMetadata,
    library: CalibrationLibrary,
) -> CalibrationFrame | None:
    if not library.darks:
        return None
    scored = [(frame, _match_score(light_meta, frame.metadata)) for frame in library.darks]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_frame, best_score = scored[0]
    return best_frame if best_score > 0 else None


def find_best_flat(
    light_meta: ImageMetadata,
    library: CalibrationLibrary,
) -> CalibrationFrame | None:
    if not library.flats:
        return None
    scored = [(frame, _match_score(light_meta, frame.metadata)) for frame in library.flats]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_frame, best_score = scored[0]
    return best_frame if best_score > 0 else None


@dataclass
class FrameMatchResult:
    """Calibration match result for a single light frame."""

    light_path: Path
    dark: CalibrationFrame | None
    flat: CalibrationFrame | None
    bias: CalibrationFrame | None


@dataclass
class BatchMatchResult:
    """Aggregate match results for all light frames."""

    matches: list[FrameMatchResult]

    @property
    def coverage(self) -> float:
        """Fraction of frames that received at least one calibration frame."""
        if not self.matches:
            return 0.0
        matched = sum(1 for m in self.matches if m.dark or m.flat or m.bias)
        return matched / len(self.matches)

    @property
    def dark_coverage(self) -> float:
        if not self.matches:
            return 0.0
        return sum(1 for m in self.matches if m.dark) / len(self.matches)

    @property
    def flat_coverage(self) -> float:
        if not self.matches:
            return 0.0
        return sum(1 for m in self.matches if m.flat) / len(self.matches)


def batch_match(
    lights: list[tuple[Path, ImageMetadata]],
    library: CalibrationLibrary,
) -> BatchMatchResult:
    """Match every light frame in *lights* to its best dark/flat from *library*."""
    matches = [
        FrameMatchResult(
            light_path=path,
            dark=find_best_dark(meta, library),
            flat=find_best_flat(meta, library),
            bias=None,
        )
        for path, meta in lights
    ]
    return BatchMatchResult(matches=matches)


def suggest_calibration_config(result: BatchMatchResult) -> "CalibrationConfig":
    """Collapse a BatchMatchResult into a deduplicated CalibrationConfig.

    Returns a CalibrationConfig with unique paths from all matched frames,
    ready to assign to AstroProject.calibration.
    """
    from astroai.project.project_file import CalibrationConfig

    def _unique(frames: list[CalibrationFrame | None]) -> list[str]:
        seen: dict[str, None] = {}
        for f in frames:
            if f is not None:
                seen[str(f.path)] = None
        return list(seen)

    darks = _unique([m.dark for m in result.matches])
    flats = _unique([m.flat for m in result.matches])
    bias = _unique([m.bias for m in result.matches])
    return CalibrationConfig(dark_frames=darks, flat_frames=flats, bias_frames=bias)
