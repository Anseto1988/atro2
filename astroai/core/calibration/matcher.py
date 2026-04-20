from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroai.core.io.fits_io import ImageMetadata


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
