"""Scan a directory for FITS calibration frames and classify them by IMAGETYP."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from astroai.core.io.fits_io import ImageMetadata, read_fits

if TYPE_CHECKING:
    from astroai.core.calibration.matcher import CalibrationLibrary

_DARK_KEYWORDS = frozenset({"dark", "dark frame", "darkframe", "bias frame", "biasframe"})
_FLAT_KEYWORDS = frozenset({"flat", "flat field", "flatfield", "flat frame", "flatframe"})
_BIAS_KEYWORDS = frozenset({"bias", "offset", "zero"})
_LIGHT_KEYWORDS = frozenset({"light", "light frame", "lightframe", "science"})

_FITS_EXTENSIONS = frozenset({".fits", ".fit", ".fts"})


def _classify_imagetyp(imagetyp: str) -> str:
    normalized = imagetyp.strip().lower()
    if normalized in _DARK_KEYWORDS:
        return "dark"
    if normalized in _FLAT_KEYWORDS:
        return "flat"
    if normalized in _BIAS_KEYWORDS:
        return "bias"
    if normalized in _LIGHT_KEYWORDS:
        return "light"
    return "unknown"


@dataclass
class ScannedFrame:
    path: Path
    frame_type: str  # "dark", "flat", "bias", "light", "unknown"
    metadata: ImageMetadata


def scan_directory(
    directory: Path,
    *,
    recursive: bool = False,
) -> list[ScannedFrame]:
    """Scan *directory* for FITS files and classify each by its IMAGETYP header.

    Silently skips files that cannot be read. Returns an empty list when the
    directory does not exist.
    """
    if not directory.is_dir():
        return []

    glob = directory.rglob("*") if recursive else directory.iterdir()
    results: list[ScannedFrame] = []

    for path in sorted(glob):
        if path.suffix.lower() not in _FITS_EXTENSIONS:
            continue
        try:
            _data, meta = read_fits(path)
        except Exception:
            continue
        imagetyp = str(meta.extra.get("IMAGETYP", meta.extra.get("FRAME", ""))).strip()
        frame_type = _classify_imagetyp(imagetyp) if imagetyp else "unknown"
        results.append(ScannedFrame(path=path, frame_type=frame_type, metadata=meta))

    return results


def partition_by_type(frames: list[ScannedFrame]) -> dict[str, list[ScannedFrame]]:
    """Group *frames* by their ``frame_type`` into a dict keyed by type string."""
    groups: dict[str, list[ScannedFrame]] = {}
    for frame in frames:
        groups.setdefault(frame.frame_type, []).append(frame)
    return groups


def build_calibration_library(
    frames: list[ScannedFrame],
    load_data: bool = False,
) -> "CalibrationLibrary":
    """Build a :class:`CalibrationLibrary` from pre-scanned frames.

    Only ``dark``, ``flat``, and ``bias`` typed frames are included.
    When *load_data* is True, raw pixel data is also stored in each
    :class:`CalibrationFrame`.
    """
    from astroai.core.calibration.matcher import CalibrationFrame, CalibrationLibrary

    groups = partition_by_type(frames)

    def _to_calib(typed_frames: list[ScannedFrame]) -> list[CalibrationFrame]:
        result: list[CalibrationFrame] = []
        for sf in typed_frames:
            data = None
            if load_data:
                try:
                    raw, _ = read_fits(sf.path)
                    data = raw
                except Exception:
                    pass
            result.append(CalibrationFrame(path=sf.path, metadata=sf.metadata, data=data))
        return result

    return CalibrationLibrary(
        darks=_to_calib(groups.get("dark", [])),
        flats=_to_calib(groups.get("flat", [])),
        bias=_to_calib(groups.get("bias", [])),
    )
