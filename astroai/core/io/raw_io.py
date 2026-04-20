from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
from numpy.typing import NDArray

from astroai.core.io.fits_io import ImageMetadata

RAW_EXTENSIONS = frozenset({".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"})


def _extract_raw_metadata(raw: rawpy.RawPy, path: Path) -> ImageMetadata:
    sizes = raw.sizes
    height = sizes.height
    width = sizes.width
    channels = raw.num_colors

    exposure: float | None = None
    gain_iso: int | None = None
    date_obs: str | None = None
    temperature: float | None = None

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        with Image.open(str(path)) as img:
            exif = img.getexif()
            if exif:
                tag_map = {v: k for k, v in TAGS.items()}
                if tag_map.get("ExposureTime") in exif:
                    val = exif[tag_map["ExposureTime"]]
                    exposure = float(val)
                if tag_map.get("ISOSpeedRatings") in exif:
                    gain_iso = int(exif[tag_map["ISOSpeedRatings"]])
                if tag_map.get("DateTimeOriginal") in exif:
                    date_obs = str(exif[tag_map["DateTimeOriginal"]])
    except Exception:
        pass

    return ImageMetadata(
        exposure=exposure,
        gain_iso=gain_iso,
        temperature=temperature,
        date_obs=date_obs,
        width=width,
        height=height,
        channels=channels,
    )


def read_raw(path: str | Path) -> tuple[NDArray[np.floating[Any]], ImageMetadata]:
    path = Path(path)
    with rawpy.imread(str(path)) as raw:
        metadata = _extract_raw_metadata(raw, path)
        rgb: NDArray[np.floating[Any]] = raw.postprocess(
            output_bps=16,
            use_camera_wb=True,
            no_auto_bright=True,
        ).astype(np.float32) / 65535.0
    return rgb, metadata
