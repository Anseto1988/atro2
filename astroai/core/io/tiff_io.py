from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from astroai.core.io.fits_io import ImageMetadata


def write_tiff32(
    path: str | Path,
    data: NDArray[np.floating[Any]],
    metadata: ImageMetadata | None = None,
) -> Path:
    path = Path(path)
    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 3:
        channels, height, width = data.shape
        if channels == 1:
            img = Image.fromarray(data[0], mode="F")
        elif channels == 3:
            img = Image.merge("RGB", [
                Image.fromarray(data[c], mode="F") for c in range(3)
            ])
        else:
            raise ValueError(f"Unsupported channel count for TIFF: {channels}")
    elif data.ndim == 2:
        height, width = data.shape
        img = Image.fromarray(data, mode="F")
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

    tiff_tags: dict[int, Any] = {}
    if metadata:
        if metadata.date_obs is not None:
            tiff_tags[306] = metadata.date_obs  # DateTime tag

    img.save(str(path), format="TIFF", compression="none", tiffinfo=tiff_tags)
    return path


def read_tiff(path: str | Path) -> tuple[NDArray[np.floating[Any]], ImageMetadata]:
    path = Path(path)
    img = Image.open(str(path))

    if img.mode == "F":
        data = np.asarray(img, dtype=np.float32)
        channels = 1
    elif img.mode == "RGB":
        raw = np.asarray(img, dtype=np.float32)
        data = np.transpose(raw, (2, 0, 1))
        channels = 3
    elif img.mode in ("I", "I;16"):
        data = np.asarray(img, dtype=np.float32)
        channels = 1
    else:
        data = np.asarray(img, dtype=np.float32)
        if data.ndim == 3:
            data = np.transpose(data, (2, 0, 1))
            channels = data.shape[0]
        else:
            channels = 1

    if data.ndim == 2:
        height, width = data.shape
    else:
        _, height, width = data.shape

    date_obs = img.tag_v2.get(306) if hasattr(img, "tag_v2") else None

    return data, ImageMetadata(
        width=width,
        height=height,
        channels=channels,
        date_obs=str(date_obs) if date_obs else None,
    )
