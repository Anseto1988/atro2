from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

from astroai.core.calibration.matcher import (
    CalibrationLibrary,
    find_best_dark,
    find_best_flat,
)
from astroai.core.io.fits_io import ImageMetadata

LoadDataFn = Callable[[Path], NDArray[np.floating[Any]]]


def apply_dark(
    light: NDArray[np.floating[Any]],
    dark: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    return cast(NDArray[np.floating[Any]], np.clip(light - dark, 0.0, None).astype(np.float32))


def apply_flat(
    light: NDArray[np.floating[Any]],
    flat: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    flat_norm = flat / np.maximum(np.median(flat), 1e-7)
    return cast(NDArray[np.floating[Any]], (light / np.maximum(flat_norm, 1e-7)).astype(np.float32))


def calibrate_frame(
    light: NDArray[np.floating[Any]],
    light_meta: ImageMetadata,
    library: CalibrationLibrary,
    load_data: LoadDataFn | None = None,
) -> NDArray[np.floating[Any]]:
    result = light.copy()

    dark_frame = find_best_dark(light_meta, library)
    if dark_frame is not None:
        dark_data = dark_frame.data
        if dark_data is None and load_data is not None:
            dark_data = load_data(dark_frame.path)
        if dark_data is not None:
            result = apply_dark(result, dark_data)

    flat_frame = find_best_flat(light_meta, library)
    if flat_frame is not None:
        flat_data = flat_frame.data
        if flat_data is None and load_data is not None:
            flat_data = load_data(flat_frame.path)
        if flat_data is not None:
            result = apply_flat(result, flat_data)

    return result
