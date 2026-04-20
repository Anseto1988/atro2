from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray


@dataclass(frozen=True)
class ImageMetadata:
    exposure: float | None = None
    gain_iso: int | None = None
    temperature: float | None = None
    date_obs: str | None = None
    width: int = 0
    height: int = 0
    channels: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


_FITS_KEY_MAP: dict[str, str] = {
    "EXPTIME": "exposure",
    "EXPOSURE": "exposure",
    "GAIN": "gain_iso",
    "ISO": "gain_iso",
    "CCD-TEMP": "temperature",
    "DATE-OBS": "date_obs",
}


def _extract_fits_metadata(header: fits.Header) -> ImageMetadata:
    vals: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key in header:
        mapped = _FITS_KEY_MAP.get(key.upper())
        if mapped and mapped not in vals:
            vals[mapped] = header[key]
        elif key and not key.startswith(("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "")):
            extra[key] = header[key]

    naxis = header.get("NAXIS", 0)
    if naxis >= 2:
        height = int(header["NAXIS2"])
        width = int(header["NAXIS1"])
        channels = int(header.get("NAXIS3", 1)) if naxis >= 3 else 1
    else:
        width = height = channels = 0

    exposure = vals.get("exposure")
    gain_iso = vals.get("gain_iso")
    temperature = vals.get("temperature")

    return ImageMetadata(
        exposure=float(exposure) if exposure is not None else None,
        gain_iso=int(gain_iso) if gain_iso is not None else None,
        temperature=float(temperature) if temperature is not None else None,
        date_obs=str(vals["date_obs"]) if vals.get("date_obs") else None,
        width=width,
        height=height,
        channels=channels,
        extra=extra,
    )


def read_fits(path: str | Path) -> tuple[NDArray[np.floating[Any]], ImageMetadata]:
    path = Path(path)
    with fits.open(str(path)) as hdul:
        hdu = hdul[0]
        data: NDArray[np.floating[Any]] = np.asarray(hdu.data, dtype=np.float32)
        metadata = _extract_fits_metadata(hdu.header)
    return data, metadata


def write_fits(
    path: str | Path,
    data: NDArray[np.floating[Any]],
    metadata: ImageMetadata | None = None,
    extra_headers: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    header = fits.Header()
    if metadata:
        if metadata.exposure is not None:
            header["EXPTIME"] = metadata.exposure
        if metadata.gain_iso is not None:
            header["GAIN"] = metadata.gain_iso
        if metadata.temperature is not None:
            header["CCD-TEMP"] = metadata.temperature
        if metadata.date_obs is not None:
            header["DATE-OBS"] = metadata.date_obs
    if extra_headers:
        for k, v in extra_headers.items():
            header[k] = v

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(str(path), overwrite=True)
    return path
