from typing import Any

from astroai.core.io.fits_io import ImageMetadata, read_fits, write_fits
from astroai.core.io.tiff_io import read_tiff, write_tiff32
from astroai.core.io.xisf_io import read_xisf, write_xisf

__all__ = [
    "ImageMetadata",
    "read_fits",
    "read_raw",
    "read_raw_metadata",
    "read_tiff",
    "read_xisf",
    "write_fits",
    "write_tiff32",
    "write_xisf",
]


def __getattr__(name: str) -> Any:
    if name == "read_raw":
        from astroai.core.io.raw_io import read_raw
        return read_raw
    if name == "read_raw_metadata":
        from astroai.core.io.raw_io import read_raw_metadata
        return read_raw_metadata
    if name == "RAW_EXTENSIONS":
        from astroai.core.io.raw_io import RAW_EXTENSIONS
        return RAW_EXTENSIONS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
