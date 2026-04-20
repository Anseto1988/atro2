from astroai.core.io.fits_io import ImageMetadata, read_fits, write_fits
from astroai.core.io.xisf_io import read_xisf

__all__ = [
    "ImageMetadata",
    "read_fits",
    "read_raw",
    "read_xisf",
    "write_fits",
]


def __getattr__(name: str):
    if name == "read_raw":
        from astroai.core.io.raw_io import read_raw
        return read_raw
    if name == "RAW_EXTENSIONS":
        from astroai.core.io.raw_io import RAW_EXTENSIONS
        return RAW_EXTENSIONS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
