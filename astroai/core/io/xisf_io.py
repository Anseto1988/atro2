from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroai.core.io.fits_io import ImageMetadata

XISF_MAGIC = b"XISF0100"
_DTYPE_MAP: dict[str, np.dtype[Any]] = {
    "UInt8": np.dtype(np.uint8),
    "UInt16": np.dtype(np.uint16),
    "UInt32": np.dtype(np.uint32),
    "Float32": np.dtype(np.float32),
    "Float64": np.dtype(np.float64),
}


def _parse_xisf_metadata(root: ET.Element) -> ImageMetadata:
    ns = {"xisf": "http://www.pixinsight.com/xisf"}
    extra: dict[str, Any] = {}
    exposure: float | None = None
    gain_iso: int | None = None
    temperature: float | None = None
    date_obs: str | None = None

    for prop in root.iter("{http://www.pixinsight.com/xisf}Property"):
        pid = prop.get("id", "")
        value = prop.get("value", prop.text or "")
        if "Exposure" in pid:
            try:
                exposure = float(value)
            except (ValueError, TypeError):
                pass
        elif "ISO" in pid or "Gain" in pid:
            try:
                gain_iso = int(float(value))
            except (ValueError, TypeError):
                pass
        elif "Temperature" in pid:
            try:
                temperature = float(value)
            except (ValueError, TypeError):
                pass
        elif "Observation" in pid and "Time" in pid:
            date_obs = str(value)
        else:
            extra[pid] = value

    for fk in root.iter("{http://www.pixinsight.com/xisf}FITSKeyword"):
        name = fk.get("name", "")
        val = fk.get("value", "").strip().strip("'").strip()
        if name == "EXPTIME" and exposure is None:
            try:
                exposure = float(val)
            except (ValueError, TypeError):
                pass
        elif name in ("GAIN", "ISO") and gain_iso is None:
            try:
                gain_iso = int(float(val))
            except (ValueError, TypeError):
                pass
        elif name == "CCD-TEMP" and temperature is None:
            try:
                temperature = float(val)
            except (ValueError, TypeError):
                pass
        elif name == "DATE-OBS" and date_obs is None:
            date_obs = val

    return ImageMetadata(
        exposure=exposure,
        gain_iso=gain_iso,
        temperature=temperature,
        date_obs=date_obs,
        extra=extra,
    )


def read_xisf(path: str | Path) -> tuple[NDArray[np.floating[Any]], ImageMetadata]:
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != XISF_MAGIC:
            raise ValueError(f"Not a valid XISF file: {path}")
        header_len = struct.unpack("<I", f.read(4))[0]
        _reserved = f.read(4)
        header_xml = f.read(header_len).rstrip(b"\x00").decode("utf-8")
        root = ET.fromstring(header_xml)

        image_elem = root.find("{http://www.pixinsight.com/xisf}Image")
        if image_elem is None:
            raise ValueError(f"No Image element in XISF header: {path}")

        geometry = image_elem.get("geometry", "")
        parts = [int(x) for x in geometry.split(":")]
        if len(parts) == 3:
            width, height, channels = parts
        elif len(parts) == 2:
            width, height = parts
            channels = 1
        else:
            raise ValueError(f"Invalid geometry: {geometry}")

        sample_format = image_elem.get("sampleFormat", "Float32")
        dtype = _DTYPE_MAP.get(sample_format, np.dtype(np.float32))

        location = image_elem.get("location", "")
        if location.startswith("attachment:"):
            loc_parts = location.split(":")
            offset = int(loc_parts[1])
            size = int(loc_parts[2])
            f.seek(offset)
            raw_bytes = f.read(size)
        else:
            raw_bytes = f.read(width * height * channels * dtype.itemsize)

        data = np.frombuffer(raw_bytes, dtype=dtype).reshape((channels, height, width))
        data = data.astype(np.float32)

        metadata = _parse_xisf_metadata(root)
        metadata = ImageMetadata(
            exposure=metadata.exposure,
            gain_iso=metadata.gain_iso,
            temperature=metadata.temperature,
            date_obs=metadata.date_obs,
            width=width,
            height=height,
            channels=channels,
            extra=metadata.extra,
        )

    return data, metadata
