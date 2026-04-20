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


_DTYPE_REVERSE_MAP: dict[np.dtype[Any], str] = {v: k for k, v in _DTYPE_MAP.items()}


def write_xisf(
    path: str | Path,
    data: NDArray[np.floating[Any]],
    metadata: ImageMetadata | None = None,
) -> Path:
    path = Path(path)
    if data.ndim == 2:
        channels, height, width = 1, data.shape[0], data.shape[1]
        data = data.reshape(1, height, width)
    elif data.ndim == 3:
        channels, height, width = data.shape
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

    data = data.astype(np.float32)
    raw_bytes = data.tobytes()
    sample_format = _DTYPE_REVERSE_MAP.get(data.dtype, "Float32")

    ns = "http://www.pixinsight.com/xisf"

    def _build_xml(attachment_offset: int) -> bytes:
        root = ET.Element(f"{{{ns}}}xisf", attrib={"version": "1.0"})
        geometry = f"{width}:{height}:{channels}"
        location = f"attachment:{attachment_offset}:{len(raw_bytes)}"
        img = ET.SubElement(root, f"{{{ns}}}Image", attrib={
            "geometry": geometry,
            "sampleFormat": sample_format,
            "location": location,
        })
        if metadata:
            if metadata.exposure is not None:
                ET.SubElement(img, f"{{{ns}}}Property", attrib={
                    "id": "Instrument:Exposure", "type": "Float64",
                    "value": str(metadata.exposure),
                })
            if metadata.gain_iso is not None:
                ET.SubElement(img, f"{{{ns}}}Property", attrib={
                    "id": "Instrument:Sensor:Gain", "type": "Int32",
                    "value": str(metadata.gain_iso),
                })
            if metadata.temperature is not None:
                ET.SubElement(img, f"{{{ns}}}Property", attrib={
                    "id": "Instrument:Sensor:Temperature", "type": "Float64",
                    "value": str(metadata.temperature),
                })
            if metadata.date_obs is not None:
                ET.SubElement(img, f"{{{ns}}}Property", attrib={
                    "id": "Observation:Time:Start", "type": "String",
                    "value": metadata.date_obs,
                })
            if metadata.extra:
                for key, val in metadata.extra.items():
                    ET.SubElement(img, f"{{{ns}}}FITSKeyword", attrib={
                        "name": key, "value": f"'{val}'", "comment": "",
                    })
        return ET.tostring(root, encoding="unicode").encode("utf-8")

    draft = _build_xml(0)
    padded_len = (len(draft) + 127) // 128 * 128
    attachment_offset = 16 + padded_len
    final_xml = _build_xml(attachment_offset)
    header_padded = final_xml.ljust(padded_len, b"\x00")

    with open(path, "wb") as f:
        f.write(XISF_MAGIC)
        f.write(struct.pack("<I", padded_len))
        f.write(b"\x00" * 4)
        f.write(header_padded)
        f.write(raw_bytes)

    return path


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
