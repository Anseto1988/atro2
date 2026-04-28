from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from astroai.core.io.xisf_io import XISF_MAGIC, read_xisf


def _build_xisf(
    tmp_path: Path,
    data: np.ndarray,
    sample_format: str = "Float32",
    properties: dict[str, str] | None = None,
    fits_keywords: dict[str, str] | None = None,
) -> Path:
    channels, height, width = data.shape
    geometry = f"{width}:{height}:{channels}"
    raw_bytes = data.astype(np.float32).tobytes()
    ns = "http://www.pixinsight.com/xisf"

    def _make_tree(offset: int) -> ET.Element:
        root = ET.Element(f"{{{ns}}}xisf", attrib={"version": "1.0"})
        img = ET.SubElement(root, f"{{{ns}}}Image", attrib={
            "geometry": geometry,
            "sampleFormat": sample_format,
            "location": f"attachment:{offset}:{len(raw_bytes)}",
        })
        if properties:
            for pid, val in properties.items():
                ET.SubElement(img, f"{{{ns}}}Property", attrib={"id": pid, "value": val})
        if fits_keywords:
            for name, val in fits_keywords.items():
                ET.SubElement(img, f"{{{ns}}}FITSKeyword",
                              attrib={"name": name, "value": f"  '{val}'  "})
        return root

    draft = ET.tostring(_make_tree(0), encoding="unicode").encode("utf-8")
    padded_len = (len(draft) + 127) // 128 * 128

    attachment_offset = 16 + padded_len
    final_xml = ET.tostring(_make_tree(attachment_offset), encoding="unicode").encode("utf-8")
    header_padded = final_xml.ljust(padded_len, b"\x00")

    path = tmp_path / "test.xisf"
    with open(path, "wb") as f:
        f.write(XISF_MAGIC)
        f.write(struct.pack("<I", padded_len))
        f.write(b"\x00" * 4)
        f.write(header_padded)
        f.write(raw_bytes)

    return path


@pytest.fixture()
def sample_xisf(tmp_path: Path) -> Path:
    data = np.random.default_rng(99).random((1, 64, 128)).astype(np.float32)
    return _build_xisf(
        tmp_path,
        data,
        properties={
            "Instrument:Exposure": "120.0",
            "Instrument:ISO": "1600",
            "Instrument:Sensor:Temperature": "-10.5",
            "Observation:Time:Start": "2024-12-01T22:00:00",
        },
    )


class TestReadXisf:
    def test_returns_float32(self, sample_xisf: Path) -> None:
        data, _meta = read_xisf(sample_xisf)
        assert data.dtype == np.float32

    def test_shape_matches(self, sample_xisf: Path) -> None:
        data, meta = read_xisf(sample_xisf)
        assert data.shape == (1, 64, 128)
        assert meta.width == 128
        assert meta.height == 64
        assert meta.channels == 1

    def test_extracts_metadata_from_properties(self, sample_xisf: Path) -> None:
        _data, meta = read_xisf(sample_xisf)
        assert meta.exposure == pytest.approx(120.0)
        assert meta.gain_iso == 1600
        assert meta.temperature == pytest.approx(-10.5)
        assert meta.date_obs == "2024-12-01T22:00:00"

    def test_extracts_metadata_from_fits_keywords(self, tmp_path: Path) -> None:
        data = np.ones((1, 32, 32), dtype=np.float32)
        path = _build_xisf(
            tmp_path,
            data,
            fits_keywords={"EXPTIME": "60.0", "GAIN": "200", "CCD-TEMP": "-20.0", "DATE-OBS": "2024-06-15"},
        )
        _data, meta = read_xisf(path)
        assert meta.exposure == pytest.approx(60.0)
        assert meta.gain_iso == 200
        assert meta.temperature == pytest.approx(-20.0)
        assert meta.date_obs == "2024-06-15"

    def test_rejects_invalid_magic(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.xisf"
        path.write_bytes(b"NOT_XISF" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Not a valid XISF"):
            read_xisf(path)

    def test_three_channel_image(self, tmp_path: Path) -> None:
        data = np.random.default_rng(7).random((3, 50, 80)).astype(np.float32)
        path = _build_xisf(tmp_path, data)
        loaded, meta = read_xisf(path)
        assert loaded.shape == (3, 50, 80)
        assert meta.channels == 3
        np.testing.assert_array_almost_equal(loaded, data, decimal=5)


class TestReadXisfEdgeCases:
    def test_invalid_property_values_silently_ignored(self, tmp_path: Path) -> None:
        """Non-numeric Exposure/ISO/Temperature values fall to except pass (lines 36-37, 41-42, 46-47)."""
        data = np.ones((1, 8, 8), dtype=np.float32)
        path = _build_xisf(
            tmp_path, data,
            properties={
                "Instrument:Exposure": "bad_val",
                "Instrument:ISO": "not_a_number",
                "Instrument:Sensor:Temperature": "N/A",
            },
        )
        _d, meta = read_xisf(path)
        assert meta.exposure is None
        assert meta.gain_iso is None
        assert meta.temperature is None

    def test_unknown_property_stored_in_extra(self, tmp_path: Path) -> None:
        """Unknown property key is stored in ImageMetadata.extra (line 51)."""
        data = np.ones((1, 8, 8), dtype=np.float32)
        path = _build_xisf(
            tmp_path, data,
            properties={"Custom:MyKey": "some_value"},
        )
        _d, meta = read_xisf(path)
        assert "Custom:MyKey" in meta.extra
        assert meta.extra["Custom:MyKey"] == "some_value"

    def test_invalid_fits_keyword_values_ignored(self, tmp_path: Path) -> None:
        """Non-numeric FITSKeyword values fall to except pass (lines 59-60, 64-65, 69-70)."""
        data = np.ones((1, 8, 8), dtype=np.float32)
        path = _build_xisf(
            tmp_path, data,
            fits_keywords={"EXPTIME": "bad", "GAIN": "bad", "CCD-TEMP": "bad"},
        )
        _d, meta = read_xisf(path)
        assert meta.exposure is None
        assert meta.gain_iso is None
        assert meta.temperature is None

    def test_no_image_element_raises(self, tmp_path: Path) -> None:
        """XISF without Image element raises ValueError (line 172)."""
        ns = "http://www.pixinsight.com/xisf"
        root = ET.Element(f"{{{ns}}}xisf", attrib={"version": "1.0"})
        xml_bytes = ET.tostring(root, encoding="unicode").encode("utf-8")
        padded = (len(xml_bytes) + 127) // 128 * 128
        import struct
        path = tmp_path / "noimage.xisf"
        with open(path, "wb") as f:
            f.write(b"XISF0100")
            f.write(struct.pack("<I", padded))
            f.write(b"\x00" * 4)
            f.write(xml_bytes.ljust(padded, b"\x00"))
        with pytest.raises(ValueError, match="No Image element"):
            read_xisf(path)

    def test_two_part_geometry(self, tmp_path: Path) -> None:
        """2-part geometry (W:H) defaults to 1 channel (lines 178-180)."""
        ns = "http://www.pixinsight.com/xisf"
        data = np.ones((1, 8, 16), dtype=np.float32)
        raw_bytes = data.astype(np.float32).tobytes()
        root = ET.Element(f"{{{ns}}}xisf", attrib={"version": "1.0"})
        # Use 2-part geometry
        attachment_offset = 16 + 128  # rough estimate
        img_elem = ET.SubElement(root, f"{{{ns}}}Image", attrib={
            "geometry": "16:8",  # W:H, no channel count
            "sampleFormat": "Float32",
            "location": f"attachment:{16 + 128}:{len(raw_bytes)}",
        })
        xml_bytes = ET.tostring(root, encoding="unicode").encode("utf-8")
        padded_len = (len(xml_bytes) + 127) // 128 * 128
        actual_offset = 16 + padded_len
        # Rebuild with correct offset
        img_elem.set("location", f"attachment:{actual_offset}:{len(raw_bytes)}")
        xml_bytes = ET.tostring(root, encoding="unicode").encode("utf-8")
        padded_len = (len(xml_bytes) + 127) // 128 * 128
        header_padded = xml_bytes.ljust(padded_len, b"\x00")
        import struct
        path = tmp_path / "2geom.xisf"
        with open(path, "wb") as f:
            f.write(b"XISF0100")
            f.write(struct.pack("<I", padded_len))
            f.write(b"\x00" * 4)
            f.write(header_padded)
            f.write(raw_bytes)
        loaded, meta = read_xisf(path)
        assert meta.width == 16
        assert meta.height == 8
        assert meta.channels == 1

    def test_invalid_geometry_raises(self, tmp_path: Path) -> None:
        """Geometry with neither 2 nor 3 parts raises ValueError (line 182)."""
        import struct
        ns = "http://www.pixinsight.com/xisf"
        root = ET.Element(f"{{{ns}}}xisf", attrib={"version": "1.0"})
        ET.SubElement(root, f"{{{ns}}}Image", attrib={
            "geometry": "100",  # only 1 part - invalid
            "sampleFormat": "Float32",
            "location": "attachment:256:400",
        })
        xml_bytes = ET.tostring(root, encoding="unicode").encode("utf-8")
        padded_len = (len(xml_bytes) + 127) // 128 * 128
        path = tmp_path / "badgeom.xisf"
        with open(path, "wb") as f:
            f.write(b"XISF0100")
            f.write(struct.pack("<I", padded_len))
            f.write(b"\x00" * 4)
            f.write(xml_bytes.ljust(padded_len, b"\x00"))
            f.write(b"\x00" * 400)
        with pytest.raises(ValueError, match="Invalid geometry"):
            read_xisf(path)

    def test_inline_data_location(self, tmp_path: Path) -> None:
        """Image data without attachment: prefix is read inline (line 195)."""
        import struct
        ns = "http://www.pixinsight.com/xisf"
        data = np.ones((1, 4, 4), dtype=np.float32)
        raw_bytes = data.tobytes()
        root = ET.Element(f"{{{ns}}}xisf", attrib={"version": "1.0"})
        ET.SubElement(root, f"{{{ns}}}Image", attrib={
            "geometry": "4:4:1",
            "sampleFormat": "Float32",
            "location": "inline:encoded",  # not attachment:
        })
        xml_bytes = ET.tostring(root, encoding="unicode").encode("utf-8")
        padded_len = (len(xml_bytes) + 127) // 128 * 128
        path = tmp_path / "inline.xisf"
        with open(path, "wb") as f:
            f.write(b"XISF0100")
            f.write(struct.pack("<I", padded_len))
            f.write(b"\x00" * 4)
            f.write(xml_bytes.ljust(padded_len, b"\x00"))
            f.write(raw_bytes)
        loaded, meta = read_xisf(path)
        assert loaded.shape == (1, 4, 4)
        assert meta.channels == 1
