"""AI-assisted coordinate extraction from FITS headers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

__all__ = ["CoordinateExtractor", "Coordinates", "ExtractionMethod"]

logger = logging.getLogger(__name__)

_SIMBAD_TAP_URL = "https://simbad.cds.unistra.fr/simbad/sim-id"
_SESAME_URL = "https://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oI"


class ExtractionMethod:
    OBJCTRA_DEC = "objctra_dec"
    RA_DEC = "ra_dec"
    CRVAL = "crval"
    OBJECT_RESOLVE = "object_resolve"
    NONE = "none"


@dataclass(frozen=True)
class Coordinates:
    ra_deg: float
    dec_deg: float
    method: str
    confidence: float
    object_name: str | None = None


def _parse_sexagesimal_ra(value: str) -> float | None:
    """Parse RA in HH MM SS.s or HH:MM:SS.s format to degrees."""
    value = value.strip().replace(":", " ")
    parts = value.split()
    if len(parts) < 2:
        return None
    try:
        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2]) if len(parts) > 2 else 0.0
        return (h + m / 60.0 + s / 3600.0) * 15.0
    except (ValueError, IndexError):
        return None


def _parse_sexagesimal_dec(value: str) -> float | None:
    """Parse Dec in DD MM SS.s or DD:MM:SS.s format to degrees."""
    value = value.strip().replace(":", " ")
    match = re.match(r"([+-]?\d+)\s+(\d+\.?\d*)\s*(\d+\.?\d*)?", value)
    if not match:
        return None
    try:
        d = float(match.group(1))
        m = float(match.group(2))
        s = float(match.group(3)) if match.group(3) else 0.0
        sign = -1.0 if d < 0 or value.strip().startswith("-") else 1.0
        return sign * (abs(d) + m / 60.0 + s / 3600.0)
    except (ValueError, IndexError):
        return None


def _try_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CoordinateExtractor:
    """Extracts sky coordinates from FITS headers with AI-fallback name resolution."""

    def __init__(
        self,
        timeout: float = 10.0,
        cache: dict[str, Coordinates] | None = None,
    ) -> None:
        self._timeout = timeout
        self._cache: dict[str, Coordinates] = cache if cache is not None else {}

    def extract(self, header: dict[str, Any]) -> Coordinates:
        """Extract coordinates using priority cascade."""
        result = self._try_objctra_dec(header)
        if result:
            return result

        result = self._try_ra_dec(header)
        if result:
            return result

        result = self._try_crval(header)
        if result:
            return result

        result = self._try_object_resolve(header)
        if result:
            return result

        return Coordinates(
            ra_deg=0.0,
            dec_deg=0.0,
            method=ExtractionMethod.NONE,
            confidence=0.0,
        )

    def _try_objctra_dec(self, header: dict[str, Any]) -> Coordinates | None:
        ra_str = header.get("OBJCTRA")
        dec_str = header.get("OBJCTDEC")
        if not ra_str or not dec_str:
            return None
        ra = _parse_sexagesimal_ra(str(ra_str))
        dec = _parse_sexagesimal_dec(str(dec_str))
        if ra is None or dec is None:
            return None
        return Coordinates(
            ra_deg=ra,
            dec_deg=dec,
            method=ExtractionMethod.OBJCTRA_DEC,
            confidence=0.95,
        )

    def _try_ra_dec(self, header: dict[str, Any]) -> Coordinates | None:
        ra_val = header.get("RA")
        dec_val = header.get("DEC")
        if ra_val is None or dec_val is None:
            return None
        if isinstance(ra_val, str):
            ra = _parse_sexagesimal_ra(ra_val)
        else:
            ra = _try_float(ra_val)
        if isinstance(dec_val, str):
            dec = _parse_sexagesimal_dec(dec_val)
        else:
            dec = _try_float(dec_val)
        if ra is None or dec is None:
            return None
        return Coordinates(
            ra_deg=ra,
            dec_deg=dec,
            method=ExtractionMethod.RA_DEC,
            confidence=0.90,
        )

    def _try_crval(self, header: dict[str, Any]) -> Coordinates | None:
        crval1 = _try_float(header.get("CRVAL1"))
        crval2 = _try_float(header.get("CRVAL2"))
        if crval1 is None or crval2 is None:
            return None
        ctype1 = str(header.get("CTYPE1", "")).upper()
        ctype2 = str(header.get("CTYPE2", "")).upper()
        if "RA" in ctype1 or "RA" in ctype2 or (not ctype1 and not ctype2):
            return Coordinates(
                ra_deg=crval1,
                dec_deg=crval2,
                method=ExtractionMethod.CRVAL,
                confidence=0.85,
            )
        if "DEC" in ctype1:
            return Coordinates(
                ra_deg=crval2,
                dec_deg=crval1,
                method=ExtractionMethod.CRVAL,
                confidence=0.85,
            )
        return Coordinates(
            ra_deg=crval1,
            dec_deg=crval2,
            method=ExtractionMethod.CRVAL,
            confidence=0.70,
        )

    def _try_object_resolve(self, header: dict[str, Any]) -> Coordinates | None:
        obj_name = header.get("OBJECT")
        if not obj_name:
            return None
        obj_name = str(obj_name).strip()
        if not obj_name or obj_name.lower() in ("", "light", "dark", "flat", "bias"):
            return None
        if obj_name in self._cache:
            cached = self._cache[obj_name]
            logger.debug("Cache hit for object: %s", obj_name)
            return cached
        coords = self._resolve_via_sesame(obj_name)
        if coords:
            self._cache[obj_name] = coords
        return coords

    def _resolve_via_sesame(self, name: str) -> Coordinates | None:
        """Resolve object name to coordinates via CDS Sesame service."""
        try:
            url = f"{_SESAME_URL}/{name}"
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url)
            if response.status_code != 200:
                logger.warning("Sesame lookup failed for %s: HTTP %d", name, response.status_code)
                return None
            return self._parse_sesame_response(response.text, name)
        except httpx.HTTPError as e:
            logger.warning("Sesame lookup error for %s: %s", name, e)
            return None

    @staticmethod
    def _parse_sesame_response(text: str, name: str) -> Coordinates | None:
        """Parse Sesame plain-text response for J2000 coordinates."""
        for line in text.splitlines():
            if line.startswith("%J "):
                parts = line[3:].split()
                if len(parts) >= 2:
                    ra = _try_float(parts[0])
                    dec = _try_float(parts[1])
                    if ra is not None and dec is not None:
                        return Coordinates(
                            ra_deg=ra,
                            dec_deg=dec,
                            method=ExtractionMethod.OBJECT_RESOLVE,
                            confidence=0.75,
                            object_name=name,
                        )
        return None
