"""Sky object catalog models and CSV loading."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

__all__ = [
    "CatalogObject",
    "ConstellationBoundarySegment",
    "NamedStar",
    "SkyObjectCatalog",
    "WcsTransform",
]

_log = logging.getLogger(__name__)
_CATALOGS_DIR = Path(__file__).resolve().parent / "catalogs"


@runtime_checkable
class WcsTransform(Protocol):
    """Abstraction for WCS coordinate-to-pixel conversion."""

    def world_to_pixel(self, ra_deg: float, dec_deg: float) -> tuple[float, float] | None:
        """Convert RA/Dec (degrees) to pixel (x, y). Returns None if outside image."""
        ...

    def image_size(self) -> tuple[int, int]:
        """Return (width, height) of the image in pixels."""
        ...

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float] | None:
        """Convert pixel (x, y) to RA/Dec (degrees). Returns None if outside WCS."""
        ...


@dataclass(frozen=True, slots=True)
class CatalogObject:
    """Deep-sky object (Messier, NGC, IC)."""

    designation: str
    ra_deg: float
    dec_deg: float
    obj_type: str = ""
    magnitude: float = 99.0
    size_arcmin: float = 0.0
    common_name: str = ""


@dataclass(frozen=True, slots=True)
class NamedStar:
    """Bright star with common name from Hipparcos."""

    hip_id: int
    name: str
    ra_deg: float
    dec_deg: float
    magnitude: float


@dataclass(frozen=True, slots=True)
class ConstellationBoundarySegment:
    """Single line segment of an IAU constellation boundary."""

    ra1_deg: float
    dec1_deg: float
    ra2_deg: float
    dec2_deg: float
    constellation: str = ""


class SkyObjectCatalog:
    """Loads and provides access to embedded sky catalogs."""

    def __init__(self) -> None:
        self._dso: list[CatalogObject] = []
        self._stars: list[NamedStar] = []
        self._boundaries: list[ConstellationBoundarySegment] = []
        self._loaded = False

    @property
    def deep_sky_objects(self) -> Sequence[CatalogObject]:
        self._ensure_loaded()
        return self._dso

    @property
    def named_stars(self) -> Sequence[NamedStar]:
        self._ensure_loaded()
        return self._stars

    @property
    def constellation_boundaries(self) -> Sequence[ConstellationBoundarySegment]:
        self._ensure_loaded()
        return self._boundaries

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load_dso()
        self._load_stars()
        self._load_boundaries()

    def _load_dso(self) -> None:
        path = _CATALOGS_DIR / "dso_catalog.csv"
        if not path.exists():
            _log.warning("DSO-Katalog nicht gefunden: %s", path)
            return
        try:
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._dso.append(CatalogObject(
                        designation=row["designation"],
                        ra_deg=float(row["ra_deg"]),
                        dec_deg=float(row["dec_deg"]),
                        obj_type=row.get("type", ""),
                        magnitude=float(row.get("mag", "99")),
                        size_arcmin=float(row.get("size_arcmin", "0")),
                        common_name=row.get("common_name", ""),
                    ))
            _log.info("DSO-Katalog geladen: %d Objekte", len(self._dso))
        except Exception:
            _log.exception("Fehler beim Laden des DSO-Katalogs")

    def _load_stars(self) -> None:
        path = _CATALOGS_DIR / "named_stars.csv"
        if not path.exists():
            _log.warning("Sternkatalog nicht gefunden: %s", path)
            return
        try:
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._stars.append(NamedStar(
                        hip_id=int(row["hip_id"]),
                        name=row["name"],
                        ra_deg=float(row["ra_deg"]),
                        dec_deg=float(row["dec_deg"]),
                        magnitude=float(row["mag"]),
                    ))
            _log.info("Sternkatalog geladen: %d Sterne", len(self._stars))
        except Exception:
            _log.exception("Fehler beim Laden des Sternkatalogs")

    def _load_boundaries(self) -> None:
        path = _CATALOGS_DIR / "constellation_boundaries.csv"
        if not path.exists():
            _log.warning("Konstellationsgrenzen nicht gefunden: %s", path)
            return
        try:
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._boundaries.append(ConstellationBoundarySegment(
                        ra1_deg=float(row["ra1_deg"]),
                        dec1_deg=float(row["dec1_deg"]),
                        ra2_deg=float(row["ra2_deg"]),
                        dec2_deg=float(row["dec2_deg"]),
                        constellation=row.get("constellation", ""),
                    ))
            _log.info("Konstellationsgrenzen geladen: %d Segmente", len(self._boundaries))
        except Exception:
            _log.exception("Fehler beim Laden der Konstellationsgrenzen")
