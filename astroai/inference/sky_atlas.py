"""Sky atlas integration for deep-sky object identification within a field of view."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import httpx

__all__ = ["SkyAtlas", "SkyObject", "SkyAtlasResult"]

logger = logging.getLogger(__name__)

_SIMBAD_TAP_URL = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"


@dataclass(frozen=True)
class SkyObject:
    name: str
    ra_deg: float
    dec_deg: float
    object_type: str
    magnitude: float | None = None
    angular_distance_arcmin: float = 0.0


@dataclass(frozen=True)
class SkyAtlasResult:
    objects: list[SkyObject] = field(default_factory=list)
    fov_center_ra: float = 0.0
    fov_center_dec: float = 0.0
    search_radius_arcmin: float = 0.0
    confidence: float = 0.0
    solve_quality: float = 0.0


def _angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Compute angular separation in arcminutes between two sky positions (degrees)."""
    ra1_r = math.radians(ra1)
    dec1_r = math.radians(dec1)
    ra2_r = math.radians(ra2)
    dec2_r = math.radians(dec2)
    cos_sep = (
        math.sin(dec1_r) * math.sin(dec2_r)
        + math.cos(dec1_r) * math.cos(dec2_r) * math.cos(ra1_r - ra2_r)
    )
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep)) * 60.0


# Common deep-sky objects for offline/fast lookup
_LOCAL_DSO_CATALOG: list[dict[str, float | str | None]] = [
    {"name": "M31", "ra": 10.6847, "dec": 41.2687, "type": "Galaxy", "mag": 3.4},
    {"name": "M42", "ra": 83.8221, "dec": -5.3911, "type": "Nebula", "mag": 4.0},
    {"name": "M45", "ra": 56.601, "dec": 24.1154, "type": "Cluster", "mag": 1.6},
    {"name": "M51", "ra": 202.4696, "dec": 47.1952, "type": "Galaxy", "mag": 8.4},
    {"name": "M101", "ra": 210.8024, "dec": 54.3488, "type": "Galaxy", "mag": 7.9},
    {"name": "M104", "ra": 189.9976, "dec": -11.6231, "type": "Galaxy", "mag": 8.0},
    {"name": "NGC7000", "ra": 314.6821, "dec": 44.3178, "type": "Nebula", "mag": 4.0},
    {"name": "IC1396", "ra": 324.7513, "dec": 57.4933, "type": "Nebula", "mag": None},
    {"name": "M1", "ra": 83.6331, "dec": 22.0145, "type": "SNR", "mag": 8.4},
    {"name": "M8", "ra": 270.9042, "dec": -24.3842, "type": "Nebula", "mag": 6.0},
    {"name": "M13", "ra": 250.4217, "dec": 36.4613, "type": "Cluster", "mag": 5.8},
    {"name": "M16", "ra": 274.7000, "dec": -13.8067, "type": "Nebula", "mag": 6.0},
    {"name": "M20", "ra": 270.6225, "dec": -23.0300, "type": "Nebula", "mag": 6.3},
    {"name": "M27", "ra": 299.9017, "dec": 22.7211, "type": "PN", "mag": 7.5},
    {"name": "M33", "ra": 23.4621, "dec": 30.6602, "type": "Galaxy", "mag": 5.7},
    {"name": "M57", "ra": 283.3963, "dec": 33.0286, "type": "PN", "mag": 8.8},
    {"name": "M63", "ra": 198.9554, "dec": 42.0294, "type": "Galaxy", "mag": 8.6},
    {"name": "M81", "ra": 148.8882, "dec": 69.0653, "type": "Galaxy", "mag": 6.9},
    {"name": "M82", "ra": 148.9685, "dec": 69.6797, "type": "Galaxy", "mag": 8.4},
    {"name": "NGC2024", "ra": 85.4246, "dec": -1.9111, "type": "Nebula", "mag": None},
    {"name": "IC5070", "ra": 312.7517, "dec": 44.3675, "type": "Nebula", "mag": None},
    {"name": "NGC6992", "ra": 313.3871, "dec": 31.7221, "type": "SNR", "mag": 7.0},
    {"name": "M78", "ra": 86.6500, "dec": 0.0783, "type": "Nebula", "mag": 8.3},
    {"name": "NGC253", "ra": 11.8880, "dec": -25.2883, "type": "Galaxy", "mag": 7.1},
    {"name": "M64", "ra": 194.1829, "dec": 21.6825, "type": "Galaxy", "mag": 8.5},
]


class SkyAtlas:
    """Queries known deep-sky objects within a given field of view."""

    def __init__(
        self,
        timeout: float = 15.0,
        use_online: bool = True,
        max_results: int = 20,
    ) -> None:
        self._timeout = timeout
        self._use_online = use_online
        self._max_results = max_results

    def query(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcmin: float = 60.0,
        solve_rms_arcsec: float | None = None,
    ) -> SkyAtlasResult:
        """Query objects near the given sky position."""
        solve_quality = self._compute_solve_quality(solve_rms_arcsec)
        objects = self._local_search(ra_deg, dec_deg, radius_arcmin)

        if self._use_online and len(objects) < 3:
            online_objs = self._simbad_cone_search(ra_deg, dec_deg, radius_arcmin)
            if online_objs:
                existing_names = {o.name for o in objects}
                for obj in online_objs:
                    if obj.name not in existing_names:
                        objects.append(obj)

        objects.sort(key=lambda o: o.angular_distance_arcmin)
        objects = objects[: self._max_results]

        confidence = self._compute_confidence(objects, solve_quality)

        return SkyAtlasResult(
            objects=objects,
            fov_center_ra=ra_deg,
            fov_center_dec=dec_deg,
            search_radius_arcmin=radius_arcmin,
            confidence=confidence,
            solve_quality=solve_quality,
        )

    def _local_search(
        self, ra_deg: float, dec_deg: float, radius_arcmin: float
    ) -> list[SkyObject]:
        results: list[SkyObject] = []
        for entry in _LOCAL_DSO_CATALOG:
            obj_ra = float(entry["ra"])  # type: ignore[arg-type]
            obj_dec = float(entry["dec"])  # type: ignore[arg-type]
            sep = _angular_separation(ra_deg, dec_deg, obj_ra, obj_dec)
            if sep <= radius_arcmin:
                mag = float(entry["mag"]) if entry["mag"] is not None else None
                results.append(SkyObject(
                    name=str(entry["name"]),
                    ra_deg=obj_ra,
                    dec_deg=obj_dec,
                    object_type=str(entry["type"]),
                    magnitude=mag,
                    angular_distance_arcmin=sep,
                ))
        return results

    def _simbad_cone_search(
        self, ra_deg: float, dec_deg: float, radius_arcmin: float
    ) -> list[SkyObject]:
        """Query Simbad TAP for objects in cone."""
        radius_deg = radius_arcmin / 60.0
        query = (
            "SELECT TOP 20 main_id, ra, dec, otype_txt, V "
            "FROM basic "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) = 1 "
            "AND V < 15 "
            "ORDER BY V ASC"
        )
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    _SIMBAD_TAP_URL,
                    data={"request": "doQuery", "lang": "adql", "format": "tsv", "query": query},
                )
            if resp.status_code != 200:
                logger.warning("Simbad TAP query failed: HTTP %d", resp.status_code)
                return []
            return self._parse_tap_response(resp.text, ra_deg, dec_deg)
        except (httpx.HTTPError, Exception) as e:
            logger.warning("Simbad TAP error: %s", e)
            return []

    @staticmethod
    def _parse_tap_response(
        text: str, center_ra: float, center_dec: float
    ) -> list[SkyObject]:
        lines = text.strip().splitlines()
        if len(lines) < 2:
            return []
        objects: list[SkyObject] = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            name = parts[0].strip()
            try:
                ra = float(parts[1])
                dec = float(parts[2])
            except (ValueError, IndexError):
                continue
            otype = parts[3].strip() if len(parts) > 3 else "Unknown"
            mag: float | None = None
            if len(parts) > 4 and parts[4].strip():
                try:
                    mag = float(parts[4])
                except ValueError:
                    pass
            sep = _angular_separation(center_ra, center_dec, ra, dec)
            objects.append(SkyObject(
                name=name,
                ra_deg=ra,
                dec_deg=dec,
                object_type=otype,
                magnitude=mag,
                angular_distance_arcmin=sep,
            ))
        return objects

    @staticmethod
    def _compute_solve_quality(rms_arcsec: float | None) -> float:
        if rms_arcsec is None:
            return 0.5
        if rms_arcsec <= 0.5:
            return 1.0
        if rms_arcsec >= 10.0:
            return 0.1
        return max(0.1, 1.0 - (rms_arcsec - 0.5) / 9.5)

    @staticmethod
    def _compute_confidence(objects: list[SkyObject], solve_quality: float) -> float:
        if not objects:
            return solve_quality * 0.3
        obj_factor = min(len(objects) / 5.0, 1.0)
        return min(solve_quality * 0.6 + obj_factor * 0.4, 1.0)
