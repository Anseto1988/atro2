from __future__ import annotations

import logging
from typing import Any

import httpx

__all__ = ["AAVSOCatalogClient", "GAIACatalogClient"]

logger = logging.getLogger(__name__)

_VIZIER_TAP_URL = "http://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync"
_AAVSO_VSX_URL = "https://www.aavso.org/vsx/index.php"


class GAIACatalogClient:
    def __init__(self, *, fail_silently: bool = True) -> None:
        self._fail_silently = fail_silently

    def query(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float,
        mag_limit: float = 15.0,
    ) -> list[dict[str, Any]]:
        adql = (
            "SELECT ra, dec, phot_g_mean_mag "
            "FROM \"I/355/gaiadr3\" "
            f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})) "
            f"AND phot_g_mean_mag < {mag_limit}"
        )
        params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql}

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(_VIZIER_TAP_URL, params=params)
                resp.raise_for_status()
            data = resp.json()
            rows = data.get("data", [])
            columns = [c["name"] for c in data.get("metadata", [])]
            return [dict(zip(columns, row)) for row in rows]
        except Exception:
            if self._fail_silently:
                logger.warning("GAIA catalog query failed", exc_info=True)
                return []
            raise


class AAVSOCatalogClient:
    def __init__(self, *, fail_silently: bool = True) -> None:
        self._fail_silently = fail_silently

    def query(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float,
    ) -> list[dict[str, Any]]:
        params = {
            "view": "api.list",
            "ra": ra_center,
            "dec": dec_center,
            "radius": radius_deg,
            "format": "json",
        }

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(_AAVSO_VSX_URL, params=params)
                resp.raise_for_status()
            data = resp.json()
            return data.get("VSXObjects", {}).get("VSXObject", [])
        except Exception:
            if self._fail_silently:
                logger.warning("AAVSO VSX query failed", exc_info=True)
                return []
            raise
