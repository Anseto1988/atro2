from __future__ import annotations

import logging
from typing import Any

import httpx

__all__ = ["AAVSOCatalogClient", "GAIACatalogClient"]

logger = logging.getLogger(__name__)

_VIZIER_TAP_URL = "http://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync"
_AAVSO_VSX_URL = "https://www.aavso.org/vsx/index.php"


class GAIACatalogClient:
    def __init__(
        self,
        *,
        fail_silently: bool = True,
        use_cache: bool = True,
    ) -> None:
        self._fail_silently = fail_silently
        self._cache = None
        if use_cache:
            try:
                from astroai.core.catalog_cache import CatalogCache
                self._cache = CatalogCache()
            except Exception:
                logger.debug("Catalog cache unavailable, proceeding without cache")

    def query(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float,
        mag_limit: float = 15.0,
    ) -> list[dict[str, Any]]:
        if self._cache is not None:
            from astroai.core.catalog_cache import CatalogCache
            key = CatalogCache.make_key(
                "gaia_tap", ra_center, dec_center, radius_deg,
                extra=f"mag{mag_limit}",
            )
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("GAIA TAP cache hit")
                return cached

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
            result = [dict(zip(columns, row)) for row in rows]

            if self._cache is not None:
                from astroai.core.catalog_cache import CatalogCache
                key = CatalogCache.make_key(
                    "gaia_tap", ra_center, dec_center, radius_deg,
                    extra=f"mag{mag_limit}",
                )
                self._cache.put(
                    key, "gaia_tap", ra_center, dec_center, radius_deg, result,
                    extra_key=f"mag{mag_limit}",
                )

            return result
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
        params: dict[str, str | float] = {
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
            result: list[dict[str, Any]] = data.get("VSXObjects", {}).get("VSXObject", [])
            return result
        except Exception:
            if self._fail_silently:
                logger.warning("AAVSO VSX query failed", exc_info=True)
                return []
            raise
