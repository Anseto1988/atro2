from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from astroai.core.catalog_cache import CatalogCache


@pytest.fixture()
def cache(tmp_path: Path) -> CatalogCache:
    return CatalogCache(cache_dir=tmp_path, ttl_days=30)


@pytest.fixture()
def sample_gaia_data() -> dict:
    return {
        "ra": [180.0, 180.1, 180.2],
        "dec": [45.0, 45.1, 45.2],
        "color_index": [0.5, 0.8, 1.2],
        "flux_ratio_rg": [1.1, 1.2, 1.3],
        "flux_ratio_bg": [0.9, 0.85, 0.8],
    }


class TestCatalogCacheInit:
    def test_creates_db_file(self, tmp_path: Path) -> None:
        cache = CatalogCache(cache_dir=tmp_path)
        assert (tmp_path / "catalog_cache.sqlite").exists()
        cache.close()

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        cache = CatalogCache(cache_dir=nested)
        assert nested.exists()
        assert (nested / "catalog_cache.sqlite").exists()
        cache.close()


class TestMakeKey:
    def test_deterministic(self) -> None:
        k1 = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.0)
        k2 = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.0)
        assert k1 == k2

    def test_different_catalogs_differ(self) -> None:
        k1 = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.0)
        k2 = CatalogCache.make_key("2mass", 180.0, 45.0, 1.0)
        assert k1 != k2

    def test_different_coords_differ(self) -> None:
        k1 = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.0)
        k2 = CatalogCache.make_key("gaia_dr3", 181.0, 45.0, 1.0)
        assert k1 != k2

    def test_extra_param_affects_key(self) -> None:
        k1 = CatalogCache.make_key("gaia_tap", 180.0, 45.0, 1.0, extra="mag15.0")
        k2 = CatalogCache.make_key("gaia_tap", 180.0, 45.0, 1.0, extra="mag16.0")
        assert k1 != k2

    def test_quantization_merges_close_coords(self) -> None:
        k1 = CatalogCache.make_key("gaia_dr3", 180.00001, 45.00002, 1.0)
        k2 = CatalogCache.make_key("gaia_dr3", 180.00003, 45.00004, 1.0)
        assert k1 == k2


class TestPutAndGet:
    def test_put_then_get(self, cache: CatalogCache, sample_gaia_data: dict) -> None:
        key = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.0)
        cache.put(key, "gaia_dr3", 180.0, 45.0, 1.0, sample_gaia_data)
        result = cache.get(key)
        assert result is not None
        assert result["ra"] == sample_gaia_data["ra"]
        assert result["dec"] == sample_gaia_data["dec"]

    def test_get_missing_returns_none(self, cache: CatalogCache) -> None:
        result = cache.get("nonexistent_hash")
        assert result is None

    def test_expired_entry_returns_none(self, tmp_path: Path, sample_gaia_data: dict) -> None:
        cache = CatalogCache(cache_dir=tmp_path, ttl_days=0)
        key = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.0)
        cache.put(key, "gaia_dr3", 180.0, 45.0, 1.0, sample_gaia_data)
        time.sleep(0.05)
        result = cache.get(key)
        assert result is None
        cache.close()

    def test_hit_count_increments(self, cache: CatalogCache, sample_gaia_data: dict) -> None:
        key = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.0)
        cache.put(key, "gaia_dr3", 180.0, 45.0, 1.0, sample_gaia_data)
        cache.get(key)
        cache.get(key)
        cache.get(key)
        stats = cache.stats()
        assert stats["total_hits"] == 3

    def test_overwrite_existing_entry(self, cache: CatalogCache) -> None:
        key = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.0)
        cache.put(key, "gaia_dr3", 180.0, 45.0, 1.0, {"version": 1})
        cache.put(key, "gaia_dr3", 180.0, 45.0, 1.0, {"version": 2})
        result = cache.get(key)
        assert result == {"version": 2}

    def test_second_call_latency_under_5ms(
        self, cache: CatalogCache, sample_gaia_data: dict,
    ) -> None:
        key = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.0)
        large_data = {
            "ra": list(range(1000)),
            "dec": list(range(1000)),
            "color_index": [0.5] * 1000,
            "flux_ratio_rg": [1.0] * 1000,
            "flux_ratio_bg": [0.9] * 1000,
        }
        cache.put(key, "gaia_dr3", 180.0, 45.0, 1.0, large_data)
        start = time.perf_counter()
        result = cache.get(key)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert result is not None
        assert elapsed_ms < 5.0, f"Cache read took {elapsed_ms:.1f}ms, expected <5ms"


class TestPurge:
    def test_purge_clears_all(self, cache: CatalogCache, sample_gaia_data: dict) -> None:
        for i in range(5):
            key = CatalogCache.make_key("gaia_dr3", 180.0 + i, 45.0, 1.0)
            cache.put(key, "gaia_dr3", 180.0 + i, 45.0, 1.0, sample_gaia_data)
        count = cache.purge()
        assert count == 5
        assert cache.stats()["total_entries"] == 0

    def test_purge_expired_only(self, tmp_path: Path, sample_gaia_data: dict) -> None:
        cache = CatalogCache(cache_dir=tmp_path, ttl_days=0)
        key_old = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.0)
        cache.put(key_old, "gaia_dr3", 180.0, 45.0, 1.0, sample_gaia_data)
        time.sleep(0.05)

        cache_fresh = CatalogCache(cache_dir=tmp_path, ttl_days=30)
        key_new = CatalogCache.make_key("2mass", 90.0, 30.0, 0.5)
        cache_fresh.put(key_new, "2mass", 90.0, 30.0, 0.5, sample_gaia_data)

        expired_count = cache_fresh.purge_expired()
        assert expired_count == 1
        assert cache_fresh.get(key_new) is not None
        cache.close()
        cache_fresh.close()


class TestStats:
    def test_empty_stats(self, cache: CatalogCache) -> None:
        stats = cache.stats()
        assert stats["total_entries"] == 0
        assert stats["total_hits"] == 0
        assert stats["by_catalog"] == {}

    def test_stats_by_catalog(self, cache: CatalogCache, sample_gaia_data: dict) -> None:
        for cat in ("gaia_dr3", "gaia_dr3", "2mass"):
            key = CatalogCache.make_key(cat, 180.0, 45.0, float(hash(cat) % 10))
            cache.put(key, cat, 180.0, 45.0, 1.0, sample_gaia_data)
        stats = cache.stats()
        assert stats["by_catalog"]["gaia_dr3"] >= 1
        assert stats["by_catalog"]["2mass"] == 1


class TestCalibratorCacheIntegration:
    def test_calibrator_uses_cache_on_second_call(self, tmp_path: Path) -> None:
        from astroai.processing.color.calibrator import (
            CatalogQueryResult,
            CatalogSource,
            SpectralColorCalibrator,
        )

        cached_data = {
            "ra": [180.0, 180.1],
            "dec": [45.0, 45.1],
            "color_index": [0.5, 0.8],
            "flux_ratio_rg": [1.1, 1.2],
            "flux_ratio_bg": [0.9, 0.85],
        }

        cache = CatalogCache(cache_dir=tmp_path)
        key = CatalogCache.make_key("gaia_dr3", 180.0, 45.0, 1.5)
        cache.put(key, "gaia_dr3", 180.0, 45.0, 1.5, cached_data)

        cal = SpectralColorCalibrator(catalog=CatalogSource.GAIA_DR3, use_cache=True)
        cal._cache = cache

        result = cal._query_gaia(180.0, 45.0, 1.5)
        assert isinstance(result, CatalogQueryResult)
        assert len(result.ra) == 2
        np.testing.assert_array_almost_equal(result.ra, [180.0, 180.1])


class TestPhotometryCacheIntegration:
    def test_gaia_client_returns_cached(self, tmp_path: Path) -> None:
        from astroai.engine.photometry.catalog import GAIACatalogClient

        cached_stars = [
            {"ra": 180.0, "dec": 45.0, "phot_g_mean_mag": 12.5},
            {"ra": 180.1, "dec": 45.1, "phot_g_mean_mag": 13.0},
        ]

        cache = CatalogCache(cache_dir=tmp_path)
        key = CatalogCache.make_key(
            "gaia_tap", 180.0, 45.0, 1.0, extra="mag15.0",
        )
        cache.put(
            key, "gaia_tap", 180.0, 45.0, 1.0, cached_stars,
            extra_key="mag15.0",
        )

        client = GAIACatalogClient(use_cache=True)
        client._cache = cache

        result = client.query(180.0, 45.0, 1.0, mag_limit=15.0)
        assert len(result) == 2
        assert result[0]["ra"] == 180.0
