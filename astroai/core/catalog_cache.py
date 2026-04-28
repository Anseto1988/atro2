"""SQLite-based local cache for astronomical catalog queries.

Avoids repeated network round-trips to GAIA DR3, 2MASS, and other
catalog services by caching cone-search results keyed on coordinates.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

__all__ = ["CatalogCache"]

logger = logging.getLogger(__name__)

_DEFAULT_TTL_DAYS = 30
_SCHEMA_VERSION = 1


def _default_cache_dir() -> Path:
    return Path.home() / ".astroai"


class CatalogCache:
    """Thread-safe SQLite catalog query cache."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_days: int = _DEFAULT_TTL_DAYS,
    ) -> None:
        self._dir = cache_dir or _default_cache_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "catalog_cache.sqlite"
        self._ttl_seconds = ttl_days * 86400
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catalog_queries (
                query_hash TEXT PRIMARY KEY,
                catalog    TEXT NOT NULL,
                ra         REAL NOT NULL,
                dec        REAL NOT NULL,
                radius_deg REAL NOT NULL,
                extra_key  TEXT NOT NULL DEFAULT '',
                result     TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                hit_count  INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_catalog_coords
            ON catalog_queries(catalog, ra, dec, radius_deg)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
            (_SCHEMA_VERSION,),
        )
        conn.commit()

    @staticmethod
    def make_key(
        catalog: str,
        ra: float,
        dec: float,
        radius_deg: float,
        extra: str = "",
    ) -> str:
        ra_q = round(ra, 4)
        dec_q = round(dec, 4)
        radius_q = round(radius_deg, 5)
        raw = f"{catalog}|{ra_q}|{dec_q}|{radius_q}|{extra}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query_hash: str) -> Any | None:
        conn = self._get_conn()
        now = time.time()
        row = conn.execute(
            "SELECT result, expires_at FROM catalog_queries WHERE query_hash = ?",
            (query_hash,),
        ).fetchone()
        if row is None:
            return None
        result_json, expires_at = row
        if now > expires_at:
            conn.execute(
                "DELETE FROM catalog_queries WHERE query_hash = ?",
                (query_hash,),
            )
            conn.commit()
            return None
        conn.execute(
            "UPDATE catalog_queries SET hit_count = hit_count + 1 WHERE query_hash = ?",
            (query_hash,),
        )
        conn.commit()
        return json.loads(result_json)

    def put(
        self,
        query_hash: str,
        catalog: str,
        ra: float,
        dec: float,
        radius_deg: float,
        result: Any,
        extra_key: str = "",
    ) -> None:
        now = time.time()
        result_json = json.dumps(result)
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO catalog_queries
                (query_hash, catalog, ra, dec, radius_deg, extra_key,
                 result, created_at, expires_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                query_hash,
                catalog,
                round(ra, 4),
                round(dec, 4),
                round(radius_deg, 5),
                extra_key,
                result_json,
                now,
                now + self._ttl_seconds,
            ),
        )
        conn.commit()

    def purge(self) -> int:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM catalog_queries")
        conn.commit()
        count = cursor.rowcount
        logger.info("Purged %d cached catalog entries", count)
        return count

    def purge_expired(self) -> int:
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute(
            "DELETE FROM catalog_queries WHERE expires_at < ?", (now,)
        )
        conn.commit()
        count = cursor.rowcount
        if count:
            logger.info("Purged %d expired catalog entries", count)
        return count

    def stats(self) -> dict[str, Any]:
        conn = self._get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM catalog_queries"
        ).fetchone()[0]
        total_hits = conn.execute(
            "SELECT COALESCE(SUM(hit_count), 0) FROM catalog_queries"
        ).fetchone()[0]
        by_catalog = dict(
            conn.execute(
                "SELECT catalog, COUNT(*) FROM catalog_queries GROUP BY catalog"
            ).fetchall()
        )
        return {
            "total_entries": total,
            "total_hits": total_hits,
            "by_catalog": by_catalog,
            "db_path": str(self._db_path),
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
