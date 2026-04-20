"""WCS solution dataclass, coordinate transforms, and ASTAP catalog management."""

from __future__ import annotations

import logging
import math
import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AstapCatalog",
    "CatalogManager",
    "WcsSolution",
    "pixel_to_radec",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WcsSolution:
    """WCS plate solution returned by the ASTAP solver.

    All angular quantities are in degrees.
    """

    ra_center: float
    dec_center: float
    pixel_scale_arcsec: float  # arcsec / pixel
    rotation_deg: float        # north angle, east of north
    fov_width_deg: float
    fov_height_deg: float
    # FITS WCS CD matrix (2x2), row-major [[CD1_1, CD1_2], [CD2_1, CD2_2]]
    cd_matrix: tuple[float, float, float, float]
    crpix1: float
    crpix2: float

    @property
    def pixel_scale_deg(self) -> float:
        return self.pixel_scale_arcsec / 3600.0


def pixel_to_radec(
    solution: WcsSolution,
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Convert pixel coordinates to (RA, Dec) in degrees using linear WCS.

    Uses the CD matrix for projection (suitable for small fields; does not
    apply full SIP distortion correction).

    Args:
        solution: Plate solution from AstapSolver.
        x: Pixel x-coordinates (0-indexed).
        y: Pixel y-coordinates (0-indexed).

    Returns:
        Tuple of (ra_deg, dec_deg) arrays.
    """
    cd1_1, cd1_2, cd2_1, cd2_2 = solution.cd_matrix
    dx = x - (solution.crpix1 - 1.0)  # convert to 0-indexed
    dy = y - (solution.crpix2 - 1.0)

    delta_ra_deg = cd1_1 * dx + cd1_2 * dy
    delta_dec_deg = cd2_1 * dx + cd2_2 * dy

    cos_dec = math.cos(math.radians(solution.dec_center))
    ra = solution.ra_center + delta_ra_deg / cos_dec
    dec = solution.dec_center + delta_dec_deg

    ra = ra % 360.0
    return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)


# ------------------------------------------------------------------
# ASTAP star catalog management
# ------------------------------------------------------------------

_CATALOG_URLS: dict[str, str] = {
    "H18": "https://sourceforge.net/projects/astap-program/files/star_databases/h18_star_database.zip",
    "D50": "https://sourceforge.net/projects/astap-program/files/star_databases/d50_star_database.zip",
}

_FOV_THRESHOLD_DEG = 2.0


class AstapCatalog(Enum):
    """ASTAP star catalog variants."""

    H18 = "H18"
    D50 = "D50"


class CatalogManager:
    """Manages ASTAP star catalog downloads and availability checks.

    ASTAP requires a local star catalog for plate solving. H18 (Hipparcos, 18 mag)
    is lightweight and suited for wide-field FOV > 2 deg. D50 (deep sky, 50M stars)
    is for narrow fields.

    Args:
        catalog_dir: Where catalogs are stored. If None, uses ASTAP's default location.
    """

    def __init__(self, catalog_dir: Path | None = None) -> None:
        self._dir = catalog_dir or self._default_catalog_dir()

    @property
    def catalog_dir(self) -> Path:
        return self._dir

    def recommend_catalog(self, fov_deg: float) -> AstapCatalog:
        """Recommend H18 or D50 based on field-of-view."""
        return AstapCatalog.H18 if fov_deg > _FOV_THRESHOLD_DEG else AstapCatalog.D50

    def is_installed(self, catalog: AstapCatalog) -> bool:
        """Check whether a catalog's data files are present."""
        pattern = f"{catalog.value.lower()}*.290"
        return any(self._dir.glob(pattern))

    def ensure_available(self, catalog: AstapCatalog) -> Path:
        """Return catalog path, raising if not installed.

        This method does NOT auto-download; use :meth:`download` for that.
        The UI layer should prompt the user before downloading.
        """
        if not self.is_installed(catalog):
            raise FileNotFoundError(
                f"ASTAP catalog {catalog.value} not found in {self._dir}. "
                f"Download from {_CATALOG_URLS[catalog.value]}"
            )
        return self._dir

    def download_url(self, catalog: AstapCatalog) -> str:
        """Return the download URL for a catalog."""
        return _CATALOG_URLS[catalog.value]

    def download(self, catalog: AstapCatalog) -> Path:
        """Download and extract a star catalog.

        Raises:
            RuntimeError: On download/extraction failure.
        """
        import io
        import zipfile

        import httpx

        url = _CATALOG_URLS[catalog.value]
        logger.info("Downloading ASTAP catalog %s from %s", catalog.value, url)

        self._dir.mkdir(parents=True, exist_ok=True)

        try:
            with httpx.Client(follow_redirects=True, timeout=300) as client:
                resp = client.get(url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Catalog download failed: {exc}") from exc

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(self._dir)

        logger.info("Catalog %s installed to %s", catalog.value, self._dir)
        return self._dir

    @staticmethod
    def _default_catalog_dir() -> Path:
        system = platform.system()
        if system == "Windows":
            return Path("C:/Program Files/astap/star_databases")
        if system == "Darwin":
            return Path("/Applications/astap.app/Contents/Resources/star_databases")
        return Path("/opt/astap/star_databases")
