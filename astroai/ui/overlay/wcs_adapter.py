"""Adapter bridging engine-level WCS to the UI WcsTransform protocol."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astropy.wcs import WCS

__all__ = ["WcsAdapter"]


class WcsAdapter:
    """Wraps an astropy WCS to satisfy the WcsTransform protocol."""

    def __init__(self, wcs: WCS, image_width: int, image_height: int) -> None:
        self._wcs = wcs
        self._w = image_width
        self._h = image_height

    def world_to_pixel(self, ra_deg: float, dec_deg: float) -> tuple[float, float] | None:
        px = self._wcs.world_to_pixel_values(ra_deg, dec_deg)
        x, y = float(px[0]), float(px[1])
        if -self._w * 0.1 <= x <= self._w * 1.1 and -self._h * 0.1 <= y <= self._h * 1.1:
            return (x, y)
        return None

    def image_size(self) -> tuple[int, int]:
        return (self._w, self._h)

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float] | None:
        world = self._wcs.pixel_to_world_values(x, y)
        return (float(world[0]), float(world[1]))

    @classmethod
    def from_engine_overlay(cls, overlay: object) -> WcsAdapter:
        """Create adapter from engine's AnnotationOverlay."""
        from astroai.engine.platesolving.annotation import AnnotationOverlay as EngineOverlay

        assert isinstance(overlay, EngineOverlay)
        h, w = overlay.image_shape
        return cls(overlay.wcs, w, h)

    @classmethod
    def from_solve_result(cls, solve_result: object, image_shape: tuple[int, int]) -> WcsAdapter:
        """Create adapter from PlateSolver's SolveResult."""
        from astroai.engine.platesolving.solver import SolveResult

        assert isinstance(solve_result, SolveResult)
        h, w = image_shape
        return cls(solve_result.wcs, w, h)
