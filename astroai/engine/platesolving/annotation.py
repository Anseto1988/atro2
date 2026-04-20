"""On-screen annotation overlay using WCS for coordinate mapping."""

from __future__ import annotations

from dataclasses import dataclass

from astropy.wcs import WCS


__all__ = ["AnnotationOverlay", "CelestialObject"]


@dataclass(frozen=True)
class CelestialObject:
    name: str
    ra: float
    dec: float
    magnitude: float = 0.0
    object_type: str = "unknown"


class AnnotationOverlay:
    def __init__(self, wcs: WCS, image_shape: tuple[int, int]) -> None:
        self._wcs = wcs
        self._image_shape = image_shape

    @property
    def wcs(self) -> WCS:
        return self._wcs

    @property
    def image_shape(self) -> tuple[int, int]:
        return self._image_shape

    def world_to_pixel(self, ra: float, dec: float) -> tuple[float, float]:
        pixel_coords = self._wcs.world_to_pixel_values(ra, dec)
        return float(pixel_coords[0]), float(pixel_coords[1])

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]:
        world_coords = self._wcs.pixel_to_world_values(x, y)
        return float(world_coords[0]), float(world_coords[1])

    def is_in_fov(self, ra: float, dec: float) -> bool:
        x, y = self.world_to_pixel(ra, dec)
        h, w = self._image_shape
        return 0 <= x < w and 0 <= y < h

    def get_visible_objects(
        self, catalog: list[CelestialObject]
    ) -> list[tuple[CelestialObject, tuple[float, float]]]:
        visible = []
        for obj in catalog:
            if self.is_in_fov(obj.ra, obj.dec):
                px = self.world_to_pixel(obj.ra, obj.dec)
                visible.append((obj, px))
        return visible

    def get_fov_corners_world(self) -> list[tuple[float, float]]:
        h, w = self._image_shape
        corners_px = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
        return [self.pixel_to_world(x, y) for x, y in corners_px]

    def get_fov_center_world(self) -> tuple[float, float]:
        h, w = self._image_shape
        return self.pixel_to_world(w / 2.0, h / 2.0)
