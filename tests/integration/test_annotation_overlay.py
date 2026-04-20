"""Integration tests for annotation overlay.

Tests WCS-to-pixel mapping and Field-of-View clipping for celestial objects.
No network access required.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from astropy.wcs import WCS

from astroai.engine.platesolving.annotation import AnnotationOverlay, CelestialObject
from astroai.engine.platesolving.wcs_writer import WCSWriter


# --- Fixtures ---


@pytest.fixture()
def standard_wcs() -> WCS:
    """1024x1024 image centered on (180, 45) with ~1 arcsec/pixel scale."""
    writer = WCSWriter()
    return writer.create_wcs(
        crval=(180.0, 45.0),
        crpix=(512.0, 512.0),
        cdelt=(-0.000277, 0.000277),
        naxis=(1024, 1024),
    )


@pytest.fixture()
def wide_field_wcs() -> WCS:
    """2048x2048 image with wider FoV (~2 deg), centered on (10, 20)."""
    writer = WCSWriter()
    return writer.create_wcs(
        crval=(10.0, 20.0),
        crpix=(1024.0, 1024.0),
        cdelt=(-0.001, 0.001),
        naxis=(2048, 2048),
    )


@pytest.fixture()
def overlay(standard_wcs: WCS) -> AnnotationOverlay:
    return AnnotationOverlay(wcs=standard_wcs, image_shape=(1024, 1024))


@pytest.fixture()
def wide_overlay(wide_field_wcs: WCS) -> AnnotationOverlay:
    return AnnotationOverlay(wcs=wide_field_wcs, image_shape=(2048, 2048))


@pytest.fixture()
def sample_catalog() -> list[CelestialObject]:
    return [
        CelestialObject(name="M31", ra=10.684, dec=41.269, magnitude=3.4, object_type="galaxy"),
        CelestialObject(name="M42", ra=83.822, dec=-5.391, magnitude=4.0, object_type="nebula"),
        CelestialObject(name="M45", ra=56.75, dec=24.116, magnitude=1.6, object_type="cluster"),
        CelestialObject(name="Target-A", ra=180.0, dec=45.0, magnitude=5.0, object_type="star"),
        CelestialObject(
            name="Target-B", ra=180.001, dec=45.001, magnitude=6.0, object_type="star"
        ),
        CelestialObject(name="Far-Away", ra=0.0, dec=-89.0, magnitude=2.0, object_type="star"),
    ]


# --- WCS-to-Pixel Mapping Tests ---


class TestWCSToPixelMapping:
    def test_center_maps_to_reference_pixel(self, overlay: AnnotationOverlay) -> None:
        x, y = overlay.world_to_pixel(180.0, 45.0)
        # FITS CRPIX is 1-based; 0-based pixel = crpix - 1 = 511
        assert x == pytest.approx(511.0, abs=1.5)
        assert y == pytest.approx(511.0, abs=1.5)

    def test_pixel_to_world_roundtrip(self, overlay: AnnotationOverlay) -> None:
        ra, dec = overlay.pixel_to_world(256.0, 768.0)
        x, y = overlay.world_to_pixel(ra, dec)
        assert x == pytest.approx(256.0, abs=0.01)
        assert y == pytest.approx(768.0, abs=0.01)

    def test_world_to_pixel_roundtrip(self, overlay: AnnotationOverlay) -> None:
        x, y = overlay.world_to_pixel(180.05, 45.02)
        ra, dec = overlay.pixel_to_world(x, y)
        assert ra == pytest.approx(180.05, abs=0.001)
        assert dec == pytest.approx(45.02, abs=0.001)

    def test_offset_from_center_in_expected_direction(
        self, overlay: AnnotationOverlay
    ) -> None:
        x_center, y_center = overlay.world_to_pixel(180.0, 45.0)
        x_east, y_east = overlay.world_to_pixel(180.01, 45.0)
        # RA increases to the east; with negative CDELT1, x should decrease
        assert x_east < x_center

    def test_dec_offset_direction(self, overlay: AnnotationOverlay) -> None:
        x_center, y_center = overlay.world_to_pixel(180.0, 45.0)
        x_north, y_north = overlay.world_to_pixel(180.0, 45.01)
        # Dec increases to north; with positive CDELT2, y should increase
        assert y_north > y_center

    def test_known_coordinate_pixel_position(self, overlay: AnnotationOverlay) -> None:
        x, y = overlay.world_to_pixel(180.0, 45.0)
        # CRPIX 512 in FITS = 0-based pixel 511
        assert abs(x - 511.0) <= 1.0
        assert abs(y - 511.0) <= 1.0

    def test_corner_coordinates_are_consistent(self, overlay: AnnotationOverlay) -> None:
        corners = overlay.get_fov_corners_world()
        assert len(corners) == 4
        for ra, dec in corners:
            assert 0.0 <= ra <= 360.0 or -180.0 <= ra <= 180.0
            assert -90.0 <= dec <= 90.0

    def test_wide_field_pixel_scale(self, wide_overlay: AnnotationOverlay) -> None:
        x1, y1 = wide_overlay.world_to_pixel(10.0, 20.0)
        x2, y2 = wide_overlay.world_to_pixel(10.001, 20.0)
        pixel_sep = abs(x2 - x1)
        assert pixel_sep == pytest.approx(1.0, abs=0.1)


# --- Field-of-View Clipping Tests ---


class TestFoVClipping:
    def test_object_at_center_is_in_fov(self, overlay: AnnotationOverlay) -> None:
        assert overlay.is_in_fov(180.0, 45.0) is True

    def test_object_near_center_is_in_fov(self, overlay: AnnotationOverlay) -> None:
        assert overlay.is_in_fov(180.001, 45.001) is True

    def test_object_far_away_not_in_fov(self, overlay: AnnotationOverlay) -> None:
        assert overlay.is_in_fov(0.0, -89.0) is False

    def test_object_just_outside_fov(self, overlay: AnnotationOverlay) -> None:
        # At dec=45, RA spans are wider by 1/cos(45) ~ 1.41
        import math
        fov_half_deg = 1024 * 0.000277 / (2.0 * math.cos(math.radians(45.0)))
        ra_outside = 180.0 + fov_half_deg + 0.05
        assert overlay.is_in_fov(ra_outside, 45.0) is False

    def test_object_at_edge_is_in_fov(self, overlay: AnnotationOverlay) -> None:
        fov_half_deg = 1024 * 0.000277 / 2.0
        ra_edge = 180.0 + fov_half_deg * 0.9
        assert overlay.is_in_fov(ra_edge, 45.0) is True

    def test_get_visible_objects_filters_correctly(
        self, overlay: AnnotationOverlay, sample_catalog: list[CelestialObject]
    ) -> None:
        visible = overlay.get_visible_objects(sample_catalog)
        visible_names = [obj.name for obj, _ in visible]
        assert "Target-A" in visible_names
        assert "Target-B" in visible_names
        assert "Far-Away" not in visible_names
        assert "M42" not in visible_names

    def test_visible_objects_have_valid_pixel_coords(
        self, overlay: AnnotationOverlay, sample_catalog: list[CelestialObject]
    ) -> None:
        visible = overlay.get_visible_objects(sample_catalog)
        for obj, (px, py) in visible:
            assert 0 <= px < 1024
            assert 0 <= py < 1024

    def test_empty_catalog_returns_empty(self, overlay: AnnotationOverlay) -> None:
        visible = overlay.get_visible_objects([])
        assert visible == []

    def test_all_outside_returns_empty(self, overlay: AnnotationOverlay) -> None:
        far_catalog = [
            CelestialObject(name="South-Pole", ra=0.0, dec=-90.0),
            CelestialObject(name="Antipodal", ra=0.0, dec=-45.0),
        ]
        visible = overlay.get_visible_objects(far_catalog)
        assert visible == []

    def test_wide_field_captures_more_objects(
        self, wide_overlay: AnnotationOverlay
    ) -> None:
        catalog = [
            CelestialObject(name="Center", ra=10.0, dec=20.0),
            CelestialObject(name="Near", ra=10.5, dec=20.5),
            CelestialObject(name="Edge", ra=11.0, dec=21.0),
            CelestialObject(name="Far", ra=50.0, dec=60.0),
        ]
        visible = wide_overlay.get_visible_objects(catalog)
        visible_names = [obj.name for obj, _ in visible]
        assert "Center" in visible_names
        assert "Near" in visible_names
        assert "Far" not in visible_names


# --- FoV Properties Tests ---


class TestFoVProperties:
    def test_fov_center_matches_wcs_center(self, overlay: AnnotationOverlay) -> None:
        center_ra, center_dec = overlay.get_fov_center_world()
        assert center_ra == pytest.approx(180.0, abs=0.01)
        assert center_dec == pytest.approx(45.0, abs=0.01)

    def test_fov_corners_span_expected_area(self, overlay: AnnotationOverlay) -> None:
        import math
        corners = overlay.get_fov_corners_world()
        ras = [c[0] for c in corners]
        decs = [c[1] for c in corners]
        ra_span = max(ras) - min(ras)
        dec_span = max(decs) - min(decs)
        expected_dec_span = 1024 * 0.000277
        # RA span is larger at higher dec due to 1/cos(dec) projection
        expected_ra_span = expected_dec_span / math.cos(math.radians(45.0))
        assert ra_span == pytest.approx(expected_ra_span, rel=0.15)
        assert dec_span == pytest.approx(expected_dec_span, rel=0.1)

    def test_image_shape_property(self, overlay: AnnotationOverlay) -> None:
        assert overlay.image_shape == (1024, 1024)

    def test_wcs_property(self, overlay: AnnotationOverlay, standard_wcs: WCS) -> None:
        assert overlay.wcs is standard_wcs
