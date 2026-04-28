"""Unit tests for AnnotationOverlay and CelestialObject."""
from __future__ import annotations
import pytest
from astropy.wcs import WCS
from astroai.engine.platesolving.annotation import AnnotationOverlay, CelestialObject


def _make_wcs(ra: float = 180.0, dec: float = 45.0, scale: float = 0.000277) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [512.0, 512.0]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-scale, scale]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (1024, 1024)
    return w


class TestCelestialObject:
    def test_fields(self) -> None:
        obj = CelestialObject(name="M42", ra=83.8, dec=-5.4, magnitude=4.0, object_type="Nebula")
        assert obj.name == "M42"
        assert obj.ra == pytest.approx(83.8)
        assert obj.object_type == "Nebula"

    def test_defaults(self) -> None:
        obj = CelestialObject(name="X", ra=0.0, dec=0.0)
        assert obj.magnitude == pytest.approx(0.0)
        assert obj.object_type == "unknown"

    def test_frozen(self) -> None:
        obj = CelestialObject(name="Y", ra=10.0, dec=20.0)
        with pytest.raises(AttributeError):
            obj.name = "Z"  # type: ignore[misc]


class TestAnnotationOverlay:
    def test_wcs_property(self) -> None:
        wcs = _make_wcs()
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        assert overlay.wcs is wcs

    def test_image_shape_property(self) -> None:
        wcs = _make_wcs()
        overlay = AnnotationOverlay(wcs, (600, 800))
        assert overlay.image_shape == (600, 800)

    def test_world_to_pixel_center(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0)
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        x, y = overlay.world_to_pixel(180.0, 45.0)
        assert x == pytest.approx(511.0, abs=2.0)
        assert y == pytest.approx(511.0, abs=2.0)

    def test_pixel_to_world_center(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0)
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        ra, dec = overlay.pixel_to_world(511.0, 511.0)
        assert ra == pytest.approx(180.0, abs=0.1)
        assert dec == pytest.approx(45.0, abs=0.1)

    def test_is_in_fov_center_true(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0)
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        assert overlay.is_in_fov(180.0, 45.0) is True

    def test_is_in_fov_far_away_false(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0)
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        assert overlay.is_in_fov(0.0, -45.0) is False

    def test_get_visible_objects_includes_center(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0)
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        catalog = [
            CelestialObject("Center", ra=180.0, dec=45.0),
            CelestialObject("Far", ra=0.0, dec=-45.0),
        ]
        visible = overlay.get_visible_objects(catalog)
        names = [obj.name for obj, _ in visible]
        assert "Center" in names
        assert "Far" not in names

    def test_get_visible_objects_returns_pixel_coords(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0)
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        catalog = [CelestialObject("X", ra=180.0, dec=45.0)]
        visible = overlay.get_visible_objects(catalog)
        assert len(visible) == 1
        _, (px, py) = visible[0]
        assert isinstance(px, float)
        assert isinstance(py, float)

    def test_get_fov_corners_returns_four(self) -> None:
        wcs = _make_wcs()
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        corners = overlay.get_fov_corners_world()
        assert len(corners) == 4

    def test_get_fov_corners_are_tuples(self) -> None:
        wcs = _make_wcs()
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        for corner in overlay.get_fov_corners_world():
            assert len(corner) == 2

    def test_get_fov_center_world(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0)
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        ra, dec = overlay.get_fov_center_world()
        assert ra == pytest.approx(180.0, abs=0.5)
        assert dec == pytest.approx(45.0, abs=0.5)

    def test_empty_catalog_returns_empty(self) -> None:
        wcs = _make_wcs()
        overlay = AnnotationOverlay(wcs, (1024, 1024))
        assert overlay.get_visible_objects([]) == []
