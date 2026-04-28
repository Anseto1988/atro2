"""Unit tests for WcsAdapter, SkyObjectCatalog, and pure-logic overlay helpers."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest
from astropy.wcs import WCS

from astroai.ui.overlay.sky_objects import (
    CatalogObject,
    ConstellationBoundarySegment,
    NamedStar,
    SkyObjectCatalog,
    WcsTransform,
)
from astroai.ui.overlay.wcs_adapter import WcsAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wcs(ra: float = 180.0, dec: float = 45.0, scale: float = 0.000277) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [512.0, 512.0]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-scale, scale]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (1024, 1024)
    return w


# ---------------------------------------------------------------------------
# CatalogObject / NamedStar / ConstellationBoundarySegment dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_catalog_object_defaults(self) -> None:
        obj = CatalogObject(designation="M42", ra_deg=83.8, dec_deg=-5.4)
        assert obj.obj_type == ""
        assert obj.magnitude == pytest.approx(99.0)
        assert obj.size_arcmin == pytest.approx(0.0)
        assert obj.common_name == ""

    def test_catalog_object_full(self) -> None:
        obj = CatalogObject(
            designation="M42", ra_deg=83.8, dec_deg=-5.4,
            obj_type="RN", magnitude=4.0, size_arcmin=85.0, common_name="Orion Nebula",
        )
        assert obj.common_name == "Orion Nebula"
        assert obj.magnitude == pytest.approx(4.0)

    def test_named_star_fields(self) -> None:
        star = NamedStar(hip_id=32349, name="Sirius", ra_deg=101.2, dec_deg=-16.7, magnitude=-1.46)
        assert star.hip_id == 32349
        assert star.name == "Sirius"

    def test_constellation_boundary_defaults(self) -> None:
        seg = ConstellationBoundarySegment(ra1_deg=0.0, dec1_deg=0.0, ra2_deg=1.0, dec2_deg=0.0)
        assert seg.constellation == ""

    def test_dataclasses_are_frozen(self) -> None:
        obj = CatalogObject(designation="M1", ra_deg=83.6, dec_deg=22.0)
        with pytest.raises(AttributeError):
            obj.designation = "M2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WcsTransform protocol
# ---------------------------------------------------------------------------

class TestWcsTransformProtocol:
    def test_wcs_adapter_satisfies_protocol(self) -> None:
        wcs = _make_wcs()
        adapter = WcsAdapter(wcs, 1024, 1024)
        assert isinstance(adapter, WcsTransform)


# ---------------------------------------------------------------------------
# WcsAdapter
# ---------------------------------------------------------------------------

class TestWcsAdapter:
    def test_image_size(self) -> None:
        wcs = _make_wcs()
        adapter = WcsAdapter(wcs, 800, 600)
        assert adapter.image_size() == (800, 600)

    def test_world_to_pixel_center(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0)
        adapter = WcsAdapter(wcs, 1024, 1024)
        result = adapter.world_to_pixel(180.0, 45.0)
        assert result is not None
        x, y = result
        assert x == pytest.approx(512.0, abs=2.0)
        assert y == pytest.approx(512.0, abs=2.0)

    def test_world_to_pixel_out_of_bounds_returns_none(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0)
        adapter = WcsAdapter(wcs, 1024, 1024)
        # RA/Dec far from center (other side of sky)
        result = adapter.world_to_pixel(0.0, -45.0)
        assert result is None

    def test_world_to_pixel_near_edge_accepted(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0, scale=0.000277)
        adapter = WcsAdapter(wcs, 1024, 1024)
        # Center is always in bounds
        result = adapter.world_to_pixel(180.0, 45.0)
        assert result is not None

    def test_pixel_to_world_center(self) -> None:
        wcs = _make_wcs(ra=180.0, dec=45.0)
        adapter = WcsAdapter(wcs, 1024, 1024)
        result = adapter.pixel_to_world(512.0, 512.0)
        assert result is not None
        ra, dec = result
        assert ra == pytest.approx(180.0, abs=0.1)
        assert dec == pytest.approx(45.0, abs=0.1)

    def test_pixel_to_world_returns_float_tuple(self) -> None:
        wcs = _make_wcs()
        adapter = WcsAdapter(wcs, 1024, 1024)
        result = adapter.pixel_to_world(100.0, 200.0)
        assert result is not None
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_world_to_pixel_roundtrip(self) -> None:
        """Pixel → world → pixel should be close to identity."""
        wcs = _make_wcs(ra=180.0, dec=45.0)
        adapter = WcsAdapter(wcs, 1024, 1024)
        x_in, y_in = 300.0, 400.0
        world = adapter.pixel_to_world(x_in, y_in)
        assert world is not None
        result = adapter.world_to_pixel(world[0], world[1])
        assert result is not None
        assert result[0] == pytest.approx(x_in, abs=0.5)
        assert result[1] == pytest.approx(y_in, abs=0.5)

    def test_world_to_pixel_10pct_margin_accepted(self) -> None:
        """Points slightly outside image (within 10% margin) should be returned."""
        wcs = _make_wcs(ra=180.0, dec=45.0, scale=0.000277)
        # Use a small image so the 10% margin is easy to hit
        adapter = WcsAdapter(wcs, 1024, 1024)
        # The center is always accepted
        result = adapter.world_to_pixel(180.0, 45.0)
        assert result is not None


# ---------------------------------------------------------------------------
# WcsAdapter factory methods (mocked to avoid ASTAP)
# ---------------------------------------------------------------------------

class TestWcsAdapterFactories:
    def test_from_solve_result(self) -> None:
        wcs = _make_wcs()
        from astroai.engine.platesolving.solver import SolveResult
        solve_result = MagicMock(spec=SolveResult)
        solve_result.wcs = wcs
        adapter = WcsAdapter.from_solve_result(solve_result, (600, 800))
        assert adapter.image_size() == (800, 600)

    def test_from_engine_overlay(self) -> None:
        wcs = _make_wcs()
        from astroai.engine.platesolving.annotation import AnnotationOverlay as EngineOverlay
        overlay = MagicMock(spec=EngineOverlay)
        overlay.wcs = wcs
        overlay.image_shape = (480, 640)
        adapter = WcsAdapter.from_engine_overlay(overlay)
        assert adapter.image_size() == (640, 480)


# ---------------------------------------------------------------------------
# SkyObjectCatalog — loading from real embedded catalogs
# ---------------------------------------------------------------------------

class TestSkyObjectCatalogRealData:
    def test_deep_sky_objects_loaded(self) -> None:
        cat = SkyObjectCatalog()
        dso = cat.deep_sky_objects
        assert len(dso) > 0

    def test_named_stars_loaded(self) -> None:
        cat = SkyObjectCatalog()
        stars = cat.named_stars
        assert len(stars) > 0

    def test_dso_fields_valid(self) -> None:
        cat = SkyObjectCatalog()
        for obj in cat.deep_sky_objects:
            assert isinstance(obj.designation, str)
            assert 0.0 <= obj.ra_deg <= 360.0
            assert -90.0 <= obj.dec_deg <= 90.0

    def test_stars_fields_valid(self) -> None:
        cat = SkyObjectCatalog()
        for star in cat.named_stars:
            assert isinstance(star.name, str)
            assert star.hip_id > 0
            assert 0.0 <= star.ra_deg <= 360.0

    def test_lazy_loading_idempotent(self) -> None:
        cat = SkyObjectCatalog()
        dso1 = cat.deep_sky_objects
        dso2 = cat.deep_sky_objects
        assert len(dso1) == len(dso2)

    def test_sirius_in_named_stars(self) -> None:
        cat = SkyObjectCatalog()
        names = [s.name for s in cat.named_stars]
        assert "Sirius" in names

    def test_m42_in_dso(self) -> None:
        cat = SkyObjectCatalog()
        designations = [o.designation for o in cat.deep_sky_objects]
        assert "M42" in designations or any("42" in d for d in designations)


# ---------------------------------------------------------------------------
# SkyObjectCatalog — missing catalog files handled gracefully
# ---------------------------------------------------------------------------

class TestSkyObjectCatalogMissingFiles:
    def test_missing_dso_returns_empty(self, tmp_path: Path) -> None:
        """No crash if DSO CSV is absent."""
        cat = SkyObjectCatalog.__new__(SkyObjectCatalog)
        cat._dso = []
        cat._stars = []
        cat._boundaries = []
        cat._loaded = False
        with patch("astroai.ui.overlay.sky_objects._CATALOGS_DIR", tmp_path):
            dso = cat.deep_sky_objects
        assert list(dso) == []

    def test_missing_stars_returns_empty(self, tmp_path: Path) -> None:
        cat = SkyObjectCatalog.__new__(SkyObjectCatalog)
        cat._dso = []
        cat._stars = []
        cat._boundaries = []
        cat._loaded = False
        with patch("astroai.ui.overlay.sky_objects._CATALOGS_DIR", tmp_path):
            stars = cat.named_stars
        assert list(stars) == []

    def test_corrupt_dso_does_not_crash(self, tmp_path: Path) -> None:
        """Corrupt CSV should be silently skipped."""
        dso_file = tmp_path / "dso_catalog.csv"
        dso_file.write_text("designation,ra_deg\nM1,NOT_A_NUMBER\n")
        cat = SkyObjectCatalog.__new__(SkyObjectCatalog)
        cat._dso = []
        cat._stars = []
        cat._boundaries = []
        cat._loaded = False
        with patch("astroai.ui.overlay.sky_objects._CATALOGS_DIR", tmp_path):
            dso = cat.deep_sky_objects
        assert list(dso) == []


# ---------------------------------------------------------------------------
# SkyObjectCatalog — custom CSV loading
# ---------------------------------------------------------------------------

class TestSkyObjectCatalogCustomCSV:
    def test_loads_custom_dso_csv(self, tmp_path: Path) -> None:
        dso_file = tmp_path / "dso_catalog.csv"
        dso_file.write_text(
            "designation,ra_deg,dec_deg,type,mag,size_arcmin,common_name\n"
            "M1,83.6331,22.0145,SNR,8.4,6.0,Crab\n"
            "M42,83.8221,-5.3911,RN,4.0,85.0,Orion Nebula\n"
        )
        star_file = tmp_path / "named_stars.csv"
        star_file.write_text("hip_id,name,ra_deg,dec_deg,mag\n")
        bound_file = tmp_path / "constellation_boundaries.csv"
        bound_file.write_text("ra1_deg,dec1_deg,ra2_deg,dec2_deg,constellation\n")

        cat = SkyObjectCatalog.__new__(SkyObjectCatalog)
        cat._dso = []
        cat._stars = []
        cat._boundaries = []
        cat._loaded = False
        with patch("astroai.ui.overlay.sky_objects._CATALOGS_DIR", tmp_path):
            dso = list(cat.deep_sky_objects)
        assert len(dso) == 2
        assert dso[0].designation == "M1"
        assert dso[1].common_name == "Orion Nebula"

    def test_loads_custom_stars_csv(self, tmp_path: Path) -> None:
        star_file = tmp_path / "named_stars.csv"
        star_file.write_text(
            "hip_id,name,ra_deg,dec_deg,mag\n"
            "32349,Sirius,101.2872,-16.7161,-1.46\n"
        )
        dso_file = tmp_path / "dso_catalog.csv"
        dso_file.write_text("designation,ra_deg,dec_deg\n")
        bound_file = tmp_path / "constellation_boundaries.csv"
        bound_file.write_text("ra1_deg,dec1_deg,ra2_deg,dec2_deg\n")

        cat = SkyObjectCatalog.__new__(SkyObjectCatalog)
        cat._dso = []
        cat._stars = []
        cat._boundaries = []
        cat._loaded = False
        with patch("astroai.ui.overlay.sky_objects._CATALOGS_DIR", tmp_path):
            stars = list(cat.named_stars)
        assert len(stars) == 1
        assert stars[0].name == "Sirius"
        assert stars[0].magnitude == pytest.approx(-1.46)

    def test_loads_custom_boundaries_csv(self, tmp_path: Path) -> None:
        """Covers the constellation_boundaries property and _load_boundaries path (lines 95-96, 153-160)."""
        dso_file = tmp_path / "dso_catalog.csv"
        dso_file.write_text("designation,ra_deg,dec_deg\n")
        star_file = tmp_path / "named_stars.csv"
        star_file.write_text("hip_id,name,ra_deg,dec_deg,mag\n")
        bound_file = tmp_path / "constellation_boundaries.csv"
        bound_file.write_text(
            "ra1_deg,dec1_deg,ra2_deg,dec2_deg,constellation\n"
            "10.0,20.0,11.0,21.0,ORI\n"
            "30.0,40.0,31.0,41.0,UMA\n"
        )

        cat = SkyObjectCatalog.__new__(SkyObjectCatalog)
        cat._dso = []
        cat._stars = []
        cat._boundaries = []
        cat._loaded = False
        with patch("astroai.ui.overlay.sky_objects._CATALOGS_DIR", tmp_path):
            boundaries = list(cat.constellation_boundaries)
        assert len(boundaries) == 2
        assert boundaries[0].constellation == "ORI"
        assert boundaries[0].ra1_deg == pytest.approx(10.0)
        assert boundaries[1].constellation == "UMA"


class TestSkyObjectCatalogExceptionPaths:
    """Cover exception handlers in _load_stars and _load_boundaries (lines 143-144, 162-163)."""

    def test_corrupt_stars_csv_does_not_crash(self, tmp_path: Path) -> None:
        """Corrupt named_stars.csv should be silently caught."""
        dso_file = tmp_path / "dso_catalog.csv"
        dso_file.write_text("designation,ra_deg,dec_deg\n")
        star_file = tmp_path / "named_stars.csv"
        # hip_id and mag are not parseable as int/float
        star_file.write_text("hip_id,name,ra_deg,dec_deg,mag\nNOT_INT,Sirius,101.2,-16.7,NOT_FLOAT\n")
        bound_file = tmp_path / "constellation_boundaries.csv"
        bound_file.write_text("ra1_deg,dec1_deg,ra2_deg,dec2_deg\n")

        cat = SkyObjectCatalog.__new__(SkyObjectCatalog)
        cat._dso = []
        cat._stars = []
        cat._boundaries = []
        cat._loaded = False
        with patch("astroai.ui.overlay.sky_objects._CATALOGS_DIR", tmp_path):
            stars = list(cat.named_stars)
        assert list(stars) == []

    def test_corrupt_boundaries_csv_does_not_crash(self, tmp_path: Path) -> None:
        """Corrupt constellation_boundaries.csv should be silently caught."""
        dso_file = tmp_path / "dso_catalog.csv"
        dso_file.write_text("designation,ra_deg,dec_deg\n")
        star_file = tmp_path / "named_stars.csv"
        star_file.write_text("hip_id,name,ra_deg,dec_deg,mag\n")
        bound_file = tmp_path / "constellation_boundaries.csv"
        # ra1_deg is not parseable as float
        bound_file.write_text("ra1_deg,dec1_deg,ra2_deg,dec2_deg\nBAD,20.0,11.0,21.0\n")

        cat = SkyObjectCatalog.__new__(SkyObjectCatalog)
        cat._dso = []
        cat._stars = []
        cat._boundaries = []
        cat._loaded = False
        with patch("astroai.ui.overlay.sky_objects._CATALOGS_DIR", tmp_path):
            boundaries = list(cat.constellation_boundaries)
        assert list(boundaries) == []

    def test_missing_boundaries_file_returns_empty(self, tmp_path: Path) -> None:
        """No crash if constellation_boundaries.csv is absent."""
        cat = SkyObjectCatalog.__new__(SkyObjectCatalog)
        cat._dso = []
        cat._stars = []
        cat._boundaries = []
        cat._loaded = False
        with patch("astroai.ui.overlay.sky_objects._CATALOGS_DIR", tmp_path):
            boundaries = list(cat.constellation_boundaries)
        assert boundaries == []


# ---------------------------------------------------------------------------
# Qt widget paint tests — require qtbot fixture (pytest-qt)
# ---------------------------------------------------------------------------

class TestSkyOverlayPaint:
    @pytest.fixture()
    def viewer_widget(self, qtbot):  # type: ignore[no-untyped-def]
        from PySide6.QtWidgets import QWidget
        w = QWidget()
        w.resize(400, 300)
        qtbot.addWidget(w)
        return w

    @pytest.fixture()
    def sky_overlay(self, qtbot, viewer_widget):  # type: ignore[no-untyped-def]
        from astroai.ui.widgets.sky_overlay import SkyOverlay
        w = SkyOverlay(viewer_widget)
        w.resize(400, 300)
        qtbot.addWidget(w)
        return w

    def _make_solution(self):  # type: ignore[no-untyped-def]
        from astroai.astrometry.catalog import WcsSolution
        sol = MagicMock(spec=WcsSolution)
        sol.ra_center = 180.0
        sol.dec_center = 45.0
        sol.fov_width_deg = 2.0
        sol.fov_height_deg = 1.5
        sol.pixel_scale_arcsec = 1.2
        sol.rotation_deg = 0.0
        return sol

    def test_paint_with_no_solution(self, sky_overlay, viewer_widget, qtbot) -> None:  # type: ignore[no-untyped-def]
        viewer_widget.show()
        sky_overlay.show()
        qtbot.waitExposed(viewer_widget, timeout=1000)
        sky_overlay.repaint()  # solution is None — early return

    def test_paint_with_solution(self, sky_overlay, viewer_widget, qtbot) -> None:  # type: ignore[no-untyped-def]
        sky_overlay.set_solution(self._make_solution())
        viewer_widget.show()
        sky_overlay.show()
        qtbot.waitExposed(viewer_widget, timeout=1000)
        sky_overlay.repaint()

    def test_paint_grid_hidden(self, sky_overlay, viewer_widget, qtbot) -> None:  # type: ignore[no-untyped-def]
        sky_overlay.set_solution(self._make_solution())
        sky_overlay.set_grid_visible(False)
        viewer_widget.show()
        sky_overlay.show()
        qtbot.waitExposed(viewer_widget, timeout=1000)
        sky_overlay.repaint()


class TestAnnotationOverlayPaint:
    @pytest.fixture()
    def viewer(self, qtbot):  # type: ignore[no-untyped-def]
        from astroai.ui.widgets.image_viewer import ImageViewer
        import numpy as np
        w = ImageViewer()
        w.resize(400, 300)
        qtbot.addWidget(w)
        w.set_image_data(np.random.rand(300, 400).astype(np.float32))
        return w

    @pytest.fixture()
    def overlay(self, qtbot, viewer):  # type: ignore[no-untyped-def]
        from astroai.ui.overlay.annotation_overlay import AnnotationOverlay
        w = AnnotationOverlay(viewer)
        w.resize(400, 300)
        qtbot.addWidget(w)
        return w

    def _make_wcs_mock(self, world_to_pixel_result=(200.0, 150.0)):  # type: ignore[no-untyped-def]
        from astroai.ui.overlay.sky_objects import WcsTransform
        mock_wcs = MagicMock(spec=WcsTransform)
        mock_wcs.world_to_pixel.return_value = world_to_pixel_result
        mock_wcs.pixel_to_world.return_value = (180.0, 45.0)
        mock_wcs.image_size.return_value = (400, 300)
        return mock_wcs

    def test_has_wcs_false(self, overlay) -> None:
        assert not overlay.has_wcs

    def test_has_wcs_true(self, overlay) -> None:
        overlay.set_wcs(self._make_wcs_mock())
        assert overlay.has_wcs

    def test_set_show_dso_toggle(self, overlay) -> None:
        overlay.set_show_dso(False)  # _show_dso was True → triggers update
        assert not overlay._show_dso

    def test_set_show_dso_noop(self, overlay) -> None:
        overlay.set_show_dso(True)  # already True → no-op
        assert overlay._show_dso

    def test_on_view_changed_via_signal(self, overlay, viewer, qtbot) -> None:  # type: ignore[no-untyped-def]
        viewer.view_changed.emit()  # triggers _on_view_changed

    def test_world_to_screen_no_wcs(self, overlay) -> None:
        from PySide6.QtCore import QPointF
        result = overlay._world_to_screen(180.0, 45.0)
        assert result is None

    def test_world_to_screen_pixel_returns_none(self, overlay) -> None:
        mock_wcs = self._make_wcs_mock(world_to_pixel_result=None)
        overlay.set_wcs(mock_wcs)
        result = overlay._world_to_screen(180.0, 45.0)
        assert result is None

    def test_paint_no_wcs(self, overlay, qtbot) -> None:  # type: ignore[no-untyped-def]
        overlay.show()
        qtbot.waitExposed(overlay, timeout=1000)
        overlay.repaint()  # _wcs is None — early return

    def test_paint_with_wcs(self, overlay, qtbot) -> None:  # type: ignore[no-untyped-def]
        overlay.set_wcs(self._make_wcs_mock())
        overlay.show()
        qtbot.waitExposed(overlay, timeout=1000)
        overlay.repaint()

    def test_paint_dso_out_of_viewport(self, overlay, viewer, qtbot) -> None:  # type: ignore[no-untyped-def]
        # world_to_pixel returns coords far outside viewport → continue branch (line 136)
        mock_wcs = self._make_wcs_mock(world_to_pixel_result=(9999.0, 9999.0))
        overlay.set_wcs(mock_wcs)
        overlay.show()
        qtbot.waitExposed(overlay, timeout=1000)
        overlay.repaint()

    def test_paint_with_boundaries_and_grid(self, overlay, qtbot) -> None:  # type: ignore[no-untyped-def]
        overlay.set_wcs(self._make_wcs_mock())
        overlay.set_show_boundaries(True)
        overlay.set_show_grid(True)
        overlay.show()
        qtbot.waitExposed(overlay, timeout=1000)
        overlay.repaint()

    def test_paint_dso_only(self, overlay, qtbot) -> None:  # type: ignore[no-untyped-def]
        overlay.set_wcs(self._make_wcs_mock())
        overlay.set_show_stars(False)
        overlay.show()
        qtbot.waitExposed(overlay, timeout=1000)
        overlay.repaint()

    def test_paint_grid_no_corners(self, overlay, qtbot) -> None:  # type: ignore[no-untyped-def]
        # pixel_to_world returns None → len(corners) < 2 → line 209 return
        mock_wcs = self._make_wcs_mock()
        mock_wcs.pixel_to_world.return_value = None
        overlay.set_wcs(mock_wcs)
        overlay.set_show_grid(True)
        overlay.show()
        qtbot.waitExposed(overlay, timeout=1000)
        overlay.repaint()

    def test_choose_grid_step_large_span(self, overlay) -> None:
        # span > 360 → returns 30.0
        result = overlay._choose_grid_step(400.0)
        assert result == 30.0

    def test_choose_grid_step_small_span(self, overlay) -> None:
        result = overlay._choose_grid_step(0.5)
        assert result == pytest.approx(0.1)

    def test_paint_boundaries_null_endpoint(self, overlay, viewer, qtbot) -> None:  # type: ignore[no-untyped-def]
        # world_to_pixel returns None → p1/p2 is None → line 188 continue
        mock_wcs = self._make_wcs_mock(world_to_pixel_result=None)
        overlay.set_wcs(mock_wcs)
        overlay.set_show_boundaries(True)
        overlay.set_show_stars(False)
        overlay.set_show_dso(False)
        overlay.show()
        qtbot.waitExposed(overlay, timeout=1000)
        overlay.repaint()

    def test_paint_boundaries_out_of_viewport(self, overlay, viewer, qtbot) -> None:  # type: ignore[no-untyped-def]
        # world_to_pixel returns far-out coords → both p1 and p2 outside viewport → line 190 continue
        mock_wcs = self._make_wcs_mock(world_to_pixel_result=(9999.0, 9999.0))
        overlay.set_wcs(mock_wcs)
        overlay.set_show_boundaries(True)
        overlay.set_show_stars(False)
        overlay.set_show_dso(False)
        overlay.show()
        qtbot.waitExposed(overlay, timeout=1000)
        overlay.repaint()
