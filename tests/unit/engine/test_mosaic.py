"""Unit tests for MosaicEngine and MosaicStep."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from astroai.engine.mosaic.engine import (
    GradientCorrector,
    MosaicConfig,
    MosaicEngine,
    MosaicPanel,
    MosaicStitcher,
    OverlapDetector,
    OverlapInfo,
    PanelSolver,
    _inside,
    _intersect,
    _pixel_scale_from_wcs,
    _polygon_area,
    _sutherland_hodgman,
)
from astroai.engine.mosaic.pipeline_step import MosaicStep
from astroai.core.pipeline.base import PipelineContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wcs(ra: float = 10.0, dec: float = 20.0, scale: float = 0.000277) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [64.0, 64.0]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-scale, scale]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (128, 128)
    return w


def _make_panel(ra: float = 10.0, dec: float = 20.0) -> MosaicPanel:
    img = np.ones((128, 128), dtype=np.float32) * 0.5
    return MosaicPanel(path=Path("dummy.fits"), image=img, wcs=_make_wcs(ra, dec))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestPolygonArea:
    def test_unit_square(self) -> None:
        verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert _polygon_area(verts) == pytest.approx(1.0)

    def test_triangle(self) -> None:
        verts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        assert _polygon_area(verts) == pytest.approx(2.0)

    def test_degenerate_fewer_than_three_points(self) -> None:
        verts = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert _polygon_area(verts) == 0.0


class TestInsideEdge:
    def test_point_on_left_of_edge(self) -> None:
        assert _inside((0.5, 0.5), (0.0, 0.0), (1.0, 0.0)) is True

    def test_point_on_right_of_edge(self) -> None:
        assert _inside((0.5, -0.5), (0.0, 0.0), (1.0, 0.0)) is False


class TestIntersect:
    def test_perpendicular_lines(self) -> None:
        pt = _intersect((0.0, 0.0), (1.0, 1.0), (0.5, 0.0), (0.5, 1.0))
        assert pt is not None
        assert pt[0] == pytest.approx(0.5)
        assert pt[1] == pytest.approx(0.5)

    def test_parallel_lines_returns_none(self) -> None:
        pt = _intersect((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        assert pt is None


class TestSutherlandHodgman:
    def test_square_clip_smaller_square(self) -> None:
        subject = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        clip = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
        result = _sutherland_hodgman(subject, clip)
        assert len(result) >= 4  # clipped polygon has vertices
        poly = np.array(result)
        area = _polygon_area(poly)
        assert area == pytest.approx(1.0, abs=0.05)

    def test_non_overlapping_returns_empty(self) -> None:
        subject = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        clip = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]])
        result = _sutherland_hodgman(subject, clip)
        assert result == []


class TestPixelScaleFromWCS:
    def test_cdelt_based_wcs(self) -> None:
        w = _make_wcs(scale=0.000277)
        scale = _pixel_scale_from_wcs(w)
        assert scale == pytest.approx(0.000277, rel=0.01)

    def test_cd_matrix_based_wcs(self) -> None:
        w = WCS(naxis=2)
        w.wcs.crpix = [64.0, 64.0]
        w.wcs.crval = [10.0, 20.0]
        w.wcs.cd = np.array([[-0.0003, 0.0], [0.0, 0.0003]])
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        scale = _pixel_scale_from_wcs(w)
        assert scale == pytest.approx(0.0003, rel=0.05)


# ---------------------------------------------------------------------------
# OverlapDetector
# ---------------------------------------------------------------------------

class TestOverlapDetector:
    def test_no_overlap_returns_empty(self) -> None:
        detector = OverlapDetector()
        p1 = _make_panel(ra=10.0, dec=20.0)
        p2 = _make_panel(ra=30.0, dec=40.0)  # far apart
        overlaps = detector.build_overlap_graph([p1, p2])
        assert overlaps == {}

    def test_single_panel_no_pairs(self) -> None:
        detector = OverlapDetector()
        panel = _make_panel()
        overlaps = detector.build_overlap_graph([panel])
        assert overlaps == {}

    def test_identical_panels_overlap(self) -> None:
        detector = OverlapDetector()
        p1 = _make_panel(ra=10.0, dec=20.0)
        p2 = _make_panel(ra=10.0, dec=20.0)
        overlaps = detector.build_overlap_graph([p1, p2])
        assert len(overlaps) == 1
        info = overlaps[(0, 1)]
        assert info.panel_a == 0
        assert info.panel_b == 1
        assert info.overlap_fraction == pytest.approx(1.0, abs=0.01)

    def test_compute_footprints_shape(self) -> None:
        detector = OverlapDetector()
        panels = [_make_panel(), _make_panel(ra=10.1)]
        footprints = detector.compute_footprints(panels)
        assert len(footprints) == 2
        for fp in footprints:
            assert fp.shape == (4, 2)


# ---------------------------------------------------------------------------
# GradientCorrector
# ---------------------------------------------------------------------------

class TestGradientCorrector:
    def test_single_panel_passthrough(self) -> None:
        gc = GradientCorrector()
        img = np.ones((16, 16), dtype=np.float32)
        fp = np.ones((16, 16), dtype=bool)
        result = gc.correct([img], [fp])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], img)

    def test_two_panels_with_overlap_equalized(self) -> None:
        gc = GradientCorrector()
        img_a = np.ones((16, 16), dtype=np.float64) * 1.0
        img_b = np.ones((16, 16), dtype=np.float64) * 0.5
        fp_both = np.ones((16, 16), dtype=bool)
        result = gc.correct([img_a, img_b], [fp_both, fp_both])
        assert len(result) == 2
        # After correction the medians in the overlap should be closer
        med_a = float(np.median(result[0]))
        med_b = float(np.median(result[1]))
        assert abs(med_a - med_b) < abs(1.0 - 0.5)

    def test_no_overlap_returns_unchanged(self) -> None:
        gc = GradientCorrector()
        img_a = np.ones((16, 16), dtype=np.float64) * 2.0
        img_b = np.ones((16, 16), dtype=np.float64) * 1.0
        fp_a = np.zeros((16, 16), dtype=bool)
        fp_a[:, :8] = True
        fp_b = np.zeros((16, 16), dtype=bool)
        fp_b[:, 8:] = True
        result = gc.correct([img_a, img_b], [fp_a, fp_b])
        np.testing.assert_array_equal(result[0], img_a)
        np.testing.assert_array_equal(result[1], img_b)


# ---------------------------------------------------------------------------
# MosaicStitcher
# ---------------------------------------------------------------------------

def _fake_reproject(input_data: object, output_wcs: object, output_shape: tuple) -> tuple:
    return (
        np.ones(output_shape, dtype=np.float64) * 0.5,
        np.ones(output_shape, dtype=np.float64),
    )


class TestMosaicStitcher:
    def test_stitch_single_panel_shape(self) -> None:
        stitcher = MosaicStitcher()
        panel = _make_panel()
        out_wcs = _make_wcs()
        out_shape = (64, 64)

        with patch("reproject.reproject_interp", _fake_reproject):
            result = stitcher.stitch([panel], out_wcs, out_shape, gradient_correct=False)

        assert result.shape == out_shape
        assert result.dtype == np.float32

    def test_stitch_two_panels_merged(self) -> None:
        stitcher = MosaicStitcher()
        panels = [_make_panel(), _make_panel(ra=10.1)]
        out_wcs = _make_wcs()
        out_shape = (64, 64)

        with patch("reproject.reproject_interp", _fake_reproject):
            result = stitcher.stitch(panels, out_wcs, out_shape, gradient_correct=False)

        assert result.shape == out_shape

    def test_distance_weight_all_false_mask(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        w = MosaicStitcher._distance_weight(mask)
        assert np.all(w == 0.0)

    def test_cosine_weight_all_false_mask(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        w = MosaicStitcher._cosine_weight(mask)
        assert np.all(w == 0.0)

    def test_cosine_weight_max_d_zero_returns_zeros(self) -> None:
        """_cosine_weight returns zeros when distance_transform_edt gives max_d=0 (line 261)."""
        mask = np.ones((4, 4), dtype=bool)
        with patch("scipy.ndimage.distance_transform_edt") as mock_edt:
            mock_edt.return_value = np.zeros((4, 4), dtype=np.float64)
            w = MosaicStitcher._cosine_weight(mask)
        assert np.all(w == 0.0)

    def test_distance_weight_full_mask_normalized(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        w = MosaicStitcher._distance_weight(mask)
        assert float(w.max()) == pytest.approx(1.0, abs=0.01)

    def test_feather_blend_mode(self) -> None:
        stitcher = MosaicStitcher()
        panel = _make_panel()
        out_wcs = _make_wcs()
        out_shape = (32, 32)

        with patch("reproject.reproject_interp", _fake_reproject):
            result = stitcher.stitch([panel], out_wcs, out_shape, blend_mode="feather", gradient_correct=False)

        assert result.shape == out_shape


# ---------------------------------------------------------------------------
# MosaicEngine
# ---------------------------------------------------------------------------

class TestMosaicEngine:
    def test_default_config(self) -> None:
        with patch("astroai.engine.mosaic.engine.PlateSolver"):
            engine = MosaicEngine()
        assert engine.config.blend_mode == "linear"
        assert engine.config.gradient_correct is True
        assert engine.config.output_scale == pytest.approx(1.0)

    def test_custom_config(self) -> None:
        cfg = MosaicConfig(blend_mode="feather", gradient_correct=False, output_scale=2.0)
        with patch("astroai.engine.mosaic.engine.PlateSolver"):
            engine = MosaicEngine(config=cfg)
        assert engine.config.blend_mode == "feather"
        assert engine.config.output_scale == pytest.approx(2.0)

    def test_compute_output_wcs_two_panels(self) -> None:
        with patch("astroai.engine.mosaic.engine.PlateSolver"):
            engine = MosaicEngine()
        p1 = _make_panel(ra=10.0, dec=20.0)
        p2 = _make_panel(ra=10.1, dec=20.0)
        out_wcs, out_shape = engine.compute_output_wcs([p1, p2])
        ny, nx = out_shape
        assert nx > 0 and ny > 0

    def test_compute_output_wcs_single_panel(self) -> None:
        with patch("astroai.engine.mosaic.engine.PlateSolver"):
            engine = MosaicEngine()
        p = _make_panel()
        out_wcs, out_shape = engine.compute_output_wcs([p])
        ny, nx = out_shape
        assert nx > 0 and ny > 0


# ---------------------------------------------------------------------------
# MosaicStep
# ---------------------------------------------------------------------------

class TestMosaicStep:
    def test_name(self) -> None:
        step = MosaicStep()
        assert step.name == "Mosaic"

    def test_stage(self) -> None:
        from astroai.core.pipeline.base import PipelineStage
        step = MosaicStep()
        assert step.stage == PipelineStage.MOSAIC

    def test_skips_when_no_panel_paths(self) -> None:
        step = MosaicStep()
        ctx = PipelineContext()
        result = step.execute(ctx)
        assert result is ctx  # unchanged

    def test_skips_empty_panel_paths_list(self) -> None:
        step = MosaicStep()
        ctx = PipelineContext()
        ctx.metadata["panel_paths"] = []
        result = step.execute(ctx)
        assert result is ctx

    def test_resolve_panel_paths_from_panel_paths_key(self) -> None:
        ctx = PipelineContext()
        ctx.metadata["panel_paths"] = ["/a/b.fits", "/c/d.fits"]
        paths = MosaicStep._resolve_panel_paths(ctx)
        assert len(paths) == 2
        assert paths[0] == Path("/a/b.fits")

    def test_resolve_panel_paths_from_mosaic_panels_key(self) -> None:
        ctx = PipelineContext()
        ctx.metadata["mosaic_panels"] = ["/x/y.fits"]
        paths = MosaicStep._resolve_panel_paths(ctx)
        assert len(paths) == 1

    def test_execute_calls_engine_and_stores_output_path(self) -> None:
        step = MosaicStep(output_path=Path("test_out.fits"))
        ctx = PipelineContext()
        ctx.metadata["panel_paths"] = [Path("a.fits"), Path("b.fits")]

        fake_result = Path("test_out.fits")
        with patch("astroai.engine.mosaic.pipeline_step.MosaicEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.stitch.return_value = fake_result
            result = step.execute(ctx)

        assert result.metadata["mosaic_output_path"] == fake_result
        instance.stitch.assert_called_once()


# ---------------------------------------------------------------------------
# PanelSolver
# ---------------------------------------------------------------------------

class TestPanelSolver:
    def _make_mock_plate_solver(self, ra: float = 10.0, dec: float = 20.0) -> MagicMock:
        mock_solver = MagicMock()
        mock_result = MagicMock()
        mock_result.wcs = _make_wcs(ra=ra, dec=dec)
        mock_result.ra_center = ra
        mock_result.dec_center = dec
        mock_solver.solve.return_value = mock_result
        return mock_solver

    def test_solve_all_single_panel(self, tmp_path: Path) -> None:
        """solve_all reads FITS and returns MosaicPanel list (lines 65-79)."""
        fits_path = tmp_path / "panel.fits"
        fits.PrimaryHDU(data=np.ones((64, 64), dtype=np.float32)).writeto(fits_path)

        solver = PanelSolver(plate_solver=self._make_mock_plate_solver())
        panels = solver.solve_all([fits_path])

        assert len(panels) == 1
        assert panels[0].path == fits_path
        assert panels[0].image.dtype == np.float32

    def test_solve_all_updates_ra_dec_hint(self, tmp_path: Path) -> None:
        """solve_all passes ra/dec from previous panel as hint to next (lines 77-78)."""
        p1 = tmp_path / "p1.fits"
        p2 = tmp_path / "p2.fits"
        for fp in (p1, p2):
            fits.PrimaryHDU(data=np.ones((32, 32), dtype=np.float32)).writeto(fp)

        call_kwargs: list[dict] = []

        def _fake_solve(path, ra_hint=None, dec_hint=None):
            call_kwargs.append({"ra_hint": ra_hint, "dec_hint": dec_hint})
            m = MagicMock()
            m.wcs = _make_wcs(ra=15.0, dec=25.0)
            m.ra_center = 15.0
            m.dec_center = 25.0
            return m

        mock_solver = MagicMock()
        mock_solver.solve.side_effect = _fake_solve
        solver = PanelSolver(plate_solver=mock_solver)
        solver.solve_all([p1, p2], ra_hint=5.0, dec_hint=10.0)

        assert call_kwargs[0] == {"ra_hint": 5.0, "dec_hint": 10.0}
        assert call_kwargs[1] == {"ra_hint": 15.0, "dec_hint": 25.0}


# ---------------------------------------------------------------------------
# MosaicStitcher — additional coverage
# ---------------------------------------------------------------------------

class TestMosaicStitcherAdditional:
    def test_stitch_with_gradient_correct_two_panels(self) -> None:
        """gradient_correct=True with 2+ panels triggers GradientCorrector (line 197)."""
        stitcher = MosaicStitcher()
        panels = [_make_panel(), _make_panel(ra=10.1)]
        out_wcs = _make_wcs()
        out_shape = (32, 32)

        with patch("reproject.reproject_interp", _fake_reproject):
            result = stitcher.stitch(panels, out_wcs, out_shape, gradient_correct=True)

        assert result.shape == out_shape
        assert result.dtype == np.float32

    def test_write_fits_creates_file(self, tmp_path: Path) -> None:
        """write_fits writes mosaic to FITS with WCS header (lines 270-276)."""
        mosaic = np.ones((32, 32), dtype=np.float32) * 0.5
        wcs = _make_wcs()
        out_path = tmp_path / "mosaic.fits"

        result = MosaicStitcher.write_fits(out_path, mosaic, wcs, n_panels=2)

        assert result == out_path
        assert out_path.exists()
        with fits.open(str(out_path)) as hdul:
            assert hdul[0].header["NPANELS"] == 2
            assert hdul[0].header.get("MOSAIC") is True


# ---------------------------------------------------------------------------
# MosaicEngine — property accessors and high-level methods
# ---------------------------------------------------------------------------

class TestMosaicEngineHighLevel:
    def test_property_accessors(self) -> None:
        """panel_solver, overlap_detector, stitcher properties (lines 296, 300, 304)."""
        with patch("astroai.engine.mosaic.engine.PlateSolver"):
            engine = MosaicEngine()
        assert isinstance(engine.panel_solver, PanelSolver)
        assert isinstance(engine.overlap_detector, OverlapDetector)
        assert isinstance(engine.stitcher, MosaicStitcher)

    def test_analyze_panels(self, tmp_path: Path) -> None:
        """analyze_panels delegates to solver + overlap detector (lines 337-345)."""
        fits_path = tmp_path / "p.fits"
        fits.PrimaryHDU(data=np.ones((64, 64), dtype=np.float32)).writeto(fits_path)

        with patch("astroai.engine.mosaic.engine.PlateSolver"):
            engine = MosaicEngine()

        panel = _make_panel()
        with patch.object(engine._panel_solver, "solve_all", return_value=[panel, panel]):
            with patch.object(engine._overlap_detector, "compute_footprints", return_value=[]):
                with patch.object(engine._overlap_detector, "build_overlap_graph", return_value={}):
                    panels, overlaps = engine.analyze_panels([fits_path, fits_path])

        assert len(panels) == 2
        assert overlaps == {}

    def test_stitch_produces_output_file(self, tmp_path: Path) -> None:
        """MosaicEngine.stitch calls subcomponents and writes result (lines 313-329)."""
        fits_path = tmp_path / "p.fits"
        fits.PrimaryHDU(data=np.ones((64, 64), dtype=np.float32)).writeto(fits_path)
        out_path = tmp_path / "mosaic.fits"

        with patch("astroai.engine.mosaic.engine.PlateSolver"):
            engine = MosaicEngine()

        panel = _make_panel()
        out_wcs = _make_wcs()
        mosaic_data = np.ones((32, 32), dtype=np.float32)

        with patch.object(engine, "analyze_panels", return_value=([panel, panel], {})):
            with patch.object(engine, "compute_output_wcs", return_value=(out_wcs, (32, 32))):
                with patch.object(engine._stitcher, "stitch", return_value=mosaic_data):
                    with patch.object(engine._stitcher, "write_fits", return_value=out_path):
                        result = engine.stitch([fits_path, fits_path], out_path)

        assert result == out_path

    def test_execute_uses_default_output_path_from_metadata(self) -> None:
        step = MosaicStep()
        ctx = PipelineContext()
        ctx.metadata["panel_paths"] = [Path("a.fits")]
        ctx.metadata["mosaic_output"] = "custom_mosaic.fits"

        with patch("astroai.engine.mosaic.pipeline_step.MosaicEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.stitch.return_value = Path("custom_mosaic.fits")
            step.execute(ctx)

        call_kwargs = instance.stitch.call_args
        assert call_kwargs.kwargs["output_path"] == Path("custom_mosaic.fits")

    def test_progress_callback_invoked(self) -> None:
        step = MosaicStep(output_path=Path("out.fits"))
        ctx = PipelineContext()
        ctx.metadata["panel_paths"] = [Path("a.fits"), Path("b.fits")]

        calls: list = []
        with patch("astroai.engine.mosaic.pipeline_step.MosaicEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.stitch.return_value = Path("out.fits")
            step.execute(ctx, progress=lambda p: calls.append(p))

        assert len(calls) == 2  # start + done
        assert calls[0].current == 0
        assert calls[-1].current == calls[-1].total

    def test_passes_ra_dec_hints_to_engine(self) -> None:
        step = MosaicStep(output_path=Path("out.fits"))
        ctx = PipelineContext()
        ctx.metadata["panel_paths"] = [Path("a.fits")]
        ctx.metadata["ra_hint"] = 15.0
        ctx.metadata["dec_hint"] = -30.0

        with patch("astroai.engine.mosaic.pipeline_step.MosaicEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.stitch.return_value = Path("out.fits")
            step.execute(ctx)

        call_kwargs = instance.stitch.call_args.kwargs
        assert call_kwargs["ra_hint"] == pytest.approx(15.0)
        assert call_kwargs["dec_hint"] == pytest.approx(-30.0)
