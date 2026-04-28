"""Mosaic engine: panel solving, overlap detection, and multi-panel assembly."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import NDArray

from astroai.engine.platesolving import PlateSolver, SolveResult

__all__ = [
    "MosaicPanel",
    "MosaicConfig",
    "OverlapInfo",
    "PanelSolver",
    "OverlapDetector",
    "GradientCorrector",
    "MosaicStitcher",
    "MosaicEngine",
]

logger = logging.getLogger(__name__)


@dataclass
class MosaicPanel:
    path: Path
    image: NDArray[np.floating[Any]]
    wcs: WCS
    weight: float = 1.0
    solve_result: SolveResult | None = None


@dataclass
class MosaicConfig:
    blend_mode: str = "linear"
    gradient_correct: bool = True
    output_scale: float = 1.0


@dataclass(frozen=True)
class OverlapInfo:
    panel_a: int
    panel_b: int
    overlap_area_deg2: float
    overlap_fraction: float


class PanelSolver:
    def __init__(self, plate_solver: PlateSolver | None = None) -> None:
        self._solver = plate_solver or PlateSolver()

    def solve_all(
        self,
        panel_paths: list[Path],
        ra_hint: float | None = None,
        dec_hint: float | None = None,
    ) -> list[MosaicPanel]:
        panels: list[MosaicPanel] = []
        for i, path in enumerate(panel_paths):
            logger.info("Solving panel %d/%d: %s", i + 1, len(panel_paths), path.name)
            result = self._solver.solve(path, ra_hint=ra_hint, dec_hint=dec_hint)
            with fits.open(str(path)) as hdul:
                image = hdul[0].data.astype(np.float32)
            panels.append(MosaicPanel(
                path=path,
                image=image,
                wcs=result.wcs,
                solve_result=result,
            ))
            ra_hint = result.ra_center
            dec_hint = result.dec_center
        return panels


class OverlapDetector:
    def compute_footprints(
        self, panels: list[MosaicPanel],
    ) -> list[NDArray[np.float64]]:
        footprints: list[NDArray[np.float64]] = []
        for panel in panels:
            fp: NDArray[np.float64] = panel.wcs.calc_footprint()
            footprints.append(fp)
        return footprints

    def build_overlap_graph(
        self,
        panels: list[MosaicPanel],
        footprints: list[NDArray[np.float64]] | None = None,
    ) -> dict[tuple[int, int], OverlapInfo]:
        if footprints is None:
            footprints = self.compute_footprints(panels)

        overlaps: dict[tuple[int, int], OverlapInfo] = {}
        n = len(panels)
        for i in range(n):
            for j in range(i + 1, n):
                area = self._overlap_area(footprints[i], footprints[j])
                if area <= 0:
                    continue
                area_i = _polygon_area(footprints[i])
                area_j = _polygon_area(footprints[j])
                smaller = min(area_i, area_j)
                fraction = area / smaller if smaller > 0 else 0.0
                overlaps[(i, j)] = OverlapInfo(
                    panel_a=i,
                    panel_b=j,
                    overlap_area_deg2=area,
                    overlap_fraction=fraction,
                )
        return overlaps

    @staticmethod
    def _overlap_area(
        fp_a: NDArray[np.float64],
        fp_b: NDArray[np.float64],
    ) -> float:
        clipped = _sutherland_hodgman(fp_a, fp_b)
        if len(clipped) < 3:
            return 0.0
        return _polygon_area(np.array(clipped))


class GradientCorrector:
    """Per-panel background offset matching in reprojected output space."""

    def correct(
        self,
        reprojected: list[NDArray[np.float64]],
        footprints: list[NDArray[np.bool_]],
    ) -> list[NDArray[np.float64]]:
        n = len(reprojected)
        if n < 2:
            return list(reprojected)

        pairwise: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                overlap = footprints[i] & footprints[j]
                if int(np.sum(overlap)) < 10:
                    continue
                med_i = float(np.nanmedian(reprojected[i][overlap]))
                med_j = float(np.nanmedian(reprojected[j][overlap]))
                pairwise[(i, j)] = med_i - med_j

        if not pairwise:
            return list(reprojected)

        corrections = self._solve_offsets(n, pairwise)
        logger.info(
            "Gradient corrections (per-panel offsets): %s",
            [f"{c:+.2f}" for c in corrections],
        )
        return [data + corrections[k] for k, data in enumerate(reprojected)]

    @staticmethod
    def _solve_offsets(
        n: int, pairwise: dict[tuple[int, int], float],
    ) -> NDArray[np.float64]:
        pairs = list(pairwise.keys())
        m = len(pairs)
        A = np.zeros((m + 1, n))
        b = np.zeros(m + 1)
        for k, (i, j) in enumerate(pairs):
            A[k, i] = 1.0
            A[k, j] = -1.0
            b[k] = -pairwise[(i, j)]
        A[m, 0] = 1.0
        b[m] = 0.0
        return np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float64)


class MosaicStitcher:
    """Reproject panels onto a common WCS grid and blend into a seamless mosaic."""

    def stitch(
        self,
        panels: list[MosaicPanel],
        output_wcs: WCS,
        output_shape: tuple[int, int],
        blend_mode: str = "linear",
        gradient_correct: bool = True,
    ) -> NDArray[np.float32]:
        from reproject import reproject_interp

        reprojected, footprints = self._reproject_all(
            panels, output_wcs, output_shape, reproject_interp,
        )

        if gradient_correct and len(panels) > 1:
            reprojected = GradientCorrector().correct(reprojected, footprints)

        return self._blend(reprojected, footprints, output_shape, blend_mode)

    @staticmethod
    def _reproject_all(
        panels: list[MosaicPanel],
        output_wcs: WCS,
        output_shape: tuple[int, int],
        reproject_fn: Any,
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.bool_]]]:
        reprojected: list[NDArray[np.float64]] = []
        footprints: list[NDArray[np.bool_]] = []
        for i, panel in enumerate(panels):
            logger.info("Reprojecting panel %d/%d: %s", i + 1, len(panels), panel.path.name)
            reproj, fp = reproject_fn(
                (panel.image, panel.wcs.to_header()),
                output_wcs,
                output_shape,
            )
            reprojected.append(np.nan_to_num(reproj, nan=0.0))
            footprints.append(fp > 0.5)
        return reprojected, footprints

    def _blend(
        self,
        reprojected: list[NDArray[np.float64]],
        footprints: list[NDArray[np.bool_]],
        output_shape: tuple[int, int],
        blend_mode: str,
    ) -> NDArray[np.float32]:
        ny, nx = output_shape
        mosaic = np.zeros((ny, nx), dtype=np.float64)
        weight_sum = np.zeros((ny, nx), dtype=np.float64)

        for reproj, fp in zip(reprojected, footprints):
            w = self._cosine_weight(fp) if blend_mode == "feather" else self._distance_weight(fp)
            mosaic += reproj * w
            weight_sum += w

        valid = weight_sum > 0
        mosaic[valid] /= weight_sum[valid]
        return mosaic.astype(np.float32)

    @staticmethod
    def _distance_weight(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        from scipy.ndimage import distance_transform_edt

        if not np.any(mask):
            return np.zeros_like(mask, dtype=np.float64)
        dist: NDArray[np.float64] = distance_transform_edt(mask)
        max_d = float(dist.max())
        return dist / max_d if max_d > 0 else dist

    @staticmethod
    def _cosine_weight(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        from scipy.ndimage import distance_transform_edt

        if not np.any(mask):
            return np.zeros_like(mask, dtype=np.float64)
        dist: NDArray[np.float64] = distance_transform_edt(mask)
        max_d = float(dist.max())
        if max_d > 0:
            return 0.5 * (1.0 - np.cos(np.pi * dist / max_d))
        return np.zeros_like(dist)

    @staticmethod
    def write_fits(
        output_path: Path,
        mosaic: NDArray[np.float32],
        wcs: WCS,
        n_panels: int,
    ) -> Path:
        header = wcs.to_header()
        header["MOSAIC"] = (True, "Multi-panel mosaic image")
        header["NPANELS"] = (n_panels, "Number of input panels")
        hdu = fits.PrimaryHDU(data=mosaic, header=header)
        hdu.writeto(str(output_path), overwrite=True)
        logger.info("Mosaic FITS written: %s", output_path)
        return output_path


class MosaicEngine:
    def __init__(
        self,
        config: MosaicConfig | None = None,
        plate_solver: PlateSolver | None = None,
    ) -> None:
        self._config = config or MosaicConfig()
        self._panel_solver = PanelSolver(plate_solver)
        self._overlap_detector = OverlapDetector()
        self._stitcher = MosaicStitcher()

    @property
    def config(self) -> MosaicConfig:
        return self._config

    @property
    def panel_solver(self) -> PanelSolver:
        return self._panel_solver

    @property
    def overlap_detector(self) -> OverlapDetector:
        return self._overlap_detector

    @property
    def stitcher(self) -> MosaicStitcher:
        return self._stitcher

    def stitch(
        self,
        panel_paths: list[Path],
        output_path: Path,
        ra_hint: float | None = None,
        dec_hint: float | None = None,
    ) -> Path:
        panels, overlaps = self.analyze_panels(panel_paths, ra_hint, dec_hint)
        out_wcs, out_shape = self.compute_output_wcs(panels)

        mosaic = self._stitcher.stitch(
            panels,
            out_wcs,
            out_shape,
            blend_mode=self._config.blend_mode,
            gradient_correct=self._config.gradient_correct,
        )
        self._stitcher.write_fits(output_path, mosaic, out_wcs, len(panels))

        logger.info(
            "Mosaic complete: %dx%d, %d panels -> %s",
            out_shape[1], out_shape[0], len(panels), output_path,
        )
        return output_path

    def analyze_panels(
        self,
        panel_paths: list[Path],
        ra_hint: float | None = None,
        dec_hint: float | None = None,
    ) -> tuple[list[MosaicPanel], dict[tuple[int, int], OverlapInfo]]:
        panels = self._panel_solver.solve_all(panel_paths, ra_hint, dec_hint)
        footprints = self._overlap_detector.compute_footprints(panels)
        overlaps = self._overlap_detector.build_overlap_graph(panels, footprints)
        logger.info(
            "Analyzed %d panels, found %d overlap pairs",
            len(panels),
            len(overlaps),
        )
        return panels, overlaps

    def compute_output_wcs(
        self, panels: list[MosaicPanel],
    ) -> tuple[WCS, tuple[int, int]]:
        all_corners = np.vstack([p.wcs.calc_footprint() for p in panels])

        ra_min, ra_max = float(all_corners[:, 0].min()), float(all_corners[:, 0].max())
        dec_min, dec_max = float(all_corners[:, 1].min()), float(all_corners[:, 1].max())
        ra_center = (ra_min + ra_max) / 2.0
        dec_center = (dec_min + dec_max) / 2.0

        pixel_scale = _pixel_scale_from_wcs(panels[0].wcs)
        pixel_scale /= self._config.output_scale

        cos_dec = np.cos(np.radians(dec_center))
        nx = int(np.ceil((ra_max - ra_min) * cos_dec / pixel_scale)) + 1
        ny = int(np.ceil((dec_max - dec_min) / pixel_scale)) + 1

        out_wcs = WCS(naxis=2)
        out_wcs.wcs.crpix = [nx / 2.0, ny / 2.0]
        out_wcs.wcs.crval = [ra_center, dec_center]
        out_wcs.wcs.cdelt = [-pixel_scale, pixel_scale]
        out_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        out_wcs.pixel_shape = (nx, ny)

        return out_wcs, (ny, nx)


# ---------------------------------------------------------------------------
# Geometry helpers (module-private)
# ---------------------------------------------------------------------------

def _pixel_scale_from_wcs(wcs: WCS) -> float:
    if wcs.wcs.has_cd():
        cd = wcs.wcs.cd
        return float((abs(cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0])) ** 0.5)
    cdelt = wcs.wcs.cdelt
    return float(abs(cdelt[0])) if cdelt[0] != 0 else 0.000277


def _polygon_area(vertices: NDArray[np.float64]) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _sutherland_hodgman(
    subject: NDArray[np.float64],
    clip: NDArray[np.float64],
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = [
        (float(p[0]), float(p[1])) for p in subject
    ]
    n_clip = len(clip)
    for i in range(n_clip):
        if not output:
            return []
        e0 = (float(clip[i][0]), float(clip[i][1]))
        e1 = (float(clip[(i + 1) % n_clip][0]), float(clip[(i + 1) % n_clip][1]))

        inp = output
        output = []
        for j in range(len(inp)):
            cur = inp[j]
            prev = inp[j - 1]
            c_in = _inside(cur, e0, e1)
            p_in = _inside(prev, e0, e1)
            if c_in:
                if not p_in:
                    pt = _intersect(prev, cur, e0, e1)
                    if pt is not None:
                        output.append(pt)
                output.append(cur)
            elif p_in:
                pt = _intersect(prev, cur, e0, e1)
                if pt is not None:
                    output.append(pt)
    return output


def _inside(
    point: tuple[float, float],
    edge_a: tuple[float, float],
    edge_b: tuple[float, float],
) -> bool:
    return (
        (edge_b[0] - edge_a[0]) * (point[1] - edge_a[1])
        - (edge_b[1] - edge_a[1]) * (point[0] - edge_a[0])
    ) >= 0


def _intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    if abs(denom) < 1e-12:
        return None
    t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / denom
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
