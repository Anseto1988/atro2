"""PSF quality analysis — 2D Gaussian fit, FWHM, ellipticity, Strehl estimation."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, optimize

__all__ = ["StarMetrics", "FrameAnalysisResult", "StarAnalyzer"]

_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548
_PSF_HALF_SIZE = 12  # patch radius for PSF fitting
_MAX_SIGMA_PX = 20.0
_MIN_SIGMA_PX = 0.3


@dataclass
class StarMetrics:
    """PSF metrics for a single detected star."""

    x: float
    y: float
    fwhm_x: float        # pixels (major / x axis)
    fwhm_y: float        # pixels (minor / y axis)
    fwhm: float          # geometric mean FWHM
    ellipticity: float   # 0 = perfectly round, 1 = fully elongated
    strehl: float        # 0..1 relative PSF quality proxy
    flux: float
    fit_residual: float  # RMS fit residual (lower = better)


@dataclass
class FrameAnalysisResult:
    """Aggregate PSF analysis for a full frame."""

    stars: list[StarMetrics] = field(default_factory=list)
    median_fwhm: float = 0.0
    p90_fwhm: float = 0.0
    median_ellipticity: float = 0.0
    median_strehl: float = 0.0
    exceeds_fwhm_limit: bool = False
    hfr_cross_val_delta: float = 0.0  # |FWHM/2 − HFR|

    @property
    def star_count(self) -> int:
        return len(self.stars)


class StarAnalyzer:
    """Per-star 2D Gaussian PSF fitting for astrophotography quality metrics.

    Detects stars via connected-component analysis, fits 2D Gaussians to each
    PSF patch, and computes FWHM, ellipticity, and a Strehl proxy. Supports
    CSV export and cross-validation against FrameScorer HFR values.
    """

    def __init__(
        self,
        detection_sigma: float = 4.0,
        min_star_area: int = 4,
        max_star_area: int = 2000,
        fwhm_limit: float = 0.0,   # 0 = threshold disabled
        max_stars: int = 200,
    ) -> None:
        self._sigma = detection_sigma
        self._min_area = min_star_area
        self._max_area = max_star_area
        self._fwhm_limit = fwhm_limit
        self._max_stars = max_stars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        frame: NDArray[np.floating[Any]],
        hfr: float | None = None,
    ) -> FrameAnalysisResult:
        """Analyse PSF quality of *frame*. Optional *hfr* enables cross-validation."""
        gray = self._to_grayscale(frame)
        candidates = self._detect_stars(gray)
        stars: list[StarMetrics] = []
        for cy, cx, flux in candidates[: self._max_stars]:
            m = self._fit_psf(gray, cy, cx, flux)
            if m is not None:
                stars.append(m)
        result = self._aggregate(stars)
        if hfr is not None and result.median_fwhm > 0:
            result.hfr_cross_val_delta = abs(result.median_fwhm / 2.0 - hfr)
        return result

    def to_csv(self, result: FrameAnalysisResult) -> str:
        """Return per-star metrics as a CSV-formatted string."""
        fields = [
            "x", "y", "fwhm_x", "fwhm_y", "fwhm",
            "ellipticity", "strehl", "flux", "fit_residual",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for m in result.stars:
            writer.writerow({f: round(getattr(m, f), 4) for f in fields})
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Star detection
    # ------------------------------------------------------------------

    def _detect_stars(
        self, gray: NDArray[np.floating[Any]]
    ) -> list[tuple[int, int, float]]:
        mean, std = float(gray.mean()), float(gray.std())
        if std < 1e-8:
            return []
        mask = gray > (mean + self._sigma * std)
        labeled, n = ndimage.label(mask)
        candidates: list[tuple[int, int, float]] = []
        for i in range(1, n + 1):
            region = labeled == i
            area = int(region.sum())
            if area < self._min_area or area > self._max_area:
                continue
            ys, xs = np.where(region)
            dy = int(ys.max() - ys.min()) + 1
            dx = int(xs.max() - xs.min()) + 1
            aspect = max(dy, dx) / max(min(dy, dx), 1)
            if aspect > 4.0:
                continue
            flux = float(gray[region].sum())
            cy = int(round(float(np.average(ys, weights=gray[ys, xs]))))
            cx = int(round(float(np.average(xs, weights=gray[ys, xs]))))
            candidates.append((cy, cx, flux))
        candidates.sort(key=lambda c: c[2], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # 2-D Gaussian PSF fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_2d(
        xy: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
        amplitude: float,
        x0: float,
        y0: float,
        sigma_x: float,
        sigma_y: float,
        theta: float,
        background: float,
    ) -> NDArray[np.floating[Any]]:
        x, y = xy
        ct, st = np.cos(theta), np.sin(theta)
        a = ct**2 / (2.0 * sigma_x**2) + st**2 / (2.0 * sigma_y**2)
        b = -np.sin(2.0 * theta) / (4.0 * sigma_x**2) + np.sin(2.0 * theta) / (4.0 * sigma_y**2)
        c = st**2 / (2.0 * sigma_x**2) + ct**2 / (2.0 * sigma_y**2)
        dx, dy = x - x0, y - y0
        return (  # type: ignore[return-value]
            amplitude * np.exp(-(a * dx**2 + 2.0 * b * dx * dy + c * dy**2)) + background
        )

    def _fit_psf(
        self,
        gray: NDArray[np.floating[Any]],
        cy: int,
        cx: int,
        flux: float,
    ) -> StarMetrics | None:
        h, w = gray.shape
        r = _PSF_HALF_SIZE
        y0 = max(cy - r, 0)
        y1 = min(cy + r + 1, h)
        x0 = max(cx - r, 0)
        x1 = min(cx + r + 1, w)
        if (y1 - y0) < 5 or (x1 - x0) < 5:
            return None

        patch = gray[y0:y1, x0:x1].astype(np.float64)
        yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]

        bg = float(np.percentile(patch, 10))
        amp_guess = float(patch.max() - bg)
        if amp_guess < 1e-8:
            return None

        local_cy = cy - y0
        local_cx = cx - x0
        p0 = [amp_guess, float(local_cx), float(local_cy), 2.0, 2.0, 0.0, bg]
        lo = [0.0,     0.0,              0.0,              _MIN_SIGMA_PX, _MIN_SIGMA_PX, -np.pi / 2, -np.inf]
        hi = [np.inf,  float(patch.shape[1]), float(patch.shape[0]), _MAX_SIGMA_PX, _MAX_SIGMA_PX,  np.pi / 2,  np.inf]

        try:
            popt, _ = optimize.curve_fit(
                self._gaussian_2d,
                (xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)),
                patch.ravel(),
                p0=p0,
                bounds=(lo, hi),
                maxfev=800,
            )
        except (RuntimeError, ValueError):
            return None

        amp, fit_cx, fit_cy, sx, sy, _theta, bg_fit = popt
        if amp < 1e-8:
            return None

        fwhm_x = float(_FWHM_FACTOR * sx)
        fwhm_y = float(_FWHM_FACTOR * sy)
        fwhm = float(np.sqrt(fwhm_x * fwhm_y))
        ellipticity = float(1.0 - min(sx, sy) / max(sx, sy))

        # Strehl proxy: ratio of measured peak to fitted amplitude (1.0 = ideal Gaussian)
        actual_peak = float(patch.max()) - float(bg_fit)
        strehl = float(np.clip(actual_peak / max(amp, 1e-8), 0.0, 1.0))

        fit_vals = self._gaussian_2d(
            (xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)), *popt
        ).reshape(patch.shape)
        residual = float(np.sqrt(np.mean((patch - fit_vals) ** 2)))

        return StarMetrics(
            x=float(x0) + fit_cx,
            y=float(y0) + fit_cy,
            fwhm_x=fwhm_x,
            fwhm_y=fwhm_y,
            fwhm=fwhm,
            ellipticity=ellipticity,
            strehl=strehl,
            flux=flux,
            fit_residual=residual,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, stars: list[StarMetrics]) -> FrameAnalysisResult:
        if not stars:
            return FrameAnalysisResult()
        fwhms = np.array([s.fwhm for s in stars], dtype=np.float64)
        return FrameAnalysisResult(
            stars=stars,
            median_fwhm=float(np.median(fwhms)),
            p90_fwhm=float(np.percentile(fwhms, 90)),
            median_ellipticity=float(np.median([s.ellipticity for s in stars])),
            median_strehl=float(np.median([s.strehl for s in stars])),
            exceeds_fwhm_limit=(
                self._fwhm_limit > 0 and float(np.median(fwhms)) > self._fwhm_limit
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grayscale(
        frame: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        if frame.ndim == 2:
            return frame.astype(np.float64)
        return (
            0.2989 * frame[..., 0]
            + 0.5870 * frame[..., 1]
            + 0.1140 * frame[..., 2]
        ).astype(np.float64)
