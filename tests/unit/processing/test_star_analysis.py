"""Unit tests for astroai.processing.stars.star_analysis."""
from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from astroai.processing.stars.star_analysis import (
    FrameAnalysisResult,
    StarAnalyzer,
    StarMetrics,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _gaussian_star(
    img: np.ndarray,
    cy: int,
    cx: int,
    amplitude: float = 0.8,
    sigma: float = 2.0,
) -> None:
    """Paint a synthetic 2-D Gaussian star onto *img* in-place."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    img += amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))


def _make_frame(
    size: int = 80,
    n_stars: int = 5,
    sigma: float = 2.0,
    noise: float = 0.005,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros((size, size), dtype=np.float64)
    margin = 15
    for _ in range(n_stars):
        cy = int(rng.integers(margin, size - margin))
        cx = int(rng.integers(margin, size - margin))
        _gaussian_star(frame, cy, cx, amplitude=rng.uniform(0.5, 0.9), sigma=sigma)
    frame += rng.normal(0, noise, frame.shape)
    return np.clip(frame, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# StarMetrics dataclass
# ──────────────────────────────────────────────────────────────────────────────


def test_star_metrics_fields() -> None:
    m = StarMetrics(x=10.0, y=20.0, fwhm_x=4.0, fwhm_y=3.5, fwhm=3.74,
                    ellipticity=0.1, strehl=0.9, flux=100.0, fit_residual=0.01)
    assert m.x == 10.0
    assert m.fwhm > 0


# ──────────────────────────────────────────────────────────────────────────────
# FrameAnalysisResult
# ──────────────────────────────────────────────────────────────────────────────


def test_frame_result_empty() -> None:
    r = FrameAnalysisResult()
    assert r.star_count == 0
    assert r.median_fwhm == 0.0
    assert not r.exceeds_fwhm_limit


def test_frame_result_star_count() -> None:
    m = StarMetrics(x=0, y=0, fwhm_x=4, fwhm_y=4, fwhm=4,
                    ellipticity=0, strehl=1, flux=1, fit_residual=0)
    r = FrameAnalysisResult(stars=[m, m])
    assert r.star_count == 2


# ──────────────────────────────────────────────────────────────────────────────
# StarAnalyzer.analyze — basic
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def analyzer() -> StarAnalyzer:
    return StarAnalyzer(detection_sigma=3.5, min_star_area=4, max_stars=50)


def test_analyze_empty_returns_empty(analyzer: StarAnalyzer) -> None:
    frame = np.zeros((60, 60), dtype=np.float64)
    r = analyzer.analyze(frame)
    assert r.star_count == 0
    assert r.median_fwhm == 0.0


def test_analyze_single_star_detected(analyzer: StarAnalyzer) -> None:
    frame = np.zeros((60, 60), dtype=np.float64)
    _gaussian_star(frame, 30, 30, amplitude=0.9, sigma=2.0)
    frame += np.random.default_rng(1).normal(0, 0.003, frame.shape)
    r = analyzer.analyze(frame)
    assert r.star_count >= 1


def test_analyze_fwhm_in_range(analyzer: StarAnalyzer) -> None:
    frame = _make_frame(n_stars=5, sigma=2.0)
    r = analyzer.analyze(frame)
    if r.star_count > 0:
        # FWHM should be ~2*2.3548*sigma = ~4.7 px; allow generous range
        assert 1.0 < r.median_fwhm < 20.0


def test_analyze_ellipticity_range(analyzer: StarAnalyzer) -> None:
    frame = _make_frame(n_stars=5, sigma=2.0)
    r = analyzer.analyze(frame)
    for s in r.stars:
        assert 0.0 <= s.ellipticity <= 1.0


def test_analyze_strehl_range(analyzer: StarAnalyzer) -> None:
    frame = _make_frame(n_stars=5, sigma=2.0)
    r = analyzer.analyze(frame)
    for s in r.stars:
        assert 0.0 <= s.strehl <= 1.0


def test_analyze_rgb_frame(analyzer: StarAnalyzer) -> None:
    gray = _make_frame(n_stars=3, sigma=2.5)
    rgb = np.stack([gray, gray, gray], axis=-1)
    r = analyzer.analyze(rgb)
    assert isinstance(r, FrameAnalysisResult)


def test_analyze_p90_geq_median(analyzer: StarAnalyzer) -> None:
    frame = _make_frame(n_stars=6)
    r = analyzer.analyze(frame)
    if r.star_count >= 2:
        assert r.p90_fwhm >= r.median_fwhm - 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# FWHM limit / threshold flag
# ──────────────────────────────────────────────────────────────────────────────


def test_fwhm_limit_exceeded() -> None:
    az = StarAnalyzer(detection_sigma=3.5, fwhm_limit=2.0)  # very tight limit
    frame = _make_frame(n_stars=5, sigma=3.0)  # large stars → large FWHM
    r = az.analyze(frame)
    if r.star_count > 0 and r.median_fwhm > 2.0:
        assert r.exceeds_fwhm_limit


def test_fwhm_limit_not_exceeded_when_disabled() -> None:
    az = StarAnalyzer(fwhm_limit=0.0)
    frame = _make_frame(n_stars=5, sigma=3.0)
    r = az.analyze(frame)
    assert not r.exceeds_fwhm_limit


# ──────────────────────────────────────────────────────────────────────────────
# HFR cross-validation
# ──────────────────────────────────────────────────────────────────────────────


def test_hfr_cross_val_delta_computed(analyzer: StarAnalyzer) -> None:
    frame = _make_frame(n_stars=5)
    r = analyzer.analyze(frame, hfr=2.0)
    if r.star_count > 0:
        assert r.hfr_cross_val_delta >= 0.0


def test_hfr_cross_val_delta_absent_when_no_hfr(analyzer: StarAnalyzer) -> None:
    frame = _make_frame(n_stars=5)
    r = analyzer.analyze(frame)
    assert r.hfr_cross_val_delta == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────────────────────


def test_csv_empty_result() -> None:
    az = StarAnalyzer()
    csv_str = az.to_csv(FrameAnalysisResult())
    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert rows == []


def test_csv_has_header() -> None:
    az = StarAnalyzer()
    frame = _make_frame(n_stars=3)
    r = az.analyze(frame)
    csv_str = az.to_csv(r)
    assert "fwhm" in csv_str.splitlines()[0]


def test_csv_row_count_matches_stars() -> None:
    az = StarAnalyzer()
    frame = _make_frame(n_stars=4)
    r = az.analyze(frame)
    if r.star_count == 0:
        pytest.skip("no stars detected in synthetic frame")
    csv_str = az.to_csv(r)
    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == r.star_count


def test_csv_fields_parseable() -> None:
    az = StarAnalyzer()
    frame = _make_frame(n_stars=3)
    r = az.analyze(frame)
    if r.star_count == 0:
        pytest.skip("no stars detected")
    csv_str = az.to_csv(r)
    reader = csv.DictReader(io.StringIO(csv_str))
    for row in reader:
        assert float(row["fwhm"]) > 0.0
        assert 0.0 <= float(row["ellipticity"]) <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_uniform_frame_returns_empty(analyzer: StarAnalyzer) -> None:
    r = analyzer.analyze(np.full((50, 50), 0.5))
    assert r.star_count == 0


def test_max_stars_respected() -> None:
    az = StarAnalyzer(detection_sigma=3.0, max_stars=3)
    frame = _make_frame(n_stars=15, size=120)
    r = az.analyze(frame)
    assert r.star_count <= 3


def test_small_frame_no_crash(analyzer: StarAnalyzer) -> None:
    frame = np.zeros((8, 8), dtype=np.float64)
    _gaussian_star(frame, 4, 4, amplitude=0.9, sigma=1.0)
    r = analyzer.analyze(frame)
    assert isinstance(r, FrameAnalysisResult)
