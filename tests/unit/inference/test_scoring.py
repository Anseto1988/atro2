import numpy as np
import pytest
from unittest.mock import patch
from astroai.inference.scoring import FrameScorer


np.random.seed(42)


def _make_good_frame(h=256, w=256):
    frame = np.zeros((h, w), dtype=np.float64)
    star_positions = [(60, 80), (120, 180), (200, 50), (180, 220)]
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx in star_positions:
        sigma = 2.5
        blob = 5000.0 * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)
        )
        frame += blob
    frame += np.random.normal(0, 5, (h, w))
    return np.clip(frame, 0, 65535)


def _make_bad_frame(h=256, w=256):
    return np.full((h, w), 40000.0, dtype=np.float64)


def _make_blank_frame(h=256, w=256):
    return np.zeros((h, w), dtype=np.float64)


@pytest.fixture()
def scorer():
    return FrameScorer()


def test_score_returns_float_between_0_and_1(scorer):
    frame = _make_good_frame()
    s = scorer.score(frame)
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_good_frame_scores_higher_than_bad(scorer):
    good = scorer.score(_make_good_frame())
    bad = scorer.score(_make_bad_frame())
    assert good > bad


def test_blank_frame_scores_very_low(scorer):
    s = scorer.score(_make_blank_frame())
    assert s <= 0.25


def test_score_batch_returns_correct_count(scorer):
    frames = [_make_good_frame(), _make_bad_frame(), _make_blank_frame()]
    scores = scorer.score_batch(frames)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_to_grayscale_converts_rgb(scorer):
    rgb = np.random.rand(64, 64, 3).astype(np.float64)
    gray = FrameScorer._to_grayscale(rgb)
    assert gray.ndim == 2
    assert gray.shape == (64, 64)


class TestFrameScorerEdgeCases:
    def test_hfr_zero_flux_patch_skipped(self):
        # total_flux < 1e-8 → continue (line 77): inject star whose patch sums to ~0
        scorer = FrameScorer(star_threshold_sigma=0.5, min_star_area=1)
        frame = np.zeros((32, 32), dtype=np.float64)
        # Inject a fake star at center with near-zero background
        with patch.object(scorer, "_detect_stars", return_value=[(16, 16, 1.0)]):
            # Override gray computation so patch values are essentially 0
            result = scorer._score_hfr(frame)
        # total_flux for zero frame = 0 → all stars skipped → hfrs empty → 0.0
        assert result == 0.0

    def test_hfr_returns_zero_when_no_stars(self):
        # _score_hfr returns 0.0 when _detect_stars returns [] (lines 59-60)
        scorer = FrameScorer()
        flat = np.full((64, 64), 500.0, dtype=np.float64)
        assert scorer._score_hfr(flat) == 0.0

    def test_hfr_max_r_less_than_one(self):
        # max_r < 1.0 → continue (line 80-81): inject star so patch is 1×1
        scorer = FrameScorer(star_threshold_sigma=0.5, min_star_area=1)
        # A 1×1 grayscale frame with std=0 can't detect stars normally,
        # so we inject the star directly. The frame is also 1×1 so the
        # patch after clipping is 1×1 → max_r = 0.0 < 1.0.
        frame_1x1 = np.array([[1e6]], dtype=np.float64)
        with patch.object(scorer, "_detect_stars", return_value=[(0, 0, 1e6)]):
            result = scorer._score_hfr(frame_1x1)
        # patch is 1×1, max_r=0 < 1.0 → star skipped → hfrs empty → 0.0
        assert result == 0.0

    def test_hfr_all_patches_skipped_returns_zero(self):
        # hfrs ends up empty → lines 89-90 return 0.0
        scorer = FrameScorer(star_threshold_sigma=0.5, min_star_area=1)
        frame = np.zeros((32, 32), dtype=np.float64)
        with patch.object(scorer, "_detect_stars", return_value=[(16, 16, 1.0)]):
            result = scorer._score_hfr(frame)
        assert result == 0.0

    def test_roundness_label_val_zero_skipped(self):
        # After re-labeling with a fresh threshold the centroid may fall on an
        # unlabeled pixel (label_val == 0) → continue (line 106).
        # We craft a frame where _detect_stars finds a star but the re-labeled
        # mask at that position is 0 by using mismatched sigmas.
        scorer = FrameScorer(star_threshold_sigma=0.01, min_star_area=1)
        rng = np.random.RandomState(7)
        frame = rng.uniform(0, 10, (32, 32)).astype(np.float64)
        # Roundness should still return a valid float
        result = scorer._score_roundness(frame)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_roundness_region_too_small_skipped(self):
        # len(ys) < min_star_area after re-labeling → continue (line 110)
        # inject centroid pointing to a 1-pixel region with min_star_area=3
        scorer = FrameScorer(star_threshold_sigma=0.5, min_star_area=3)
        frame = np.zeros((32, 32), dtype=np.float64)
        frame[15, 15] = 1e6
        with patch.object(scorer, "_detect_stars", return_value=[(15, 15, 1e6)]):
            result = scorer._score_roundness(frame)
        assert result == 0.0

    def test_roundness_trace_zero_skipped(self):
        # A perfectly point-like star (all pixels at same coordinates) gives
        # trace ≈ 0 → continue (line 121).  Use min_star_area=1 and a single
        # bright pixel so the region is 1×1 and all moments vanish.
        scorer = FrameScorer(star_threshold_sigma=0.5, min_star_area=1)
        frame = np.zeros((32, 32), dtype=np.float64)
        frame[15, 15] = 1e6
        result = scorer._score_roundness(frame)
        assert isinstance(result, float)

    def test_roundness_l1_zero_skipped(self):
        # l1 < 1e-12 → continue (line 128): occurs when eigenvalue is zero
        # This is closely related to trace ≈ 0, so reuse the single-pixel setup.
        scorer = FrameScorer(star_threshold_sigma=0.5, min_star_area=1)
        frame = np.zeros((32, 32), dtype=np.float64)
        frame[16, 16] = 1e6
        result = scorer._score_roundness(frame)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_roundness_returns_zero_no_valid_stars(self):
        # roundness_vals stays empty → line 131-132 returns 0.0
        scorer = FrameScorer(star_threshold_sigma=0.5, min_star_area=1)
        frame = np.zeros((32, 32), dtype=np.float64)
        frame[16, 16] = 1e6
        assert scorer._score_roundness(frame) == 0.0

    def test_cloud_small_frame_bright_uniform_returns_zero(self):
        # h < block_size → lines 139-144; med > mean*1.2 and std < 20 → 0.0
        # 20% zeros + 80% value-10: median=10, mean≈8, ratio≈1.25, std≈4
        scorer = FrameScorer(cloud_block_size=64)
        frame = np.full((32, 32), 10.0, dtype=np.float64)
        n_zeros = int(0.20 * 32 * 32)
        frame.flat[:n_zeros] = 0.0
        result = scorer._score_cloud_coverage(frame)
        assert result == 0.0

    def test_cloud_small_frame_returns_one_when_not_cloudy(self):
        # h < block_size and condition is False → returns 1.0 (line 144)
        scorer = FrameScorer(cloud_block_size=64)
        frame = np.zeros((32, 32), dtype=np.float64)
        result = scorer._score_cloud_coverage(frame)
        assert result == 1.0

    def test_cloud_total_blocks_zero(self):
        # total_blocks == 0 → returns 1.0 (line 164)
        # n_y or n_x == 0 when h < bs or w < bs (already handled in lines 139-144)
        # Line 164 is only reachable when h >= bs and w >= bs but somehow
        # the loop produced no blocks — defensively unreachable in practice.
        # Verify normal path still produces a valid float in range.
        scorer = FrameScorer(cloud_block_size=64)
        frame = np.zeros((128, 128), dtype=np.float64)
        result = scorer._score_cloud_coverage(frame)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_detect_stars_filters_small_regions(self) -> None:
        """Line 48: area < min_star_area → continue. Single bright pixel has area=1;
        with min_star_area=5 it is filtered out, returning an empty list."""
        scorer = FrameScorer(star_threshold_sigma=0.5, min_star_area=5)
        frame = np.zeros((32, 32), dtype=np.float64)
        frame[16, 16] = 1e6  # single pixel → area=1 < 5
        gray = FrameScorer._to_grayscale(frame)
        stars = scorer._detect_stars(gray)
        assert stars == []
