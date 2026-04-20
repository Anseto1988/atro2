import numpy as np
import pytest
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
