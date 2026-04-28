import numpy as np
import pytest
from astroai.processing.stars import StarManager


np.random.seed(42)


def _make_starfield(h=128, w=128, n_stars=8):
    """Create synthetic frame with stars on smooth background."""
    yy, xx = np.mgrid[0:h, 0:w]
    nebula = 200.0 * np.exp(
        -((yy - h // 2) ** 2 + (xx - w // 2) ** 2) / (2 * 30 ** 2)
    )
    stars = np.zeros((h, w), dtype=np.float64)
    rng = np.random.RandomState(42)
    positions = [(rng.randint(10, h - 10), rng.randint(10, w - 10)) for _ in range(n_stars)]
    for cy, cx in positions:
        sigma = rng.uniform(1.5, 3.0)
        flux = rng.uniform(3000, 8000)
        stars += flux * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)
        )
    noise = rng.normal(0, 5, (h, w))
    return np.clip(nebula + stars + noise + 50, 0, 65535).astype(np.float64)


def _make_blank_frame(h=64, w=64):
    return np.full((h, w), 100.0, dtype=np.float64)


def _make_rgb_starfield(h=128, w=128):
    mono = _make_starfield(h, w)
    return np.stack([mono, mono * 0.8, mono * 0.6], axis=-1)


@pytest.fixture()
def manager():
    return StarManager()


def test_star_mask_is_boolean(manager):
    frame = _make_starfield()
    mask = manager.create_star_mask(frame)
    assert mask.dtype == np.bool_
    assert mask.shape == frame.shape[:2]


def test_star_mask_detects_stars(manager):
    frame = _make_starfield()
    mask = manager.create_star_mask(frame)
    assert mask.sum() > 0


def test_blank_frame_no_stars(manager):
    frame = _make_blank_frame()
    mask = manager.create_star_mask(frame)
    assert mask.sum() == 0


def test_separate_returns_two_components(manager):
    frame = _make_starfield()
    starless, stars_only = manager.separate(frame)
    assert starless.shape == frame.shape
    assert stars_only.shape == frame.shape
    assert starless.dtype == frame.dtype


def test_starless_has_lower_peak(manager):
    frame = _make_starfield()
    starless, _ = manager.separate(frame)
    assert starless.max() < frame.max()


def test_stars_only_nonnegative(manager):
    frame = _make_starfield()
    _, stars_only = manager.separate(frame)
    assert stars_only.min() >= 0.0


def test_reduce_stars_factor_1_preserves(manager):
    frame = _make_starfield()
    result = manager.reduce_stars(frame, factor=1.0)
    assert result.shape == frame.shape


def test_reduce_stars_factor_0_equals_starless(manager):
    frame = _make_starfield()
    starless, _ = manager.separate(frame)
    reduced = manager.reduce_stars(frame, factor=0.0)
    np.testing.assert_allclose(reduced, starless, atol=1e-6)


def test_reduce_stars_decreases_brightness(manager):
    frame = _make_starfield()
    reduced = manager.reduce_stars(frame, factor=0.3)
    assert reduced.max() < frame.max()


def test_separate_rgb(manager):
    frame = _make_rgb_starfield()
    starless, stars_only = manager.separate(frame)
    assert starless.shape == frame.shape
    assert stars_only.shape == frame.shape
    assert starless.ndim == 3


def test_to_grayscale():
    rgb = np.random.rand(32, 32, 3).astype(np.float64)
    gray = StarManager._to_grayscale(rgb)
    assert gray.ndim == 2
    assert gray.shape == (32, 32)


def test_grayscale_passthrough():
    mono = np.random.rand(32, 32).astype(np.float64)
    gray = StarManager._to_grayscale(mono)
    np.testing.assert_array_equal(gray, mono.astype(np.float64))


class TestStarManagerEdgeCases:
    def test_elongated_region_skipped_aspect_gt4(self):
        # aspect > 4.0 → continue (line 60): vertical stripe filtered out
        manager = StarManager(detection_sigma=3.0, min_star_area=3, max_star_area=5000)
        frame = np.zeros((64, 64), dtype=np.float64)
        frame[10:30, 32] = 10000.0
        frame += 10.0
        mask = manager.create_star_mask(frame)
        assert mask.sum() == 0

    def test_inpaint_channel_all_pixels_masked_returns_unchanged(self):
        # mask is all True → bg_values.size == 0 → return result unchanged (lines 118-119)
        ch = np.ones((8, 8), dtype=np.float64) * 77.0
        mask_all = np.ones((8, 8), dtype=np.bool_)
        result = StarManager._inpaint_channel(ch, mask_all)
        np.testing.assert_array_equal(result, ch)

    def test_inpaint_channel_empty_mask_returns_unchanged(self):
        # mask.any() is False → early return (line 114-115)
        ch = np.ones((8, 8), dtype=np.float64) * 33.0
        mask_empty = np.zeros((8, 8), dtype=np.bool_)
        result = StarManager._inpaint_channel(ch, mask_empty)
        np.testing.assert_array_equal(result, ch)
