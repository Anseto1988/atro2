import numpy as np
import pytest
from astroai.processing.stars import StarManager
from astroai.processing.stars.star_manager import AUTO_TILE_THRESHOLD


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

    def test_area_filter_small_region_skipped(self):
        """Line 52: area < min_area → continue. Isolated single pixel is filtered when min_area=5."""
        manager = StarManager(detection_sigma=0.5, min_star_area=5, max_star_area=5000)
        frame = np.zeros((32, 32), dtype=np.float64)
        frame[16, 16] = 1e6  # single pixel → area=1 < 5
        mask = manager.create_star_mask(frame)
        assert mask.sum() == 0


class TestTileProcessing:

    def test_needs_tiling_above_threshold(self):
        assert StarManager.needs_tiling(6000, 4000) is True

    def test_needs_tiling_below_threshold(self):
        assert StarManager.needs_tiling(128, 128) is False

    def test_needs_tiling_at_boundary(self):
        assert StarManager.needs_tiling(4096, 4096) is False
        assert StarManager.needs_tiling(4097, 4096) is True

    def test_cosine_weight_center_is_one(self):
        w = StarManager._cosine_weight_1d(512, 64, at_start=False, at_end=False)
        assert w[256] == 1.0
        assert w[100] == 1.0

    def test_cosine_weight_boundary_full_at_edge(self):
        w = StarManager._cosine_weight_1d(512, 64, at_start=True, at_end=False)
        assert w[0] == 1.0
        w2 = StarManager._cosine_weight_1d(512, 64, at_start=False, at_end=True)
        assert w2[-1] == 1.0

    def test_cosine_weight_ramp_at_overlap(self):
        w = StarManager._cosine_weight_1d(512, 64, at_start=False, at_end=False)
        assert w[0] == pytest.approx(0.0, abs=1e-10)
        assert w[63] == pytest.approx(1.0, abs=1e-10)
        assert w[-1] == pytest.approx(0.0, abs=1e-10)
        assert 0.0 < w[32] < 1.0

    def test_cosine_weight_zero_overlap_returns_ones(self):
        w = StarManager._cosine_weight_1d(512, 0, at_start=False, at_end=False)
        np.testing.assert_array_equal(w, np.ones(512))

    def test_cosine_weight_overlap_gte_length_returns_ones(self):
        w = StarManager._cosine_weight_1d(64, 64, at_start=False, at_end=False)
        np.testing.assert_array_equal(w, np.ones(64))

    def test_process_tiled_identity_6000x4000(self):
        image = np.random.RandomState(0).rand(6000, 4000).astype(np.float64)
        result = StarManager.process_tiled(
            image, lambda tile: tile, tile_size=512, overlap=64,
        )
        assert result.shape == (6000, 4000)
        assert result.dtype == image.dtype
        np.testing.assert_allclose(result, image, atol=1e-6)

    def test_process_tiled_rgb_6000x4000(self):
        rng = np.random.RandomState(1)
        image = rng.rand(6000, 4000, 3).astype(np.float64)
        result = StarManager.process_tiled(
            image, lambda tile: tile, tile_size=512, overlap=64,
        )
        assert result.shape == (6000, 4000, 3)
        np.testing.assert_allclose(result, image, atol=1e-6)

    def test_process_tiled_progress_callback(self):
        image = np.ones((6000, 4000), dtype=np.float64)
        calls: list[tuple[int, int]] = []
        StarManager.process_tiled(
            image, lambda tile: tile, tile_size=512, overlap=64,
            on_progress=lambda idx, total: calls.append((idx, total)),
        )
        assert len(calls) > 0
        assert calls[-1][0] == calls[-1][1]
        assert all(c[1] == calls[0][1] for c in calls)

    def test_process_tiled_small_image_works(self):
        image = np.ones((256, 256), dtype=np.float64) * 42.0
        result = StarManager.process_tiled(
            image, lambda tile: tile, tile_size=512, overlap=64,
        )
        assert result.shape == (256, 256)
        np.testing.assert_allclose(result, 42.0, atol=1e-10)

    def test_process_tiled_constant_value_preserved(self):
        image = np.full((6000, 4000), 100.0, dtype=np.float64)
        result = StarManager.process_tiled(
            image, lambda tile: tile * 0.5, tile_size=512, overlap=64,
        )
        np.testing.assert_allclose(result, 50.0, atol=1e-6)

    def test_process_tiled_rgb_small_needs_padding(self):
        # 3D image smaller than tile_size → triggers pad_w.append((0,0)) (line 215)
        image = np.ones((64, 64, 3), dtype=np.float64) * 0.5
        result = StarManager.process_tiled(
            image, lambda tile: tile, tile_size=256, overlap=32,
        )
        assert result.shape == (64, 64, 3)
        np.testing.assert_allclose(result, 0.5, atol=1e-10)
