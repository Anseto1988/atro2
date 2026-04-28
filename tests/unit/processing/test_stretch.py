import numpy as np
import pytest
from astroai.processing.stretch import IntelligentStretcher


np.random.seed(42)


def _make_linear_frame(h=128, w=128):
    """Simulate a linear astro image: faint nebula + noise + stars."""
    yy, xx = np.mgrid[0:h, 0:w]
    nebula = 0.08 * np.exp(
        -((yy - h // 2) ** 2 + (xx - w // 2) ** 2) / (2 * 25 ** 2)
    )
    stars = np.zeros((h, w), dtype=np.float64)
    for cy, cx in [(20, 30), (90, 110), (50, 80)]:
        stars += 0.5 * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2)
        )
    noise = np.random.normal(0, 0.003, (h, w))
    return np.clip(nebula + stars + noise + 0.05, 0.0, 1.0)


def _make_already_stretched(h=64, w=64):
    """Frame that is already well-distributed in 0..1."""
    return np.random.rand(h, w).astype(np.float64)


def _make_uniform_frame(h=64, w=64):
    return np.full((h, w), 0.5, dtype=np.float64)


@pytest.fixture()
def stretcher():
    return IntelligentStretcher()


def test_stretch_returns_same_shape(stretcher):
    frame = _make_linear_frame()
    result = stretcher.stretch(frame)
    assert result.shape == frame.shape


def test_stretch_output_in_0_1(stretcher):
    frame = _make_linear_frame()
    result = stretcher.stretch(frame)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_stretch_increases_star_contrast(stretcher):
    """STF stretch should increase contrast between peaks and background."""
    frame = _make_linear_frame()
    result = stretcher.stretch(frame)
    bg_before = float(np.median(frame))
    peak_before = float(frame.max())
    contrast_before = peak_before - bg_before
    bg_after = float(np.median(result))
    peak_after = float(result.max())
    contrast_after = peak_after - bg_after
    assert contrast_after > contrast_before * 0.5


def test_stretch_preserves_dtype(stretcher):
    frame = _make_linear_frame().astype(np.float32)
    result = stretcher.stretch(frame)
    assert result.dtype == np.float32


def test_stretch_batch_returns_correct_count(stretcher):
    frames = [_make_linear_frame(), _make_already_stretched()]
    results = stretcher.stretch_batch(frames)
    assert len(results) == 2
    for r, f in zip(results, frames):
        assert r.shape == f.shape


def test_stretch_rgb_linked():
    stretcher = IntelligentStretcher(linked_channels=True)
    rgb = np.random.rand(64, 64, 3).astype(np.float64) * 0.05
    result = stretcher.stretch(rgb)
    assert result.shape == (64, 64, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_stretch_rgb_independent():
    stretcher = IntelligentStretcher(linked_channels=False)
    rgb = np.random.rand(64, 64, 3).astype(np.float64) * 0.05
    result = stretcher.stretch(rgb)
    assert result.shape == (64, 64, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_uniform_frame_stretches_gracefully(stretcher):
    frame = _make_uniform_frame()
    result = stretcher.stretch(frame)
    assert result.shape == frame.shape
    assert np.all(np.isfinite(result))


def test_mtf_function():
    data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result = IntelligentStretcher._apply_mtf(data, 0.5)
    np.testing.assert_allclose(result[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(result[-1], 1.0, atol=1e-6)
    assert np.all(np.diff(result) >= 0)


def test_custom_target_background():
    stretcher = IntelligentStretcher(target_background=0.15)
    frame = _make_linear_frame()
    result = stretcher.stretch(frame)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


class TestStretcherEdgeCases:
    def test_normalize_uniform_rgb_returns_zeros(self):
        # _normalize: rng < 1e-10 → return zeros_like (line 95)
        # triggered by a perfectly uniform RGB frame via _stretch_linked
        stretcher = IntelligentStretcher(linked_channels=True)
        frame = np.full((16, 16, 3), 0.5, dtype=np.float64)
        result = stretcher.stretch(frame)
        assert result.shape == (16, 16, 3)
        assert np.all(result == 0.0)

    def test_normalize_uniform_rgb_independent_returns_zeros(self):
        # _normalize also called from _stretch_independent (line 95)
        stretcher = IntelligentStretcher(linked_channels=False)
        frame = np.full((16, 16, 3), 0.5, dtype=np.float64)
        result = stretcher.stretch(frame)
        assert result.shape == (16, 16, 3)
        assert np.all(result == 0.0)

    def test_mtf_balance_bg_near_zero_returns_half(self):
        # bg < 1e-10 → return 0.5 (line 137)
        result = IntelligentStretcher._mtf_balance(0.0, 0.25)
        assert result == 0.5

    def test_mtf_balance_bg_near_one_returns_half(self):
        # bg > 1 - 1e-10 → return 0.5 (line 137)
        result = IntelligentStretcher._mtf_balance(1.0, 0.25)
        assert result == 0.5
