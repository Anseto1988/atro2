import numpy as np
import pytest
from astroai.processing.denoise import Denoiser


np.random.seed(42)


def _make_noisy_frame(h=128, w=128):
    """Create a synthetic astro frame with signal and noise."""
    yy, xx = np.mgrid[0:h, 0:w]
    nebula = 200.0 * np.exp(
        -((yy - h // 2) ** 2 + (xx - w // 2) ** 2) / (2 * 30 ** 2)
    )
    stars = np.zeros((h, w), dtype=np.float64)
    for cy, cx in [(30, 40), (80, 100), (60, 20)]:
        stars += 5000.0 * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2)
        )
    noise = np.random.normal(0, 50, (h, w))
    return np.clip(nebula + stars + noise + 100, 0, 65535).astype(np.float64)


def _make_clean_frame(h=128, w=128):
    """Smooth gradient frame with no noise."""
    yy, _ = np.mgrid[0:h, 0:w]
    return (yy / h * 1000.0).astype(np.float64)


def _make_uniform_frame(h=64, w=64):
    return np.full((h, w), 500.0, dtype=np.float64)


@pytest.fixture()
def denoiser():
    return Denoiser(strength=1.0)


def test_denoise_returns_same_shape(denoiser):
    frame = _make_noisy_frame()
    result = denoiser.denoise(frame)
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_denoise_reduces_noise(denoiser):
    frame = _make_noisy_frame()
    result = denoiser.denoise(frame)
    noise_before = float(np.std(frame - _make_clean_frame(128, 128)))
    noise_after = float(np.std(result - _make_clean_frame(128, 128)))
    assert noise_after <= noise_before


def test_denoise_preserves_value_range(denoiser):
    frame = _make_noisy_frame()
    result = denoiser.denoise(frame)
    assert result.min() >= 0.0
    assert result.max() <= 65535.0


def test_denoise_uniform_frame_unchanged(denoiser):
    frame = _make_uniform_frame()
    result = denoiser.denoise(frame)
    np.testing.assert_allclose(result, frame, atol=1e-6)


def test_denoise_batch_returns_correct_count(denoiser):
    frames = [_make_noisy_frame(), _make_clean_frame()]
    results = denoiser.denoise_batch(frames)
    assert len(results) == 2
    for r, f in zip(results, frames):
        assert r.shape == f.shape


def test_denoise_rgb_frame(denoiser):
    h, w = 64, 64
    rgb = np.random.rand(h, w, 3).astype(np.float64) * 1000
    result = denoiser.denoise(rgb)
    assert result.shape == (h, w, 3)
    assert result.dtype == rgb.dtype


def test_strength_zero_returns_original():
    denoiser = Denoiser(strength=0.0)
    frame = _make_noisy_frame()
    result = denoiser.denoise(frame)
    np.testing.assert_allclose(result, frame, atol=1e-10)


def test_noise_estimation():
    frame = np.random.normal(1000, 50, (128, 128)).astype(np.float64)
    noise_est = Denoiser._estimate_noise(frame)
    assert 10.0 < noise_est < 100.0
