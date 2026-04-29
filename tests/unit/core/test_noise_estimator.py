"""Unit tests for astroai.core.noise_estimator."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.noise_estimator import NoiseEstimate, NoiseEstimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_image(h: int, w: int, sigma: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 0.1)
    return np.clip(base + rng.normal(0, sigma, (h, w)), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# NoiseEstimate dataclass
# ---------------------------------------------------------------------------

class TestNoiseEstimate:
    def test_fields_accessible(self):
        est = NoiseEstimate(sky_sigma=0.01, snr_db=20.0, noise_level_pct=10.0, suggested_strength=0.5)
        assert est.sky_sigma == pytest.approx(0.01)
        assert est.snr_db == pytest.approx(20.0)
        assert est.noise_level_pct == pytest.approx(10.0)
        assert est.suggested_strength == pytest.approx(0.5)

    def test_frozen(self):
        est = NoiseEstimate(sky_sigma=0.01, snr_db=20.0, noise_level_pct=10.0, suggested_strength=0.5)
        with pytest.raises(Exception):
            est.sky_sigma = 0.99  # type: ignore[misc]

    def test_str_contains_key_fields(self):
        est = NoiseEstimate(sky_sigma=0.0123, snr_db=18.5, noise_level_pct=12.3, suggested_strength=0.4)
        s = str(est)
        assert "σ=" in s
        assert "SNR=" in s
        assert "strength=" in s


# ---------------------------------------------------------------------------
# NoiseEstimator construction
# ---------------------------------------------------------------------------

class TestNoiseEstimatorConstruction:
    def test_defaults(self):
        ne = NoiseEstimator()
        assert ne.iterations == 5
        assert ne.kappa == pytest.approx(3.0)

    def test_custom_params(self):
        ne = NoiseEstimator(iterations=3, kappa=2.5)
        assert ne.iterations == 3

    def test_invalid_iterations(self):
        with pytest.raises(ValueError, match="iterations"):
            NoiseEstimator(iterations=0)

    def test_invalid_kappa(self):
        with pytest.raises(ValueError, match="kappa"):
            NoiseEstimator(kappa=0.0)

    def test_negative_kappa(self):
        with pytest.raises(ValueError):
            NoiseEstimator(kappa=-1.0)


# ---------------------------------------------------------------------------
# estimate() — input shapes
# ---------------------------------------------------------------------------

class TestEstimateInputShapes:
    def test_2d_grayscale(self):
        img = _make_image(64, 64, sigma=0.02)
        est = NoiseEstimator().estimate(img)
        assert isinstance(est, NoiseEstimate)

    def test_3d_single_channel(self):
        img = _make_image(64, 64, sigma=0.02)[:, :, np.newaxis]
        est = NoiseEstimator().estimate(img)
        assert isinstance(est, NoiseEstimate)

    def test_3d_rgb(self):
        rng = np.random.default_rng(7)
        img = np.clip(0.1 + rng.normal(0, 0.02, (64, 64, 3)), 0, 1).astype(np.float32)
        est = NoiseEstimator().estimate(img)
        assert isinstance(est, NoiseEstimate)

    def test_unsupported_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            NoiseEstimator().estimate(np.zeros((64,)))

    def test_4d_raises(self):
        with pytest.raises(ValueError, match="shape"):
            NoiseEstimator().estimate(np.zeros((2, 64, 64, 3)))

    def test_float64_accepted(self):
        img = _make_image(64, 64, 0.02).astype(np.float64)
        est = NoiseEstimator().estimate(img)
        assert est.sky_sigma > 0

    def test_float32_accepted(self):
        img = _make_image(64, 64, 0.02).astype(np.float32)
        est = NoiseEstimator().estimate(img)
        assert est.sky_sigma > 0


# ---------------------------------------------------------------------------
# sky_sigma accuracy
# ---------------------------------------------------------------------------

class TestSkySigmaAccuracy:
    def test_low_noise_image_low_sigma(self):
        img = _make_image(128, 128, sigma=0.003)
        est = NoiseEstimator().estimate(img)
        assert est.sky_sigma < 0.010

    def test_medium_noise_image(self):
        img = _make_image(128, 128, sigma=0.020)
        est = NoiseEstimator().estimate(img)
        assert 0.010 < est.sky_sigma < 0.040

    def test_high_noise_image(self):
        img = _make_image(128, 128, sigma=0.060)
        est = NoiseEstimator().estimate(img)
        assert est.sky_sigma > 0.030

    def test_sigma_increases_with_noise(self):
        low = NoiseEstimator().estimate(_make_image(64, 64, sigma=0.005)).sky_sigma
        high = NoiseEstimator().estimate(_make_image(64, 64, sigma=0.050)).sky_sigma
        assert high > low

    def test_pure_flat_image(self):
        img = np.full((64, 64), 0.5, dtype=np.float32)
        est = NoiseEstimator().estimate(img)
        assert est.sky_sigma == pytest.approx(0.0, abs=1e-10)

    def test_sigma_positive(self):
        img = _make_image(64, 64, sigma=0.01)
        est = NoiseEstimator().estimate(img)
        assert est.sky_sigma >= 0.0

    def test_different_seeds_similar_sigma(self):
        s1 = NoiseEstimator().estimate(_make_image(128, 128, 0.02, seed=1)).sky_sigma
        s2 = NoiseEstimator().estimate(_make_image(128, 128, 0.02, seed=2)).sky_sigma
        assert abs(s1 - s2) < 0.005


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------

class TestSnr:
    def test_snr_higher_for_less_noise(self):
        clean = NoiseEstimator().estimate(_make_image(128, 128, sigma=0.005)).snr_db
        noisy = NoiseEstimator().estimate(_make_image(128, 128, sigma=0.050)).snr_db
        assert clean > noisy

    def test_flat_image_snr_zero(self):
        img = np.full((64, 64), 0.5, dtype=np.float32)
        est = NoiseEstimator().estimate(img)
        assert est.snr_db == pytest.approx(0.0)

    def test_snr_is_float(self):
        img = _make_image(64, 64, sigma=0.01)
        est = NoiseEstimator().estimate(img)
        assert isinstance(est.snr_db, float)


# ---------------------------------------------------------------------------
# noise_level_pct
# ---------------------------------------------------------------------------

class TestNoiseLevelPct:
    def test_capped_at_100(self):
        img = _make_image(64, 64, sigma=0.20)
        est = NoiseEstimator().estimate(img)
        assert est.noise_level_pct <= 100.0

    def test_low_noise_low_pct(self):
        img = _make_image(128, 128, sigma=0.002)
        est = NoiseEstimator().estimate(img)
        assert est.noise_level_pct < 10.0

    def test_pct_non_negative(self):
        img = _make_image(64, 64, sigma=0.01)
        est = NoiseEstimator().estimate(img)
        assert est.noise_level_pct >= 0.0


# ---------------------------------------------------------------------------
# suggested_strength mapping
# ---------------------------------------------------------------------------

class TestSuggestedStrength:
    def test_very_clean_image_low_strength(self):
        img = _make_image(128, 128, sigma=0.002)
        est = NoiseEstimator().estimate(img)
        assert est.suggested_strength <= 0.4

    def test_medium_noise_medium_strength(self):
        img = _make_image(128, 128, sigma=0.020)
        est = NoiseEstimator().estimate(img)
        assert 0.3 <= est.suggested_strength <= 0.8

    def test_very_noisy_high_strength(self):
        img = _make_image(128, 128, sigma=0.080)
        est = NoiseEstimator().estimate(img)
        assert est.suggested_strength >= 0.8

    def test_strength_in_valid_range(self):
        for sigma in [0.001, 0.005, 0.015, 0.030, 0.060, 0.100]:
            img = _make_image(64, 64, sigma)
            est = NoiseEstimator().estimate(img)
            assert 0.0 <= est.suggested_strength <= 1.0

    def test_strength_monotone_with_sigma(self):
        sigmas = [0.002, 0.008, 0.020, 0.050]
        strengths = [
            NoiseEstimator().estimate(_make_image(64, 64, s)).suggested_strength
            for s in sigmas
        ]
        assert strengths == sorted(strengths)


# ---------------------------------------------------------------------------
# _strength_from_sigma unit tests
# ---------------------------------------------------------------------------

class TestStrengthFromSigma:
    def test_zero_sigma(self):
        s = NoiseEstimator._strength_from_sigma(0.0)
        assert s == pytest.approx(0.2)

    def test_boundary_0005(self):
        s = NoiseEstimator._strength_from_sigma(0.005)
        assert s == pytest.approx(0.2)

    def test_boundary_0015(self):
        s = NoiseEstimator._strength_from_sigma(0.015)
        assert s == pytest.approx(0.5)

    def test_boundary_0040(self):
        s = NoiseEstimator._strength_from_sigma(0.040)
        assert s == pytest.approx(0.8)

    def test_large_sigma_capped_at_1(self):
        s = NoiseEstimator._strength_from_sigma(10.0)
        assert s == pytest.approx(1.0)

    def test_midpoint_first_range(self):
        s = NoiseEstimator._strength_from_sigma(0.010)
        assert s == pytest.approx(0.35, abs=0.01)

    def test_midpoint_second_range(self):
        s = NoiseEstimator._strength_from_sigma(0.0275)
        assert s == pytest.approx(0.65, abs=0.02)


# ---------------------------------------------------------------------------
# Sigma-clipping iterations
# ---------------------------------------------------------------------------

class TestSigmaClipping:
    def test_single_iteration_still_works(self):
        img = _make_image(64, 64, sigma=0.02)
        est = NoiseEstimator(iterations=1).estimate(img)
        assert est.sky_sigma > 0

    def test_many_iterations_converges(self):
        img = _make_image(128, 128, sigma=0.02)
        est_3 = NoiseEstimator(iterations=3).estimate(img)
        est_10 = NoiseEstimator(iterations=10).estimate(img)
        assert abs(est_3.sky_sigma - est_10.sky_sigma) < 0.005

    def test_strict_kappa_lower_sigma(self):
        img = _make_image(128, 128, sigma=0.02)
        est_strict = NoiseEstimator(kappa=1.5).estimate(img)
        est_loose = NoiseEstimator(kappa=5.0).estimate(img)
        # strict kappa clips more aggressively → lower sigma estimate
        assert est_strict.sky_sigma <= est_loose.sky_sigma + 0.005

    def test_sparse_data_fallback(self):
        # Very tiny image — still produces a result
        img = np.array([[0.1, 0.2], [0.15, 0.12]], dtype=np.float32)
        est = NoiseEstimator().estimate(img)
        assert isinstance(est.sky_sigma, float)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_zeros_image(self):
        img = np.zeros((64, 64), dtype=np.float32)
        est = NoiseEstimator().estimate(img)
        assert est.sky_sigma == pytest.approx(0.0, abs=1e-10)
        assert est.noise_level_pct == pytest.approx(0.0, abs=1e-10)

    def test_all_ones_image(self):
        img = np.ones((64, 64), dtype=np.float32)
        est = NoiseEstimator().estimate(img)
        assert est.sky_sigma == pytest.approx(0.0, abs=1e-10)

    def test_large_image_performance(self):
        img = _make_image(512, 512, sigma=0.02)
        est = NoiseEstimator().estimate(img)
        assert est.sky_sigma > 0

    def test_non_contiguous_array(self):
        img = _make_image(64, 64, sigma=0.02)
        flipped = np.asfortranarray(img)
        est = NoiseEstimator().estimate(flipped)
        assert est.sky_sigma > 0
