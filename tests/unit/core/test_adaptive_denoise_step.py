"""Unit tests for AdaptiveDenoiseStep."""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroai.core.noise_estimator import NoiseEstimate
from astroai.core.pipeline.adaptive_denoise_step import AdaptiveDenoiseStep
from astroai.core.pipeline.base import PipelineContext, PipelineProgress, PipelineStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(image: np.ndarray | None = None, use_result: bool = True) -> PipelineContext:
    ctx = PipelineContext()
    if image is not None:
        if use_result:
            ctx.result = image
        else:
            ctx.images = [image]
    return ctx


def _noisy(h: int = 64, w: int = 64, sigma: float = 0.02, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.2, sigma, (h, w)), 0.0, 1.0).astype(np.float32)


_FAKE_ESTIMATE = NoiseEstimate(
    sky_sigma=0.02, snr_db=15.0, noise_level_pct=20.0, suggested_strength=0.55
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_params(self):
        step = AdaptiveDenoiseStep()
        assert step._tile_size == 512
        assert step._tile_overlap == 64

    def test_custom_params(self):
        step = AdaptiveDenoiseStep(tile_size=256, tile_overlap=32)
        assert step._tile_size == 256
        assert step._tile_overlap == 32

    def test_estimator_created(self):
        step = AdaptiveDenoiseStep(estimator_iterations=3, estimator_kappa=2.5)
        assert step._estimator.iterations == 3
        assert step._estimator.kappa == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_name(self):
        assert AdaptiveDenoiseStep().name == "Adaptive Denoise"

    def test_stage(self):
        assert AdaptiveDenoiseStep().stage == PipelineStage.PROCESSING


# ---------------------------------------------------------------------------
# execute() — image source
# ---------------------------------------------------------------------------

@patch("astroai.core.pipeline.adaptive_denoise_step.DenoiseStep")
class TestExecuteImageSource:
    def _mock_step(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        return mock_step

    def test_image_from_result(self, MockDenoiseStep):
        self._mock_step(MockDenoiseStep)
        step = AdaptiveDenoiseStep()
        img = _noisy()
        ctx = _ctx(img, use_result=True)
        with patch.object(step._estimator, "estimate", return_value=_FAKE_ESTIMATE):
            out = step.execute(ctx)
        assert out is not None

    def test_image_from_images_list(self, MockDenoiseStep):
        self._mock_step(MockDenoiseStep)
        step = AdaptiveDenoiseStep()
        img = _noisy()
        ctx = _ctx(img, use_result=False)
        with patch.object(step._estimator, "estimate", return_value=_FAKE_ESTIMATE):
            out = step.execute(ctx)
        assert out is not None

    def test_result_takes_priority_over_images(self, MockDenoiseStep):
        mock_step = self._mock_step(MockDenoiseStep)
        step = AdaptiveDenoiseStep()
        img_result = _noisy(sigma=0.01)
        img_list = _noisy(sigma=0.05, seed=99)
        ctx = PipelineContext()
        ctx.result = img_result
        ctx.images = [img_list]
        captured_images = []
        def capture_estimate(img):
            captured_images.append(img)
            return _FAKE_ESTIMATE
        with patch.object(step._estimator, "estimate", side_effect=capture_estimate):
            step.execute(ctx)
        np.testing.assert_array_equal(captured_images[0], img_result)

    def test_no_image_skips_gracefully(self, MockDenoiseStep):
        mock_step = self._mock_step(MockDenoiseStep)
        step = AdaptiveDenoiseStep()
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out is ctx
        mock_step.execute.assert_not_called()


# ---------------------------------------------------------------------------
# execute() — metadata
# ---------------------------------------------------------------------------

@patch("astroai.core.pipeline.adaptive_denoise_step.DenoiseStep")
class TestExecuteMetadata:
    def test_stores_suggested_strength_in_metadata(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep()
        ctx = _ctx(_noisy())
        with patch.object(step._estimator, "estimate", return_value=_FAKE_ESTIMATE):
            out = step.execute(ctx)
        assert out.metadata[AdaptiveDenoiseStep.METADATA_KEY] == pytest.approx(0.55)

    def test_metadata_key_constant(self):
        assert AdaptiveDenoiseStep.METADATA_KEY == "estimated_denoise_strength"

    def test_different_estimates_stored_correctly(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep()
        est = NoiseEstimate(sky_sigma=0.005, snr_db=28.0, noise_level_pct=5.0, suggested_strength=0.25)
        ctx = _ctx(_noisy())
        with patch.object(step._estimator, "estimate", return_value=est):
            out = step.execute(ctx)
        assert out.metadata[AdaptiveDenoiseStep.METADATA_KEY] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# execute() — DenoiseStep delegation
# ---------------------------------------------------------------------------

@patch("astroai.core.pipeline.adaptive_denoise_step.DenoiseStep")
class TestDelegation:
    def test_delegates_to_denoise_step(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep()
        ctx = _ctx(_noisy())
        with patch.object(step._estimator, "estimate", return_value=_FAKE_ESTIMATE):
            step.execute(ctx)
        MockDenoiseStep.assert_called_once_with(
            strength=pytest.approx(0.55),
            tile_size=512,
            tile_overlap=64,
        )
        mock_step.execute.assert_called_once()

    def test_delegation_uses_custom_tile_params(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep(tile_size=256, tile_overlap=32)
        ctx = _ctx(_noisy())
        with patch.object(step._estimator, "estimate", return_value=_FAKE_ESTIMATE):
            step.execute(ctx)
        MockDenoiseStep.assert_called_once_with(
            strength=pytest.approx(0.55),
            tile_size=256,
            tile_overlap=32,
        )

    def test_returns_context_from_delegate(self, MockDenoiseStep):
        expected_ctx = PipelineContext()
        expected_ctx.metadata["from_delegate"] = True
        mock_step = MagicMock()
        mock_step.execute.return_value = expected_ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep()
        ctx = _ctx(_noisy())
        with patch.object(step._estimator, "estimate", return_value=_FAKE_ESTIMATE):
            out = step.execute(ctx)
        assert out.metadata.get("from_delegate") is True


# ---------------------------------------------------------------------------
# execute() — progress callbacks
# ---------------------------------------------------------------------------

@patch("astroai.core.pipeline.adaptive_denoise_step.DenoiseStep")
class TestProgressCallbacks:
    def test_progress_called_twice(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep()
        ctx = _ctx(_noisy())
        calls = []
        def capture(prog):
            calls.append(prog)
        with patch.object(step._estimator, "estimate", return_value=_FAKE_ESTIMATE):
            step.execute(ctx, capture)
        assert len(calls) == 2

    def test_first_progress_is_analysis(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep()
        ctx = _ctx(_noisy())
        calls = []
        with patch.object(step._estimator, "estimate", return_value=_FAKE_ESTIMATE):
            step.execute(ctx, lambda p: calls.append(p))
        assert calls[0].current == 0
        assert calls[0].total == 2
        assert "Rauschanalyse" in calls[0].message or "analyse" in calls[0].message.lower()

    def test_second_progress_contains_strength(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep()
        ctx = _ctx(_noisy())
        calls = []
        with patch.object(step._estimator, "estimate", return_value=_FAKE_ESTIMATE):
            step.execute(ctx, lambda p: calls.append(p))
        assert "0.55" in calls[1].message
        assert calls[1].current == 1

    def test_no_progress_callback_does_not_raise(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep()
        ctx = _ctx(_noisy())
        with patch.object(step._estimator, "estimate", return_value=_FAKE_ESTIMATE):
            step.execute(ctx)  # default noop_callback


# ---------------------------------------------------------------------------
# Integration — real estimator (no mock)
# ---------------------------------------------------------------------------

@patch("astroai.core.pipeline.adaptive_denoise_step.DenoiseStep")
class TestRealEstimator:
    def test_runs_end_to_end_with_real_estimator(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep(estimator_iterations=3)
        img = _noisy(h=64, w=64, sigma=0.03)
        ctx = _ctx(img)
        out = step.execute(ctx)
        strength = out.metadata.get(AdaptiveDenoiseStep.METADATA_KEY)
        assert strength is not None
        assert 0.0 <= strength <= 1.0

    def test_low_noise_image_gets_low_strength(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep(estimator_iterations=3)
        rng = np.random.default_rng(7)
        img = np.clip(rng.normal(0.5, 0.001, (64, 64)), 0, 1).astype(np.float32)
        ctx = _ctx(img)
        out = step.execute(ctx)
        strength = out.metadata[AdaptiveDenoiseStep.METADATA_KEY]
        assert strength < 0.5

    def test_high_noise_image_gets_higher_strength(self, MockDenoiseStep):
        mock_step = MagicMock()
        mock_step.execute.side_effect = lambda ctx, progress: ctx
        MockDenoiseStep.return_value = mock_step
        step = AdaptiveDenoiseStep(estimator_iterations=3)
        rng = np.random.default_rng(7)
        img_noisy = np.clip(rng.normal(0.3, 0.12, (64, 64)), 0, 1).astype(np.float32)
        img_clean = np.clip(rng.normal(0.3, 0.002, (64, 64)), 0, 1).astype(np.float32)
        ctx_noisy = _ctx(img_noisy)
        ctx_clean = _ctx(img_clean)
        out_noisy = step.execute(ctx_noisy)
        out_clean = step.execute(ctx_clean)
        assert (out_noisy.metadata[AdaptiveDenoiseStep.METADATA_KEY] >
                out_clean.metadata[AdaptiveDenoiseStep.METADATA_KEY])
