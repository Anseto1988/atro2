"""Tests for StarAligner: star-detection path and phase-correlation fallback."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.engine.registration.star_aligner import StarAligner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_star_field(
    h: int = 128,
    w: int = 128,
    positions: list[tuple[int, int]] | None = None,
    sigma: float = 2.0,
    noise: float = 0.005,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    img = rng.random((h, w)).astype(np.float64) * noise
    if positions is None:
        positions = [(20, 30), (60, 70), (90, 40), (45, 100), (10, 80)]
    y_grid, x_grid = np.ogrid[:h, :w]
    for cy, cx in positions:
        img += np.exp(-((y_grid - cy) ** 2 + (x_grid - cx) ** 2) / (2 * sigma ** 2))
    return img


def _shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(img)
    src_y = slice(max(-dy, 0), img.shape[0] + min(-dy, 0))
    src_x = slice(max(-dx, 0), img.shape[1] + min(-dx, 0))
    dst_y = slice(max(dy, 0), img.shape[0] + min(dy, 0))
    dst_x = slice(max(dx, 0), img.shape[1] + min(dx, 0))
    out[dst_y, dst_x] = img[src_y, src_x]
    return out


# ---------------------------------------------------------------------------
# Star detection path
# ---------------------------------------------------------------------------

class TestStarAlignerStarPath:
    def test_returns_tuple(self) -> None:
        aligner = StarAligner()
        ref = _make_star_field()
        tgt = _shift_image(ref, 5, 3)
        result = aligner.align(ref, tgt)
        assert isinstance(result, tuple) and len(result) == 2

    def test_transform_is_3x3(self) -> None:
        aligner = StarAligner()
        ref = _make_star_field()
        tgt = _shift_image(ref, 5, 3)
        _, transform = aligner.align(ref, tgt)
        assert transform.shape == (3, 3)

    def test_aligned_shape_matches_input(self) -> None:
        aligner = StarAligner()
        ref = _make_star_field()
        tgt = _shift_image(ref, 5, 3)
        aligned, _ = aligner.align(ref, tgt)
        assert aligned.shape == ref.shape

    def test_detected_shift_approximately_correct(self) -> None:
        aligner = StarAligner(log_sigma=2.0, detection_threshold=0.01)
        ref = _make_star_field(noise=0.001)
        dy_true, dx_true = 5, 3
        tgt = _shift_image(ref, dy_true, dx_true)
        _, transform = aligner.align(ref, tgt)
        assert abs(transform[1, 2] - dy_true) < 2.0
        assert abs(transform[0, 2] - dx_true) < 2.0

    def test_rgb_frame_aligned(self) -> None:
        aligner = StarAligner()
        gray = _make_star_field()
        ref = np.stack([gray, gray, gray], axis=-1)
        tgt = np.stack([_shift_image(gray, 4, 2)] * 3, axis=-1)
        aligned, _ = aligner.align(ref, tgt)
        assert aligned.shape == ref.shape

    def test_align_batch_length(self) -> None:
        aligner = StarAligner()
        ref = _make_star_field()
        targets = [_shift_image(ref, i, i) for i in range(3)]
        result = aligner.align_batch(ref, targets)
        assert len(result) == 3

    def test_metadata_method_star(self) -> None:
        from astroai.engine.registration.pipeline_step import RegistrationStep
        from astroai.core.pipeline.base import PipelineContext
        rng = np.random.default_rng(0)
        frames = [_make_star_field(rng=rng) for _ in range(2)]
        ctx = PipelineContext(images=frames)
        step = RegistrationStep(method="star")
        out = step.execute(ctx)
        assert out.metadata["registration_method"] == "star"


# ---------------------------------------------------------------------------
# Phase-correlation fallback (< 3 stars)
# ---------------------------------------------------------------------------

class TestStarAlignerFallback:
    def test_fallback_when_no_stars(self) -> None:
        """Pure noise → no stars detected → phase correlation fallback."""
        aligner = StarAligner(detection_threshold=1.0)  # threshold so high no stars pass
        rng = np.random.default_rng(7)
        ref = rng.random((64, 64)).astype(np.float64)
        tgt = np.roll(ref, 4, axis=1)
        aligned, transform = aligner.align(ref, tgt)
        assert aligned.shape == ref.shape
        assert transform.shape == (3, 3)

    def test_fallback_returns_nonzero_shift(self) -> None:
        """Fallback phase correlation must detect a real shift."""
        aligner = StarAligner(detection_threshold=1.0)
        rng = np.random.default_rng(99)
        ref = rng.random((64, 64)).astype(np.float64)
        tgt = np.roll(ref, 5, axis=0)
        _, transform = aligner.align(ref, tgt)
        # Phase correlation must find some shift
        total_shift = abs(transform[0, 2]) + abs(transform[1, 2])
        assert total_shift > 0.0

    def test_metadata_method_phase_correlation(self) -> None:
        from astroai.engine.registration.pipeline_step import RegistrationStep
        from astroai.core.pipeline.base import PipelineContext
        rng = np.random.default_rng(0)
        frames = [rng.random((32, 32)).astype(np.float32) for _ in range(2)]
        ctx = PipelineContext(images=frames)
        step = RegistrationStep(method="phase_correlation")
        out = step.execute(ctx)
        assert out.metadata["registration_method"] == "phase_correlation"


# ---------------------------------------------------------------------------
# RegistrationStep method parameter wiring
# ---------------------------------------------------------------------------

class TestRegistrationStepMethod:
    def test_default_method_is_star(self) -> None:
        from astroai.engine.registration.pipeline_step import RegistrationStep
        from astroai.engine.registration.star_aligner import StarAligner
        step = RegistrationStep()
        assert isinstance(step._aligner, StarAligner)

    def test_explicit_phase_correlation(self) -> None:
        from astroai.engine.registration.pipeline_step import RegistrationStep
        from astroai.engine.registration.aligner import FrameAligner
        step = RegistrationStep(method="phase_correlation")
        assert isinstance(step._aligner, FrameAligner)

    def test_explicit_star(self) -> None:
        from astroai.engine.registration.pipeline_step import RegistrationStep
        from astroai.engine.registration.star_aligner import StarAligner
        step = RegistrationStep(method="star")
        assert isinstance(step._aligner, StarAligner)
