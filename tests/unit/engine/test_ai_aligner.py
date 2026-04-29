"""Tests for AIAligner, AIAlignmentStep, and AlignmentQualityPanel."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.engine.registration.ai_aligner import (
    AIAligner,
    AlignmentResult,
    _dlt_homography,
    _dlt_homography_lstsq,
    _dog_extract,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _star_field(
    h: int = 128,
    w: int = 128,
    positions: list[tuple[int, int]] | None = None,
    noise: float = 0.005,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    img = rng.random((h, w)).astype(np.float64) * noise
    if positions is None:
        positions = [(20, 30), (60, 70), (90, 40), (45, 100), (10, 80), (70, 20)]
    y_grid, x_grid = np.ogrid[:h, :w]
    for cy, cx in positions:
        img += np.exp(-((y_grid - cy) ** 2 + (x_grid - cx) ** 2) / (2 * 2.0 ** 2))
    return img.clip(0, 1)


def _translate(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(img)
    src_y = slice(max(-dy, 0), img.shape[0] + min(-dy, 0))
    src_x = slice(max(-dx, 0), img.shape[1] + min(-dx, 0))
    dst_y = slice(max(dy, 0), img.shape[0] + min(dy, 0))
    dst_x = slice(max(dx, 0), img.shape[1] + min(dx, 0))
    out[dst_y, dst_x] = img[src_y, src_x]
    return out


# ---------------------------------------------------------------------------
# AlignmentResult dataclass
# ---------------------------------------------------------------------------

class TestAlignmentResult:
    def test_fields_accessible(self) -> None:
        ar = AlignmentResult(
            aligned=np.zeros((4, 4)),
            transform=np.eye(3),
            confidence=0.8,
            inlier_count=10,
            keypoints_matched=12,
        )
        assert ar.confidence == pytest.approx(0.8)
        assert ar.inlier_count == 10
        assert ar.transform.shape == (3, 3)


# ---------------------------------------------------------------------------
# DLT helpers
# ---------------------------------------------------------------------------

class TestDLTHomography:
    def test_identity_from_four_points(self) -> None:
        pts = np.array([[10, 20], [50, 30], [80, 10], [40, 90]], dtype=np.float64)
        H = _dlt_homography(pts, pts)
        np.testing.assert_allclose(H, np.eye(3), atol=1e-6)

    def test_translation_recovered(self) -> None:
        src = np.array([[10, 20], [50, 30], [80, 10], [40, 90]], dtype=np.float64)
        dst = src + np.array([[5, 3]])
        H = _dlt_homography(src, dst)
        assert H[0, 2] == pytest.approx(5.0, abs=1e-4)
        assert H[1, 2] == pytest.approx(3.0, abs=1e-4)

    def test_lstsq_over_determined(self) -> None:
        rng = np.random.default_rng(7)
        src = rng.integers(10, 100, (10, 2)).astype(np.float64)
        dst = src + np.array([[7, -4]])
        H = _dlt_homography_lstsq(src, dst)
        assert H[0, 2] == pytest.approx(7.0, abs=0.1)
        assert H[1, 2] == pytest.approx(-4.0, abs=0.1)


# ---------------------------------------------------------------------------
# _dog_extract
# ---------------------------------------------------------------------------

class TestDogExtract:
    def test_returns_arrays(self) -> None:
        kpts, descs = _dog_extract(_star_field(), max_keypoints=64)
        assert kpts.ndim == 2 and kpts.shape[1] == 2
        assert descs.ndim == 2 and descs.shape[1] == 64

    def test_keypoints_and_descriptors_same_length(self) -> None:
        kpts, descs = _dog_extract(_star_field(), max_keypoints=64)
        assert len(kpts) == len(descs)

    def test_max_keypoints_respected(self) -> None:
        kpts, _ = _dog_extract(_star_field(), max_keypoints=5)
        assert len(kpts) <= 5

    def test_flat_image_returns_empty(self) -> None:
        kpts, descs = _dog_extract(np.zeros((32, 32)), max_keypoints=64)
        assert len(kpts) == 0 and len(descs) == 0


# ---------------------------------------------------------------------------
# AIAligner.align()
# ---------------------------------------------------------------------------

class TestAIAlignerAlign:
    def test_returns_alignment_result(self) -> None:
        aligner = AIAligner(rng_seed=0)
        ref = _star_field()
        tgt = _translate(ref, 5, 3)
        result = aligner.align(ref, tgt)
        assert isinstance(result, AlignmentResult)

    def test_aligned_shape_matches_input(self) -> None:
        aligner = AIAligner(rng_seed=0)
        ref = _star_field()
        tgt = _translate(ref, 5, 3)
        result = aligner.align(ref, tgt)
        assert result.aligned.shape == ref.shape

    def test_transform_is_3x3(self) -> None:
        aligner = AIAligner(rng_seed=0)
        ref = _star_field()
        tgt = _translate(ref, 5, 3)
        result = aligner.align(ref, tgt)
        assert result.transform.shape == (3, 3)

    def test_confidence_in_range(self) -> None:
        aligner = AIAligner(rng_seed=0)
        ref = _star_field()
        tgt = _translate(ref, 5, 3)
        result = aligner.align(ref, tgt)
        assert 0.0 <= result.confidence <= 1.0

    def test_translation_detected(self) -> None:
        aligner = AIAligner(rng_seed=42)
        ref = _star_field(noise=0.001)
        dy_true, dx_true = 6, 4
        tgt = _translate(ref, dy_true, dx_true)
        result = aligner.align(ref, tgt)
        # H maps ref→target: H[0,2]=dx, H[1,2]=dy
        assert abs(result.transform[0, 2] - dx_true) < 3.0
        assert abs(result.transform[1, 2] - dy_true) < 3.0

    def test_rgb_frame_shape_preserved(self) -> None:
        aligner = AIAligner(rng_seed=0)
        gray = _star_field()
        ref = np.stack([gray] * 3, axis=-1)
        tgt = np.stack([_translate(gray, 4, 2)] * 3, axis=-1)
        result = aligner.align(ref, tgt)
        assert result.aligned.shape == ref.shape

    def test_identity_when_no_keypoints(self) -> None:
        aligner = AIAligner(rng_seed=0)
        ref = np.zeros((64, 64), dtype=np.float64)
        tgt = np.zeros((64, 64), dtype=np.float64)
        result = aligner.align(ref, tgt)
        assert result.confidence == pytest.approx(0.0)
        np.testing.assert_array_equal(result.aligned, tgt)

    def test_align_batch_length(self) -> None:
        aligner = AIAligner(rng_seed=0)
        ref = _star_field()
        targets = [_translate(ref, i, i) for i in range(3)]
        results = aligner.align_batch(ref, targets)
        assert len(results) == 3
        assert all(isinstance(r, AlignmentResult) for r in results)


# ---------------------------------------------------------------------------
# AIAlignmentStep
# ---------------------------------------------------------------------------

class TestAIAlignmentStep:
    def test_name_and_stage(self) -> None:
        from astroai.core.pipeline.ai_alignment_step import AIAlignmentStep
        from astroai.core.pipeline.base import PipelineStage
        step = AIAlignmentStep()
        assert step.name == "AI Alignment"
        assert step.stage is PipelineStage.REGISTRATION

    def test_execute_stores_metadata(self) -> None:
        from astroai.core.pipeline.ai_alignment_step import AIAlignmentStep
        from astroai.core.pipeline.base import PipelineContext
        rng = np.random.default_rng(1)
        frames = [_star_field(rng=rng) for _ in range(3)]
        ctx = PipelineContext(images=frames)
        step = AIAlignmentStep(rng_seed=0)
        out = step.execute(ctx)
        assert "ai_alignment_scores" in out.metadata
        assert len(out.metadata["ai_alignment_scores"]) == 3

    def test_single_frame_passthrough(self) -> None:
        from astroai.core.pipeline.ai_alignment_step import AIAlignmentStep
        from astroai.core.pipeline.base import PipelineContext
        frame = _star_field()
        ctx = PipelineContext(images=[frame])
        out = AIAlignmentStep().execute(ctx)
        assert len(out.images) == 1

    def test_low_quality_frames_rejected(self) -> None:
        from astroai.core.pipeline.ai_alignment_step import AIAlignmentStep
        from astroai.core.pipeline.base import PipelineContext
        rng = np.random.default_rng(99)
        # Mix: good frame + pure noise (will have zero keypoints → conf=0)
        ref = _star_field(rng=rng)
        noisy = rng.random((128, 128)).astype(np.float64)
        ctx = PipelineContext(images=[ref, noisy])
        # threshold=0.99 forces rejection of anything with conf < 0.99
        step = AIAlignmentStep(quality_threshold=0.99, rng_seed=7)
        out = step.execute(ctx)
        # noisy frame should be rejected
        assert out.metadata["ai_alignment_rejected"][1] is True
        assert len(out.images) == 1  # only reference accepted

    def test_rejection_reasons_populated(self) -> None:
        from astroai.core.pipeline.ai_alignment_step import AIAlignmentStep
        from astroai.core.pipeline.base import PipelineContext
        rng = np.random.default_rng(5)
        ref = _star_field(rng=rng)
        noisy = rng.random((128, 128)).astype(np.float64)
        ctx = PipelineContext(images=[ref, noisy])
        out = AIAlignmentStep(quality_threshold=0.99, rng_seed=3).execute(ctx)
        reasons = out.metadata["ai_alignment_reasons"]
        assert reasons[0] == ""                # reference has no reason
        assert reasons[1] != ""                # rejected frame has a reason


# ---------------------------------------------------------------------------
# AlignmentQualityPanel (headless Qt)
# ---------------------------------------------------------------------------

class TestAlignmentQualityPanel:
    def test_construct(self, qtbot) -> None:
        from astroai.ui.models import PipelineModel
        from astroai.ui.widgets.alignment_quality_panel import AlignmentQualityPanel
        model = PipelineModel()
        panel = AlignmentQualityPanel(model)
        qtbot.addWidget(panel)
        assert panel._table.rowCount() == 0

    def test_update_results_populates_table(self, qtbot) -> None:
        from astroai.ui.models import PipelineModel
        from astroai.ui.widgets.alignment_quality_panel import AlignmentQualityPanel
        model = PipelineModel()
        panel = AlignmentQualityPanel(model)
        qtbot.addWidget(panel)
        panel.update_alignment_results(
            scores=[1.0, 0.8, 0.1],
            rejected=[False, False, True],
            reasons=["", "", "confidence=0.100 < threshold=0.300"],
            reference_index=0,
        )
        assert panel._table.rowCount() == 3
        assert panel._table.item(2, 2).text() == "Abgelehnt"

    def test_reset_clears_table(self, qtbot) -> None:
        from astroai.ui.models import PipelineModel
        from astroai.ui.widgets.alignment_quality_panel import AlignmentQualityPanel
        model = PipelineModel()
        panel = AlignmentQualityPanel(model)
        qtbot.addWidget(panel)
        panel.update_alignment_results([0.9], [False], [""])
        model.reset()
        assert panel._table.rowCount() == 0

    def test_reference_frame_labeled(self, qtbot) -> None:
        from astroai.ui.models import PipelineModel
        from astroai.ui.widgets.alignment_quality_panel import AlignmentQualityPanel
        model = PipelineModel()
        panel = AlignmentQualityPanel(model)
        qtbot.addWidget(panel)
        panel.update_alignment_results(
            scores=[1.0, 0.7],
            rejected=[False, False],
            reasons=["", ""],
            reference_index=0,
        )
        assert panel._table.item(0, 2).text() == "Referenz"
        assert panel._table.item(0, 1).text() == "—"
