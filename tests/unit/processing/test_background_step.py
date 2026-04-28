"""Unit tests for BackgroundRemovalStep and ChannelCombineStep."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.processing.background.pipeline_step import BackgroundRemovalStep
from astroai.processing.channels.pipeline_step import ChannelCombineStep, CombineMode
from astroai.processing.channels.narrowband_mapper import NarrowbandPalette


def _img(value: float = 100.0, shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


# ---------------------------------------------------------------------------
# BackgroundRemovalStep
# ---------------------------------------------------------------------------

class TestBackgroundRemovalStep:
    def test_name(self) -> None:
        step = BackgroundRemovalStep()
        assert step.name == "Background Removal"

    def test_stage(self) -> None:
        step = BackgroundRemovalStep()
        assert step.stage == PipelineStage.PROCESSING

    def test_empty_context_unchanged(self) -> None:
        step = BackgroundRemovalStep()
        ctx = PipelineContext()
        result = step.execute(ctx)
        assert result.result is None
        assert result.images == []

    def test_processes_result_image(self) -> None:
        step = BackgroundRemovalStep()
        ctx = PipelineContext(result=_img(200.0))
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (32, 32)

    def test_processes_images_list(self) -> None:
        step = BackgroundRemovalStep()
        frames = [_img(100.0 + i) for i in range(3)]
        ctx = PipelineContext(images=frames)
        out = step.execute(ctx)
        assert len(out.images) == 3
        for img in out.images:
            assert img.shape == (32, 32)

    def test_progress_called_per_frame(self) -> None:
        step = BackgroundRemovalStep()
        frames = [_img(float(i + 1)) for i in range(4)]
        ctx = PipelineContext(images=frames)
        calls: list = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) == 4

    def test_result_takes_priority_over_images(self) -> None:
        step = BackgroundRemovalStep()
        # Both result and images present — result takes priority
        ctx = PipelineContext(result=_img(100.0), images=[_img(50.0)])
        out = step.execute(ctx)
        # result was processed (not None)
        assert out.result is not None

    def test_output_dtype_preserved(self) -> None:
        step = BackgroundRemovalStep()
        ctx = PipelineContext(result=_img(100.0).astype(np.float32))
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.dtype == np.float32

    def test_output_clipped_non_negative(self) -> None:
        step = BackgroundRemovalStep()
        ctx = PipelineContext(result=_img(100.0))
        out = step.execute(ctx)
        assert out.result is not None
        assert float(out.result.min()) >= 0.0


# ---------------------------------------------------------------------------
# ChannelCombineStep
# ---------------------------------------------------------------------------

class TestChannelCombineStep:
    def test_name(self) -> None:
        step = ChannelCombineStep()
        assert step.name == "Channel Combine"

    def test_stage(self) -> None:
        step = ChannelCombineStep()
        assert step.stage == PipelineStage.PROCESSING

    def test_lrgb_mode_with_channels(self) -> None:
        L = _img(0.5)
        R = _img(0.4)
        G = _img(0.5)
        B = _img(0.3)
        step = ChannelCombineStep(
            mode=CombineMode.LRGB,
            channels={"L": L, "R": R, "G": G, "B": B},
        )
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape[:2] == (32, 32)

    def test_narrowband_sho_mode(self) -> None:
        Ha = _img(0.5)
        OIII = _img(0.3)
        SII = _img(0.4)
        step = ChannelCombineStep(
            mode=CombineMode.NARROWBAND,
            palette=NarrowbandPalette.SHO,
            channels={"Ha": Ha, "OIII": OIII, "SII": SII},
        )
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.result is not None

    def test_empty_channels_result_none(self) -> None:
        step = ChannelCombineStep(mode=CombineMode.LRGB)
        ctx = PipelineContext()
        out = step.execute(ctx)
        # No channels set → combine returns None, result stays None
        assert out.result is None

    def test_set_channels_updates_channels(self) -> None:
        step = ChannelCombineStep()
        R = _img(0.6)
        G = _img(0.5)
        B = _img(0.4)
        step.set_channels({"R": R, "G": G, "B": B})
        ctx = PipelineContext()
        out = step.execute(ctx)
        # RGB without L still goes through combiner (may or may not produce result)
        # Just ensure no exception
        assert isinstance(out, PipelineContext)

    def test_progress_called(self) -> None:
        step = ChannelCombineStep()
        ctx = PipelineContext()
        calls: list = []
        step.execute(ctx, progress=lambda p: calls.append(p))
        assert len(calls) >= 2  # start and end

    def test_combine_mode_enum_values(self) -> None:
        assert CombineMode.LRGB.value == "lrgb"
        assert CombineMode.NARROWBAND.value == "narrowband"

    def test_context_unchanged_if_combine_returns_none(self) -> None:
        step = ChannelCombineStep(mode=CombineMode.LRGB)
        ctx = PipelineContext(result=_img(100.0))
        original_result = ctx.result
        out = step.execute(ctx)
        # No channels → combine = None → existing result should be untouched
        assert out.result is not None
