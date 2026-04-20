"""End-to-end integration test for LRGB channel combination.

Verifies: 4 separate FITS (L, R, G, B) -> ChannelCombiner -> LRGB stack with correct combination.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.processing.channels.combiner import ChannelCombiner
from astroai.processing.channels.pipeline_step import ChannelCombineStep, CombineMode
from astroai.core.pipeline.base import PipelineContext, PipelineProgress


def make_channel(
    height: int = 64,
    width: int = 64,
    base_level: float = 0.5,
    seed: int = 42,
) -> NDArray[np.floating[Any]]:
    """Generate a single-channel synthetic frame."""
    rng = np.random.default_rng(seed)
    return rng.uniform(
        base_level * 0.8, base_level * 1.2, (height, width)
    ).astype(np.float32)


@pytest.fixture()
def lrgb_channels() -> dict[str, NDArray[np.floating[Any]]]:
    """4 separate channel arrays simulating L, R, G, B FITS data."""
    return {
        "L": make_channel(64, 64, base_level=0.6, seed=1),
        "R": make_channel(64, 64, base_level=0.4, seed=2),
        "G": make_channel(64, 64, base_level=0.5, seed=3),
        "B": make_channel(64, 64, base_level=0.3, seed=4),
    }


class TestChannelCombinerE2E:
    """Integration tests for ChannelCombiner LRGB workflow."""

    def test_combine_lrgb_produces_rgb(
        self, lrgb_channels: dict[str, NDArray[np.floating[Any]]]
    ) -> None:
        combiner = ChannelCombiner()
        result = combiner.combine_lrgb(
            L=lrgb_channels["L"],
            R=lrgb_channels["R"],
            G=lrgb_channels["G"],
            B=lrgb_channels["B"],
        )

        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32

    def test_combine_lrgb_output_range(
        self, lrgb_channels: dict[str, NDArray[np.floating[Any]]]
    ) -> None:
        combiner = ChannelCombiner()
        result = combiner.combine_lrgb(
            L=lrgb_channels["L"],
            R=lrgb_channels["R"],
            G=lrgb_channels["G"],
            B=lrgb_channels["B"],
        )

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_combine_without_luminance(
        self, lrgb_channels: dict[str, NDArray[np.floating[Any]]]
    ) -> None:
        combiner = ChannelCombiner()
        result = combiner.combine_lrgb(
            L=None,
            R=lrgb_channels["R"],
            G=lrgb_channels["G"],
            B=lrgb_channels["B"],
        )

        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32

    def test_combine_rgb_only_no_luminance_modification(
        self, lrgb_channels: dict[str, NDArray[np.floating[Any]]]
    ) -> None:
        combiner = ChannelCombiner()
        result = combiner.combine_lrgb(
            L=None,
            R=lrgb_channels["R"],
            G=lrgb_channels["G"],
            B=lrgb_channels["B"],
        )

        np.testing.assert_allclose(result[..., 0], lrgb_channels["R"], atol=1e-5)
        np.testing.assert_allclose(result[..., 1], lrgb_channels["G"], atol=1e-5)
        np.testing.assert_allclose(result[..., 2], lrgb_channels["B"], atol=1e-5)

    def test_combine_raises_on_all_none(self) -> None:
        combiner = ChannelCombiner()
        with pytest.raises(ValueError):
            combiner.combine_lrgb(L=None, R=None, G=None, B=None)

    def test_combine_partial_channels(self) -> None:
        combiner = ChannelCombiner()
        r_only = make_channel(32, 32, base_level=0.5, seed=10)
        result = combiner.combine_lrgb(L=None, R=r_only, G=None, B=None)

        assert result.shape == (32, 32, 3)
        assert result[..., 0].sum() > 0
        assert result[..., 1].sum() == 0.0
        assert result[..., 2].sum() == 0.0


class TestChannelCombineStepE2E:
    """Integration tests for ChannelCombineStep pipeline integration."""

    def test_pipeline_step_lrgb(
        self, lrgb_channels: dict[str, NDArray[np.floating[Any]]]
    ) -> None:
        step = ChannelCombineStep(mode=CombineMode.LRGB, channels=lrgb_channels)
        ctx = PipelineContext()

        out_ctx = step.execute(ctx)

        assert out_ctx.result is not None
        assert out_ctx.result.shape == (64, 64, 3)
        assert out_ctx.result.dtype == np.float32

    def test_pipeline_step_progress(
        self, lrgb_channels: dict[str, NDArray[np.floating[Any]]]
    ) -> None:
        step = ChannelCombineStep(mode=CombineMode.LRGB, channels=lrgb_channels)
        ctx = PipelineContext()
        calls: list[PipelineProgress] = []

        step.execute(ctx, progress=lambda p: calls.append(p))

        assert len(calls) >= 2

    def test_pipeline_step_set_channels(self) -> None:
        step = ChannelCombineStep(mode=CombineMode.LRGB)
        channels = {
            "L": make_channel(32, 32, 0.5, seed=20),
            "R": make_channel(32, 32, 0.4, seed=21),
            "G": make_channel(32, 32, 0.4, seed=22),
            "B": make_channel(32, 32, 0.3, seed=23),
        }
        step.set_channels(channels)
        ctx = PipelineContext()

        out_ctx = step.execute(ctx)

        assert out_ctx.result is not None
        assert out_ctx.result.shape == (32, 32, 3)
