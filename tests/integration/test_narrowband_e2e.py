"""End-to-end integration test for narrowband channel mapping.

Verifies: Ha + OIII FITS -> NarrowbandMapper (SHO/HOO) -> correct palette mapping output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from astroai.processing.channels.narrowband_mapper import (
    NarrowbandMapper,
    NarrowbandPalette,
)
from astroai.processing.channels.pipeline_step import (
    ChannelCombineStep,
    CombineMode,
)
from astroai.core.pipeline.base import PipelineContext, PipelineProgress


def make_narrowband_channel(
    height: int = 64,
    width: int = 64,
    emission_strength: float = 0.7,
    seed: int = 42,
) -> NDArray[np.floating[Any]]:
    """Generate synthetic narrowband channel with emission nebula pattern."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height // 2, width // 2
    nebula = emission_strength * np.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (height / 4) ** 2)
    )
    noise = rng.uniform(0, 0.05, (height, width))
    return (nebula + noise).astype(np.float32)


@pytest.fixture()
def ha_channel() -> NDArray[np.floating[Any]]:
    return make_narrowband_channel(64, 64, emission_strength=0.8, seed=1)


@pytest.fixture()
def oiii_channel() -> NDArray[np.floating[Any]]:
    return make_narrowband_channel(64, 64, emission_strength=0.6, seed=2)


@pytest.fixture()
def sii_channel() -> NDArray[np.floating[Any]]:
    return make_narrowband_channel(64, 64, emission_strength=0.4, seed=3)


class TestNarrowbandMapperE2E:
    """Integration tests for NarrowbandMapper palette mapping."""

    def test_sho_palette_mapping(
        self,
        ha_channel: NDArray[np.floating[Any]],
        oiii_channel: NDArray[np.floating[Any]],
        sii_channel: NDArray[np.floating[Any]],
    ) -> None:
        mapper = NarrowbandMapper()
        result = mapper.map(
            Ha=ha_channel, OIII=oiii_channel, SII=sii_channel,
            palette=NarrowbandPalette.SHO,
        )

        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32
        # SHO: R=SII, G=Ha, B=OIII
        np.testing.assert_allclose(result[..., 0], sii_channel, atol=1e-5)
        np.testing.assert_allclose(result[..., 1], ha_channel, atol=1e-5)
        np.testing.assert_allclose(result[..., 2], oiii_channel, atol=1e-5)

    def test_hoo_palette_mapping(
        self,
        ha_channel: NDArray[np.floating[Any]],
        oiii_channel: NDArray[np.floating[Any]],
    ) -> None:
        mapper = NarrowbandMapper()
        result = mapper.map(
            Ha=ha_channel, OIII=oiii_channel, SII=None,
            palette=NarrowbandPalette.HOO,
        )

        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32
        # HOO: R=Ha, G=OIII, B=OIII
        np.testing.assert_allclose(result[..., 0], ha_channel, atol=1e-5)
        np.testing.assert_allclose(result[..., 1], oiii_channel, atol=1e-5)
        np.testing.assert_allclose(result[..., 2], oiii_channel, atol=1e-5)

    def test_output_range(
        self,
        ha_channel: NDArray[np.floating[Any]],
        oiii_channel: NDArray[np.floating[Any]],
        sii_channel: NDArray[np.floating[Any]],
    ) -> None:
        mapper = NarrowbandMapper()
        result = mapper.map(
            Ha=ha_channel, OIII=oiii_channel, SII=sii_channel,
            palette=NarrowbandPalette.SHO,
        )

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_raises_on_all_none(self) -> None:
        mapper = NarrowbandMapper()
        with pytest.raises(ValueError):
            mapper.map(Ha=None, OIII=None, SII=None)

    def test_ha_oiii_only_sho(
        self,
        ha_channel: NDArray[np.floating[Any]],
        oiii_channel: NDArray[np.floating[Any]],
    ) -> None:
        mapper = NarrowbandMapper()
        result = mapper.map(
            Ha=ha_channel, OIII=oiii_channel, SII=None,
            palette=NarrowbandPalette.SHO,
        )

        assert result.shape == (64, 64, 3)
        assert np.isfinite(result).all()
        # R channel should be zeros (SII=None)
        assert result[..., 0].sum() == 0.0


class TestNarrowbandPipelineStepE2E:
    """Integration tests for narrowband via ChannelCombineStep."""

    def test_pipeline_step_narrowband_sho(
        self,
        ha_channel: NDArray[np.floating[Any]],
        oiii_channel: NDArray[np.floating[Any]],
        sii_channel: NDArray[np.floating[Any]],
    ) -> None:
        channels = {"Ha": ha_channel, "OIII": oiii_channel, "SII": sii_channel}
        step = ChannelCombineStep(
            mode=CombineMode.NARROWBAND,
            palette=NarrowbandPalette.SHO,
            channels=channels,
        )
        ctx = PipelineContext()

        out_ctx = step.execute(ctx)

        assert out_ctx.result is not None
        assert out_ctx.result.shape == (64, 64, 3)

    def test_pipeline_step_narrowband_hoo(
        self,
        ha_channel: NDArray[np.floating[Any]],
        oiii_channel: NDArray[np.floating[Any]],
    ) -> None:
        channels = {"Ha": ha_channel, "OIII": oiii_channel}
        step = ChannelCombineStep(
            mode=CombineMode.NARROWBAND,
            palette=NarrowbandPalette.HOO,
            channels=channels,
        )
        ctx = PipelineContext()

        out_ctx = step.execute(ctx)

        assert out_ctx.result is not None
        assert out_ctx.result.shape == (64, 64, 3)
        assert out_ctx.result.dtype == np.float32

    def test_pipeline_step_progress_callback(
        self,
        ha_channel: NDArray[np.floating[Any]],
        oiii_channel: NDArray[np.floating[Any]],
        sii_channel: NDArray[np.floating[Any]],
    ) -> None:
        channels = {"Ha": ha_channel, "OIII": oiii_channel, "SII": sii_channel}
        step = ChannelCombineStep(
            mode=CombineMode.NARROWBAND,
            palette=NarrowbandPalette.SHO,
            channels=channels,
        )
        ctx = PipelineContext()
        calls: list[PipelineProgress] = []

        step.execute(ctx, progress=lambda p: calls.append(p))

        assert len(calls) >= 2
