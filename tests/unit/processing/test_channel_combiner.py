"""Tests for ChannelCombiner, NarrowbandMapper, and ChannelCombineStep."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.core.pipeline.base import PipelineContext, PipelineStage
from astroai.processing.channels import (
    ChannelCombineStep,
    ChannelCombiner,
    CombineMode,
    NarrowbandMapper,
    NarrowbandPalette,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 32, 48

def _chan(seed: int = 0, lo: float = 0.1, hi: float = 0.9) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(lo, hi, size=(H, W)).astype(np.float32)


# ---------------------------------------------------------------------------
# ChannelCombiner
# ---------------------------------------------------------------------------

class TestChannelCombiner:
    def setup_method(self) -> None:
        self.c = ChannelCombiner()

    def test_lrgb_full_shape(self) -> None:
        r = self.c.combine_lrgb(_chan(0), _chan(1), _chan(2), _chan(3))
        assert r.shape == (H, W, 3)
        assert r.dtype == np.float32

    def test_lrgb_range(self) -> None:
        r = self.c.combine_lrgb(_chan(0), _chan(1), _chan(2), _chan(3))
        assert float(r.min()) >= 0.0
        assert float(r.max()) <= 1.0 + 1e-6

    def test_lrgb_none_L_returns_rgb(self) -> None:
        r = self.c.combine_lrgb(None, _chan(1), _chan(2), _chan(3))
        assert r.shape == (H, W, 3)
        np.testing.assert_allclose(r[..., 0], np.clip(_chan(1), 0, 1), atol=1e-5)

    def test_lrgb_none_colour_channels(self) -> None:
        r = self.c.combine_lrgb(_chan(0), None, None, None)
        assert r.shape == (H, W, 3)
        # All colour channels zero → RGB is black, result is black
        np.testing.assert_allclose(r, 0.0, atol=1e-5)

    def test_lrgb_luminance_applied(self) -> None:
        L = np.full((H, W), 0.5, dtype=np.float32)
        R = np.full((H, W), 0.4, dtype=np.float32)
        G = np.full((H, W), 0.4, dtype=np.float32)
        B = np.full((H, W), 0.4, dtype=np.float32)
        r = self.c.combine_lrgb(L, R, G, B)
        # Result luminance should be close to the L value
        lum = 0.299 * r[..., 0] + 0.587 * r[..., 1] + 0.114 * r[..., 2]
        np.testing.assert_allclose(lum, 0.5, atol=0.05)

    def test_lrgb_no_channels_raises(self) -> None:
        with pytest.raises(ValueError):
            self.c.combine_lrgb(None, None, None, None)

    def test_lrgb_3d_input_handled(self) -> None:
        # 3-channel input treated as grayscale via first channel
        ch = np.stack([_chan(0), _chan(1), _chan(2)], axis=-1)
        r = self.c.combine_lrgb(ch, ch, ch, ch)
        assert r.shape == (H, W, 3)


# ---------------------------------------------------------------------------
# NarrowbandMapper
# ---------------------------------------------------------------------------

class TestNarrowbandMapper:
    def setup_method(self) -> None:
        self.m = NarrowbandMapper()
        self.ha = _chan(10)
        self.oiii = _chan(11)
        self.sii = _chan(12)

    def test_sho_channels(self) -> None:
        r = self.m.map(self.ha, self.oiii, self.sii, NarrowbandPalette.SHO)
        assert r.shape == (H, W, 3)
        np.testing.assert_allclose(r[..., 0], self.sii, atol=1e-6)   # R=SII
        np.testing.assert_allclose(r[..., 1], self.ha,  atol=1e-6)   # G=Ha
        np.testing.assert_allclose(r[..., 2], self.oiii, atol=1e-6)  # B=OIII

    def test_hoo_channels(self) -> None:
        r = self.m.map(self.ha, self.oiii, self.sii, NarrowbandPalette.HOO)
        np.testing.assert_allclose(r[..., 0], self.ha,   atol=1e-6)  # R=Ha
        np.testing.assert_allclose(r[..., 1], self.oiii, atol=1e-6)  # G=OIII
        np.testing.assert_allclose(r[..., 2], self.oiii, atol=1e-6)  # B=OIII

    def test_nho_channels(self) -> None:
        r = self.m.map(self.ha, self.oiii, self.sii, NarrowbandPalette.NHO)
        np.testing.assert_allclose(r[..., 0], self.ha,   atol=1e-6)  # R=NII≈Ha
        np.testing.assert_allclose(r[..., 1], self.ha,   atol=1e-6)  # G=Ha
        np.testing.assert_allclose(r[..., 2], self.oiii, atol=1e-6)  # B=OIII

    def test_range(self) -> None:
        r = self.m.map(self.ha, self.oiii, self.sii, NarrowbandPalette.SHO)
        assert float(r.min()) >= 0.0
        assert float(r.max()) <= 1.0 + 1e-6

    def test_dtype(self) -> None:
        r = self.m.map(self.ha, self.oiii, self.sii, NarrowbandPalette.HOO)
        assert r.dtype == np.float32

    def test_none_channel_zeros(self) -> None:
        r = self.m.map(self.ha, self.oiii, None, NarrowbandPalette.SHO)
        np.testing.assert_allclose(r[..., 0], 0.0, atol=1e-6)  # SII missing → zeros

    def test_all_none_raises(self) -> None:
        with pytest.raises(ValueError):
            self.m.map(None, None, None, NarrowbandPalette.SHO)


# ---------------------------------------------------------------------------
# ChannelCombineStep
# ---------------------------------------------------------------------------

class TestChannelCombineStep:
    def test_name(self) -> None:
        step = ChannelCombineStep()
        assert step.name == "Channel Combine"

    def test_stage(self) -> None:
        step = ChannelCombineStep()
        assert step.stage is PipelineStage.PROCESSING

    def test_lrgb_mode_sets_result(self) -> None:
        channels = {"L": _chan(0), "R": _chan(1), "G": _chan(2), "B": _chan(3)}
        step = ChannelCombineStep(mode=CombineMode.LRGB, channels=channels)
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (H, W, 3)

    def test_narrowband_mode_sets_result(self) -> None:
        channels = {"Ha": _chan(10), "OIII": _chan(11), "SII": _chan(12)}
        step = ChannelCombineStep(
            mode=CombineMode.NARROWBAND,
            palette=NarrowbandPalette.SHO,
            channels=channels,
        )
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.result is not None
        assert out.result.shape == (H, W, 3)

    def test_set_channels_updates(self) -> None:
        step = ChannelCombineStep(mode=CombineMode.LRGB)
        step.set_channels({"R": _chan(1), "G": _chan(2), "B": _chan(3)})
        ctx = PipelineContext()
        out = step.execute(ctx)
        assert out.result is not None

    def test_progress_callback_called(self) -> None:
        calls: list = []
        channels = {"R": _chan(1), "G": _chan(2), "B": _chan(3)}
        step = ChannelCombineStep(mode=CombineMode.LRGB, channels=channels)
        step.execute(PipelineContext(), progress=lambda p: calls.append(p))
        assert len(calls) == 2
        assert calls[0].current == 0
        assert calls[1].current == 1
