"""Unit tests for ChannelCombiner and NarrowbandMapper."""
from __future__ import annotations
import numpy as np
import pytest
from astroai.processing.channels.combiner import ChannelCombiner
from astroai.processing.channels.narrowband_mapper import NarrowbandMapper, NarrowbandPalette


def _ch(value: float, shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


# ---------------------------------------------------------------------------
# ChannelCombiner
# ---------------------------------------------------------------------------

class TestChannelCombiner:
    def test_raises_with_no_channels(self) -> None:
        combiner = ChannelCombiner()
        with pytest.raises(ValueError, match="At least one channel"):
            combiner.combine_lrgb(L=None, R=None, G=None, B=None)

    def test_output_shape_rgb(self) -> None:
        combiner = ChannelCombiner()
        r, g, b = _ch(0.6), _ch(0.5), _ch(0.4)
        out = combiner.combine_lrgb(L=None, R=r, G=g, B=b)
        assert out.shape == (16, 16, 3)

    def test_output_dtype_float32(self) -> None:
        combiner = ChannelCombiner()
        out = combiner.combine_lrgb(L=None, R=_ch(0.5), G=_ch(0.5), B=_ch(0.5))
        assert out.dtype == np.float32

    def test_rgb_without_l_returns_unmodified(self) -> None:
        combiner = ChannelCombiner()
        r, g, b = _ch(0.4), _ch(0.5), _ch(0.6)
        out = combiner.combine_lrgb(L=None, R=r, G=g, B=b)
        # channel 0 should be R
        np.testing.assert_allclose(out[..., 0], r, atol=1e-5)

    def test_missing_channel_becomes_zero(self) -> None:
        combiner = ChannelCombiner()
        out = combiner.combine_lrgb(L=None, R=_ch(0.5), G=None, B=None)
        # G and B channels should be zero
        np.testing.assert_allclose(out[..., 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[..., 2], 0.0, atol=1e-5)

    def test_output_clipped_to_0_1(self) -> None:
        combiner = ChannelCombiner()
        out = combiner.combine_lrgb(L=_ch(1.0), R=_ch(0.9), G=_ch(0.8), B=_ch(0.7))
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_lrgb_luminance_scaling(self) -> None:
        combiner = ChannelCombiner()
        r, g, b = _ch(0.3), _ch(0.3), _ch(0.3)
        L_bright = _ch(0.9)
        out = combiner.combine_lrgb(L=L_bright, R=r, G=g, B=b)
        # After luminance scaling, result should be brighter than original RGB
        out_no_l = combiner.combine_lrgb(L=None, R=r, G=g, B=b)
        assert float(out.mean()) > float(out_no_l.mean())

    def test_only_l_provided(self) -> None:
        combiner = ChannelCombiner()
        L = _ch(0.7)
        out = combiner.combine_lrgb(L=L, R=None, G=None, B=None)
        assert out.shape == (16, 16, 3)
        # All colour channels are zero → RGB = 0 → result black
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_3d_channel_uses_first_slice(self) -> None:
        combiner = ChannelCombiner()
        r_3d = np.full((16, 16, 3), 0.5, dtype=np.float32)
        out = combiner.combine_lrgb(L=None, R=r_3d, G=_ch(0.5), B=_ch(0.5))
        assert out.shape == (16, 16, 3)


# ---------------------------------------------------------------------------
# NarrowbandMapper
# ---------------------------------------------------------------------------

class TestNarrowbandMapper:
    def test_raises_with_no_channels(self) -> None:
        mapper = NarrowbandMapper()
        with pytest.raises(ValueError, match="At least one narrowband channel"):
            mapper.map(Ha=None, OIII=None, SII=None)

    def test_output_shape(self) -> None:
        mapper = NarrowbandMapper()
        out = mapper.map(Ha=_ch(0.5), OIII=_ch(0.3), SII=_ch(0.4))
        assert out.shape == (16, 16, 3)

    def test_output_dtype_float32(self) -> None:
        mapper = NarrowbandMapper()
        out = mapper.map(Ha=_ch(0.5), OIII=_ch(0.3), SII=_ch(0.4))
        assert out.dtype == np.float32

    def test_sho_channel_assignment(self) -> None:
        # SHO: R=SII, G=Ha, B=OIII
        mapper = NarrowbandMapper()
        Ha = _ch(0.1)
        OIII = _ch(0.2)
        SII = _ch(0.3)
        out = mapper.map(Ha=Ha, OIII=OIII, SII=SII, palette=NarrowbandPalette.SHO)
        np.testing.assert_allclose(out[..., 0], SII, atol=1e-5)
        np.testing.assert_allclose(out[..., 1], Ha, atol=1e-5)
        np.testing.assert_allclose(out[..., 2], OIII, atol=1e-5)

    def test_hoo_channel_assignment(self) -> None:
        # HOO: R=Ha, G=OIII, B=OIII
        mapper = NarrowbandMapper()
        Ha = _ch(0.6)
        OIII = _ch(0.4)
        out = mapper.map(Ha=Ha, OIII=OIII, SII=None, palette=NarrowbandPalette.HOO)
        np.testing.assert_allclose(out[..., 0], Ha, atol=1e-5)
        np.testing.assert_allclose(out[..., 1], OIII, atol=1e-5)
        np.testing.assert_allclose(out[..., 2], OIII, atol=1e-5)

    def test_nho_channel_assignment(self) -> None:
        # NHO: R=Ha, G=Ha, B=OIII
        mapper = NarrowbandMapper()
        Ha = _ch(0.5)
        OIII = _ch(0.3)
        out = mapper.map(Ha=Ha, OIII=OIII, SII=None, palette=NarrowbandPalette.NHO)
        np.testing.assert_allclose(out[..., 0], Ha, atol=1e-5)
        np.testing.assert_allclose(out[..., 1], Ha, atol=1e-5)
        np.testing.assert_allclose(out[..., 2], OIII, atol=1e-5)

    def test_missing_channel_becomes_zero(self) -> None:
        mapper = NarrowbandMapper()
        out = mapper.map(Ha=_ch(0.5), OIII=None, SII=None, palette=NarrowbandPalette.SHO)
        # SII and OIII missing → R=0 (SII), B=0 (OIII)
        np.testing.assert_allclose(out[..., 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[..., 2], 0.0, atol=1e-5)

    def test_output_clipped_to_0_1(self) -> None:
        mapper = NarrowbandMapper()
        # Values at 1.0 should still be clamped
        out = mapper.map(Ha=_ch(1.0), OIII=_ch(1.0), SII=_ch(1.0))
        assert float(out.max()) <= 1.0
        assert float(out.min()) >= 0.0

    def test_palette_enum_values(self) -> None:
        assert NarrowbandPalette.SHO.value == "sho"
        assert NarrowbandPalette.HOO.value == "hoo"
        assert NarrowbandPalette.NHO.value == "nho"
