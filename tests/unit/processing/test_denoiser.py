"""Targeted tests to close coverage gaps in denoiser.py.

Covers lines: 25-26, 36, 43-54, 57-64, 68-72, 99-100, 127, 129, 133-171,
              174-182, 235-239, 242-246, 249-261, 264-298
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 64, w: int = 64, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(100, 1000, (h, w)).astype(np.float64)


def _make_rgb_frame(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(0, 500, (h, w, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# _DoubleConv (lines 25-26, 36)
# ---------------------------------------------------------------------------

class TestDoubleConv:
    def test_init_creates_sequential(self) -> None:
        from astroai.processing.denoise.denoiser import _DoubleConv
        dc = _DoubleConv(1, 16)
        assert isinstance(dc.conv, torch.nn.Sequential)

    def test_forward_returns_tensor(self) -> None:
        from astroai.processing.denoise.denoiser import _DoubleConv
        dc = _DoubleConv(1, 8)
        x = torch.zeros(1, 1, 16, 16)
        out = dc(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 8, 16, 16)


# ---------------------------------------------------------------------------
# SimpleUNet (lines 43-54, 57-64, 68-72)
# ---------------------------------------------------------------------------

class TestSimpleUNet:
    def test_init_attributes(self) -> None:
        from astroai.processing.denoise.denoiser import SimpleUNet
        net = SimpleUNet(in_channels=1, out_channels=1)
        for attr in ["enc1", "enc2", "enc3", "bottleneck", "up3", "dec3",
                     "up2", "dec2", "up1", "dec1", "out_conv"]:
            assert hasattr(net, attr)

    def test_forward_shape(self) -> None:
        from astroai.processing.denoise.denoiser import SimpleUNet
        net = SimpleUNet(in_channels=1, out_channels=1)
        x = torch.zeros(1, 1, 64, 64)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 1, 64, 64)

    def test_forward_multichannel(self) -> None:
        from astroai.processing.denoise.denoiser import SimpleUNet
        net = SimpleUNet(in_channels=3, out_channels=3)
        x = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 3, 32, 32)

    def test_match_and_cat_pads_correctly(self) -> None:
        from astroai.processing.denoise.denoiser import SimpleUNet
        # up is smaller than skip -> should be padded and concatenated
        up = torch.zeros(1, 8, 14, 14)
        skip = torch.zeros(1, 8, 16, 16)
        result = SimpleUNet._match_and_cat(up, skip)
        assert result.shape == (1, 16, 16, 16)

    def test_match_and_cat_no_padding_needed(self) -> None:
        from astroai.processing.denoise.denoiser import SimpleUNet
        up = torch.zeros(1, 4, 8, 8)
        skip = torch.zeros(1, 4, 8, 8)
        result = SimpleUNet._match_and_cat(up, skip)
        assert result.shape == (1, 8, 8, 8)


# ---------------------------------------------------------------------------
# Denoiser with torch model (lines 99-100, 127, 133-171)
# ---------------------------------------------------------------------------

class TestDenoiserTorchModel:
    def _make_tiny_model(self) -> torch.nn.Module:
        class TinyModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x
        return TinyModel()

    def test_init_calls_eval(self) -> None:
        """Lines 99-100: model.eval() and dm.to_device(model) are called."""
        from astroai.processing.denoise.denoiser import Denoiser
        model = self._make_tiny_model()
        model.eval = MagicMock(wraps=model.eval)
        Denoiser(model=model)
        model.eval.assert_called_once()

    def test_denoise_single_dispatches_to_torch(self) -> None:
        """Line 127: _denoise_single -> _denoise_torch when model is set."""
        from astroai.processing.denoise.denoiser import Denoiser
        model = self._make_tiny_model()
        d = Denoiser(model=model, strength=1.0, tile_size=32, tile_overlap=4)
        frame = _make_frame(64, 64)
        result = d._denoise_single(frame)
        assert result.shape == frame.shape

    def test_denoise_torch_tiling(self) -> None:
        """Lines 133-171: full tiling loop in _denoise_torch."""
        from astroai.processing.denoise.denoiser import Denoiser
        model = self._make_tiny_model()
        # tile_size smaller than image forces multiple tiles
        d = Denoiser(model=model, strength=1.0, tile_size=16, tile_overlap=4)
        frame = _make_frame(32, 32)
        result = d._denoise_torch(frame)
        assert result.shape == frame.shape

    def test_denoise_torch_uniform_frame(self) -> None:
        """Lines 135: rng branch when vmax == vmin."""
        from astroai.processing.denoise.denoiser import Denoiser
        model = self._make_tiny_model()
        d = Denoiser(model=model, strength=1.0, tile_size=16, tile_overlap=2)
        frame = np.full((16, 16), 500.0, dtype=np.float64)
        result = d._denoise_torch(frame)
        assert result.shape == (16, 16)

    def test_denoise_full_call_with_model(self) -> None:
        """Full denoise() call dispatching through torch path."""
        from astroai.processing.denoise.denoiser import Denoiser
        model = self._make_tiny_model()
        d = Denoiser(model=model, strength=0.5, tile_size=16, tile_overlap=2)
        frame = _make_frame(32, 32)
        result = d.denoise(frame)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_denoise_rgb_with_model(self) -> None:
        """denoise() with model on RGB frame (3 channels)."""
        from astroai.processing.denoise.denoiser import Denoiser
        model = self._make_tiny_model()
        d = Denoiser(model=model, strength=1.0, tile_size=16, tile_overlap=2)
        frame = _make_rgb_frame(32, 32)
        result = d.denoise(frame)
        assert result.shape == (32, 32, 3)

    def test_denoise_torch_padding_branch(self) -> None:
        """Line 156: np.pad called when image smaller than tile_size."""
        from astroai.processing.denoise.denoiser import Denoiser
        model = self._make_tiny_model()
        # tile_size=32 but image is only 10x10 -> ph = 32-10 > 0 -> padding branch
        d = Denoiser(model=model, strength=1.0, tile_size=32, tile_overlap=0)
        frame = _make_frame(10, 10)
        result = d._denoise_torch(frame)
        assert result.shape == (10, 10)


# ---------------------------------------------------------------------------
# Denoiser with ONNX session (lines 129, 174-182)
# ---------------------------------------------------------------------------

class TestDenoiserOnnx:
    def _make_onnx_session(self) -> MagicMock:
        mock_input = MagicMock()
        mock_input.name = "input"
        session = MagicMock()
        session.get_inputs.return_value = [mock_input]

        def _run(output_names, feed_dict):
            inp = feed_dict["input"]
            out = np.zeros_like(inp, dtype=np.float32)
            return [out]

        session.run.side_effect = _run
        return session

    def test_denoise_single_dispatches_to_onnx(self) -> None:
        """Line 129: _denoise_single -> _denoise_onnx when onnx_session is set."""
        from astroai.processing.denoise.denoiser import Denoiser
        session = self._make_onnx_session()
        d = Denoiser(onnx_session=session)
        frame = _make_frame(32, 32)
        result = d._denoise_single(frame)
        assert result.shape == frame.shape

    def test_denoise_onnx_direct(self) -> None:
        """Lines 174-182: _denoise_onnx normalises, calls session.run, clips."""
        from astroai.processing.denoise.denoiser import Denoiser
        session = self._make_onnx_session()
        d = Denoiser(onnx_session=session)
        frame = _make_frame(32, 32)
        result = d._denoise_onnx(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.float64

    def test_denoise_onnx_uniform_frame(self) -> None:
        """_denoise_onnx: uniform frame (vmax == vmin) uses rng=1.0 branch."""
        from astroai.processing.denoise.denoiser import Denoiser
        session = self._make_onnx_session()
        d = Denoiser(onnx_session=session)
        frame = np.full((32, 32), 300.0, dtype=np.float64)
        result = d._denoise_onnx(frame)
        assert result.shape == (32, 32)

    def test_denoise_full_call_onnx(self) -> None:
        """Full denoise() call using ONNX path."""
        from astroai.processing.denoise.denoiser import Denoiser
        session = self._make_onnx_session()
        d = Denoiser(onnx_session=session, strength=0.8)
        frame = _make_frame(32, 32)
        result = d.denoise(frame)
        assert result.shape == frame.shape

    def test_denoise_onnx_via_sys_modules_patch(self) -> None:
        """ONNX path using patch.dict(sys.modules, ...) pattern."""
        mock_ort = MagicMock()
        mock_session = self._make_onnx_session()
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            from astroai.processing.denoise.denoiser import Denoiser
            d = Denoiser(onnx_session=mock_session, strength=1.0)
            frame = _make_frame(32, 32)
            result = d.denoise(frame)
        assert result.shape == (32, 32)


# ---------------------------------------------------------------------------
# NAFNetDenoiser (lines 235-239, 242-246, 249-261, 264-298)
# ---------------------------------------------------------------------------

class TestNAFNetDenoiser:
    def _make_nafnet_session(self) -> MagicMock:
        mock_input = MagicMock()
        mock_input.name = "input"
        session = MagicMock()
        session.get_inputs.return_value = [mock_input]

        def _run(output_names, feed_dict):
            inp = feed_dict["input"]
            out = np.zeros_like(inp, dtype=np.float32)
            return [out]

        session.run.side_effect = _run
        return session

    def test_init_stores_params(self) -> None:
        """Lines 235-239: __init__ stores strength, tile_size, etc."""
        from astroai.processing.denoise.denoiser import NAFNetDenoiser
        with patch("astroai.processing.denoise.denoiser.ModelDownloader"):
            n = NAFNetDenoiser(strength=0.7, tile_size=128, tile_overlap=16)
            assert n._strength == pytest.approx(0.7)
            assert n._tile_size == 128
            assert n._tile_overlap == 16
            assert n._session is None

    def test_get_session_caches(self) -> None:
        """Lines 242-246: _get_session() downloads once and caches result."""
        from astroai.processing.denoise.denoiser import NAFNetDenoiser
        mock_session = self._make_nafnet_session()

        with patch("astroai.processing.denoise.denoiser.ModelDownloader") as MockDL:
            instance = MockDL.return_value
            instance.load_onnx_session.return_value = mock_session
            n = NAFNetDenoiser()
            s1 = n._get_session()
            s2 = n._get_session()
            assert s1 is s2
            instance.load_onnx_session.assert_called_once_with(
                NAFNetDenoiser.MODEL_NAME, fallback_to_dummy=True
            )

    def test_denoise_grayscale(self) -> None:
        """Lines 249-261: denoise() for grayscale frame."""
        from astroai.processing.denoise.denoiser import NAFNetDenoiser
        mock_session = self._make_nafnet_session()

        with patch("astroai.processing.denoise.denoiser.ModelDownloader") as MockDL:
            MockDL.return_value.load_onnx_session.return_value = mock_session
            n = NAFNetDenoiser(strength=1.0, tile_size=16, tile_overlap=2)
            frame = _make_frame(16, 16).astype(np.float32)
            result = n.denoise(frame)
            assert result.shape == (16, 16)
            assert result.dtype == frame.dtype

    def test_denoise_rgb(self) -> None:
        """Lines 253-256: denoise() for RGB frame."""
        from astroai.processing.denoise.denoiser import NAFNetDenoiser
        mock_session = self._make_nafnet_session()

        with patch("astroai.processing.denoise.denoiser.ModelDownloader") as MockDL:
            MockDL.return_value.load_onnx_session.return_value = mock_session
            n = NAFNetDenoiser(strength=0.5, tile_size=16, tile_overlap=2)
            frame = _make_rgb_frame(16, 16).astype(np.float32)
            result = n.denoise(frame)
            assert result.shape == (16, 16, 3)

    def test_denoise_channel_tiling(self) -> None:
        """Lines 264-298: _denoise_channel tiling loop with padding branch."""
        from astroai.processing.denoise.denoiser import NAFNetDenoiser
        mock_session = self._make_nafnet_session()

        with patch("astroai.processing.denoise.denoiser.ModelDownloader") as MockDL:
            MockDL.return_value.load_onnx_session.return_value = mock_session
            # tile_size < image height forces multiple tiles + padding
            n = NAFNetDenoiser(strength=1.0, tile_size=8, tile_overlap=2)
            channel = _make_frame(20, 20)
            result = n._denoise_channel(channel)
            assert result.shape == (20, 20)

    def test_denoise_channel_uniform(self) -> None:
        """_denoise_channel: uniform input (vmax == vmin) uses rng=1.0."""
        from astroai.processing.denoise.denoiser import NAFNetDenoiser
        mock_session = self._make_nafnet_session()

        with patch("astroai.processing.denoise.denoiser.ModelDownloader") as MockDL:
            MockDL.return_value.load_onnx_session.return_value = mock_session
            n = NAFNetDenoiser(strength=1.0, tile_size=16, tile_overlap=2)
            channel = np.full((16, 16), 200.0, dtype=np.float64)
            result = n._denoise_channel(channel)
            assert result.shape == (16, 16)

    def test_denoise_preserves_float32_dtype(self) -> None:
        """blended.astype(original_dtype) preserves float32 input dtype."""
        from astroai.processing.denoise.denoiser import NAFNetDenoiser
        mock_session = self._make_nafnet_session()

        with patch("astroai.processing.denoise.denoiser.ModelDownloader") as MockDL:
            MockDL.return_value.load_onnx_session.return_value = mock_session
            n = NAFNetDenoiser(strength=1.0, tile_size=16, tile_overlap=2)
            frame = np.random.default_rng(1).uniform(0, 1, (16, 16)).astype(np.float32)
            result = n.denoise(frame)
            assert result.dtype == np.float32

    def test_strength_zero_returns_original(self) -> None:
        """strength=0 -> output equals input."""
        from astroai.processing.denoise.denoiser import NAFNetDenoiser
        mock_session = self._make_nafnet_session()

        with patch("astroai.processing.denoise.denoiser.ModelDownloader") as MockDL:
            MockDL.return_value.load_onnx_session.return_value = mock_session
            n = NAFNetDenoiser(strength=0.0, tile_size=16, tile_overlap=2)
            frame = _make_frame(16, 16).astype(np.float32)
            result = n.denoise(frame)
            np.testing.assert_allclose(result, frame, atol=1e-6)

    def test_denoise_channel_padding_branch(self) -> None:
        """Line 288: np.pad called when image smaller than tile_size."""
        from astroai.processing.denoise.denoiser import NAFNetDenoiser
        mock_session = self._make_nafnet_session()

        with patch("astroai.processing.denoise.denoiser.ModelDownloader") as MockDL:
            MockDL.return_value.load_onnx_session.return_value = mock_session
            # tile_size=32 but image is 10x10 -> ph = 32-10 > 0 -> padding branch
            n = NAFNetDenoiser(strength=1.0, tile_size=32, tile_overlap=0)
            channel = _make_frame(10, 10)
            result = n._denoise_channel(channel)
            assert result.shape == (10, 10)
