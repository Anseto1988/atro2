"""AI-powered denoising for astrophotography frames."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from astroai.inference.backends import DeviceManager

__all__ = ["Denoiser", "SimpleUNet"]


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleUNet(nn.Module):
    """Lightweight U-Net for single-channel astro denoising."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.enc1 = _DoubleConv(in_channels, 32)
        self.enc2 = _DoubleConv(32, 64)
        self.enc3 = _DoubleConv(64, 128)
        self.bottleneck = _DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self._match_and_cat(self.up3(b), e3)], dim=0).unsqueeze(0).squeeze(0))
        d2 = self.dec2(torch.cat([self._match_and_cat(self.up2(d3), e2)], dim=0).unsqueeze(0).squeeze(0))
        d1 = self.dec1(torch.cat([self._match_and_cat(self.up1(d2), e1)], dim=0).unsqueeze(0).squeeze(0))
        return self.out_conv(d1)

    @staticmethod
    def _match_and_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diff_h = skip.size(2) - up.size(2)
        diff_w = skip.size(3) - up.size(3)
        up = F.pad(up, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([skip, up], dim=1)


class Denoiser:
    """AI denoiser for astrophotography frames.

    Supports three modes:
    - PyTorch U-Net model (GPU-accelerated)
    - ONNX model via ModelRegistry
    - Statistical fallback (wavelet-inspired bilateral filtering)
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        onnx_session: Any | None = None,
        strength: float = 1.0,
        tile_size: int = 256,
        tile_overlap: int = 32,
    ) -> None:
        self._dm = DeviceManager()
        self._model = model
        self._onnx = onnx_session
        self._strength = np.clip(strength, 0.0, 1.0)
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        if self._model is not None:
            self._model.eval()
            self._dm.to_device(self._model)

    def denoise(
        self, frame: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Denoise a single frame. Returns denoised frame same shape/dtype."""
        original_dtype = frame.dtype
        img = frame.astype(np.float64)
        was_rgb = img.ndim == 3

        if was_rgb:
            channels = [img[..., c] for c in range(img.shape[2])]
            denoised_channels = [self._denoise_single(ch) for ch in channels]
            result = np.stack(denoised_channels, axis=-1)
        else:
            result = self._denoise_single(img)

        blended = img + self._strength * (result - img)
        return blended.astype(original_dtype)

    def denoise_batch(
        self, frames: list[NDArray[np.floating[Any]]]
    ) -> list[NDArray[np.floating[Any]]]:
        return [self.denoise(f) for f in frames]

    def _denoise_single(self, channel: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        if self._model is not None:
            return self._denoise_torch(channel)
        if self._onnx is not None:
            return self._denoise_onnx(channel)
        return self._denoise_statistical(channel)

    def _denoise_torch(self, channel: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        h, w = channel.shape
        vmin, vmax = float(channel.min()), float(channel.max())
        rng = vmax - vmin if vmax > vmin else 1.0
        normalized = (channel - vmin) / rng

        result = np.zeros_like(normalized)
        weight_map = np.zeros_like(normalized)

        ts = self._tile_size
        overlap = self._tile_overlap
        step = ts - overlap

        for y in range(0, h, step):
            for x in range(0, w, step):
                y1 = min(y + ts, h)
                x1 = min(x + ts, w)
                y0 = max(y1 - ts, 0)
                x0 = max(x1 - ts, 0)

                tile = normalized[y0:y1, x0:x1]
                ph = ts - tile.shape[0]
                pw = ts - tile.shape[1]
                if ph > 0 or pw > 0:
                    tile = np.pad(tile, ((0, ph), (0, pw)), mode="reflect")

                tensor = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0)
                tensor = self._dm.to_device(tensor)

                with torch.no_grad():
                    out = self._model(tensor)  # type: ignore[misc]

                out_np = out.squeeze().cpu().numpy()
                out_crop = out_np[:y1 - y0, :x1 - x0]
                result[y0:y1, x0:x1] += out_crop
                weight_map[y0:y1, x0:x1] += 1.0

        weight_map = np.maximum(weight_map, 1.0)
        result /= weight_map
        return np.clip(result * rng + vmin, vmin, vmax)

    def _denoise_onnx(self, channel: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        vmin, vmax = float(channel.min()), float(channel.max())
        rng = vmax - vmin if vmax > vmin else 1.0
        normalized = ((channel - vmin) / rng).astype(np.float32)
        inp = normalized[np.newaxis, np.newaxis, :, :]
        input_name = self._onnx.get_inputs()[0].name
        out = self._onnx.run(None, {input_name: inp})[0]
        result = out.squeeze()
        return np.clip(result.astype(np.float64) * rng + vmin, vmin, vmax)

    @staticmethod
    def _denoise_statistical(
        channel: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Multi-scale non-local means inspired statistical denoiser."""
        from scipy.ndimage import gaussian_filter, median_filter

        result = channel.copy()
        noise_est = Denoiser._estimate_noise(channel)

        if noise_est < 1e-10:
            return result

        med = median_filter(result, size=3)
        diff = np.abs(result - med)
        impulse_mask = diff > 5.0 * noise_est
        result[impulse_mask] = med[impulse_mask]

        scales = [1.0, 2.0, 4.0]
        weights = [0.5, 0.3, 0.2]
        smoothed = np.zeros_like(result)
        for sigma, w in zip(scales, weights):
            smoothed += w * gaussian_filter(result, sigma=sigma)

        adaptive_weight = np.clip(diff / (3.0 * noise_est + 1e-10), 0.0, 1.0)
        result = result * adaptive_weight + smoothed * (1.0 - adaptive_weight)
        return result

    @staticmethod
    def _estimate_noise(channel: NDArray[np.floating[Any]]) -> float:
        """Robust noise estimation via MAD of Laplacian."""
        from scipy.ndimage import convolve
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        filtered = convolve(channel, laplacian)
        mad = float(np.median(np.abs(filtered - np.median(filtered))))
        return mad * 1.4826 / np.sqrt(20.0)
