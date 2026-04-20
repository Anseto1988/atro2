"""Frame stacking engine for astrophotography."""

from typing import Any, Callable, cast

import numpy as np

__all__ = ["FrameStacker"]


class FrameStacker:
    """Combines multiple aligned frames into a single stacked output image."""

    def _validate(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("No frames provided")
        shape = frames[0].shape
        for i, f in enumerate(frames):
            if f.shape != shape:
                raise ValueError(
                    f"Frame {i} shape {f.shape} does not match"
                    f" first frame shape {shape}"
                )
        return np.stack(frames, axis=0)

    def stack_mean(self, frames: list[np.ndarray]) -> np.ndarray:
        cube = self._validate(frames)
        return cast(np.ndarray, np.mean(cube, axis=0))

    def stack_median(self, frames: list[np.ndarray]) -> np.ndarray:
        cube = self._validate(frames)
        return cast(np.ndarray, np.median(cube, axis=0))

    def stack_sigma_clip(
        self,
        frames: list[np.ndarray],
        sigma_low: float = 2.5,
        sigma_high: float = 2.5,
    ) -> np.ndarray:
        cube = self._validate(frames).astype(np.float64)
        masked = np.ma.array(cube, mask=False)
        for _ in range(5):
            mean = masked.mean(axis=0, keepdims=True)
            std = masked.std(axis=0, keepdims=True)
            diff = cube - mean
            reject = (diff < -sigma_low * std) | (diff > sigma_high * std)
            new_mask = masked.mask | reject
            if np.array_equal(new_mask, masked.mask):
                break
            masked = np.ma.array(cube, mask=new_mask)
        return np.asarray(masked.mean(axis=0))

    def stack(
        self,
        frames: list[np.ndarray],
        method: str = "sigma_clip",
        **kwargs: Any,
    ) -> np.ndarray:
        methods: dict[str, Callable[..., np.ndarray]] = {
            "mean": self.stack_mean,
            "median": self.stack_median,
            "sigma_clip": self.stack_sigma_clip,
        }
        if method not in methods:
            raise ValueError(
                f"Unknown method '{method}'."
                f" Choose from {list(methods.keys())}"
            )
        return methods[method](frames, **kwargs)
