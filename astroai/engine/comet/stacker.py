"""Dual-Tracking Comet Stacker: Kometenkopf- und Stern-Alignment gleichzeitig."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.ndimage import shift as ndimage_shift

from astroai.engine.comet.tracker import CometPosition
from astroai.engine.stacking.stacker import FrameStacker

__all__ = ["CometStacker", "CometStackResult", "TrackingMode"]

logger = logging.getLogger(__name__)

TrackingMode = Literal["stars", "comet", "blend"]


@dataclass
class CometStackResult:
    """Ergebnis des Dual-Tracking-Stacks."""
    star_stack: np.ndarray
    comet_stack: np.ndarray
    blend: np.ndarray | None = None
    blend_factor: float = 0.5


class CometStacker:
    """Erzeugt parallele Stacks: einmal stern-aligned, einmal kometen-aligned.

    Args:
        stack_method: Stacking-Methode ('mean', 'median', 'sigma_clip').
        sigma_low: Sigma-Clipping untere Schwelle.
        sigma_high: Sigma-Clipping obere Schwelle.
    """

    def __init__(
        self,
        stack_method: str = "sigma_clip",
        sigma_low: float = 2.5,
        sigma_high: float = 2.5,
    ) -> None:
        self._stacker = FrameStacker()
        self._stack_method = stack_method
        self._sigma_low = sigma_low
        self._sigma_high = sigma_high

    def stack(
        self,
        frames: Sequence[np.ndarray],
        comet_positions: Sequence[CometPosition],
        tracking_mode: TrackingMode = "blend",
        blend_factor: float = 0.5,
    ) -> CometStackResult:
        """Führt den Dual-Tracking-Stack durch.

        Args:
            frames: Stern-registrierte Frames (bereits auf Sterne ausgerichtet).
            comet_positions: CometPosition je Frame (vom CometTracker).
            tracking_mode: 'stars', 'comet', oder 'blend'.
            blend_factor: 0.0 = nur Sterne, 1.0 = nur Komet (nur bei 'blend').

        Returns:
            CometStackResult mit star_stack, comet_stack und optionalem blend.
        """
        if len(frames) != len(comet_positions):
            raise ValueError(
                f"frames ({len(frames)}) und comet_positions ({len(comet_positions)}) "
                "müssen gleich lang sein"
            )
        if not frames:
            raise ValueError("Keine Frames übergeben")

        frames_list = list(frames)
        star_stack = self._build_star_stack(frames_list)
        comet_stack = self._build_comet_stack(frames_list, comet_positions)

        blend = None
        if tracking_mode == "blend":
            blend = self._blend(star_stack, comet_stack, blend_factor)

        return CometStackResult(
            star_stack=star_stack,
            comet_stack=comet_stack,
            blend=blend,
            blend_factor=blend_factor,
        )

    def _build_star_stack(self, frames: list[np.ndarray]) -> np.ndarray:
        kwargs: dict[str, float] = {}
        if self._stack_method == "sigma_clip":
            kwargs = {"sigma_low": self._sigma_low, "sigma_high": self._sigma_high}
        result = self._stacker.stack(frames, method=self._stack_method, **kwargs)
        logger.info("Star stack complete: shape=%s", result.shape)
        return result

    def _build_comet_stack(
        self,
        frames: list[np.ndarray],
        comet_positions: Sequence[CometPosition],
    ) -> np.ndarray:
        ys = [p.y for p in comet_positions]
        xs = [p.x for p in comet_positions]
        ref_y = float(np.median(ys))
        ref_x = float(np.median(xs))

        shifted: list[np.ndarray] = []
        for frame, pos in zip(frames, comet_positions):
            dy = ref_y - pos.y
            dx = ref_x - pos.x
            if abs(dy) < 0.01 and abs(dx) < 0.01:
                shifted.append(frame)
            else:
                shift_vec = (-dy, -dx, 0.0) if frame.ndim == 3 else (-dy, -dx)
                shifted.append(
                    ndimage_shift(frame, shift_vec, order=3, mode="constant", cval=0.0)
                )

        kwargs: dict[str, float] = {}
        if self._stack_method == "sigma_clip":
            kwargs = {"sigma_low": self._sigma_low, "sigma_high": self._sigma_high}
        result = self._stacker.stack(shifted, method=self._stack_method, **kwargs)
        logger.info(
            "Comet stack complete: shape=%s, ref=(%.2f, %.2f)", result.shape, ref_y, ref_x
        )
        return result

    @staticmethod
    def _blend(star: np.ndarray, comet: np.ndarray, factor: float) -> np.ndarray:
        factor = float(np.clip(factor, 0.0, 1.0))
        return (1.0 - factor) * star + factor * comet
