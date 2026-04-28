"""Pipeline step for LRGB / Narrowband channel combination."""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.processing.channels.combiner import ChannelCombiner
from astroai.processing.channels.narrowband_mapper import NarrowbandMapper, NarrowbandPalette

__all__ = ["ChannelCombineStep", "CombineMode"]


class CombineMode(Enum):
    LRGB = "lrgb"
    NARROWBAND = "narrowband"


class ChannelCombineStep(PipelineStep):
    """Combines multi-channel frames into a single colour image."""

    def __init__(
        self,
        mode: CombineMode = CombineMode.LRGB,
        palette: NarrowbandPalette = NarrowbandPalette.SHO,
        channels: dict[str, NDArray[np.floating[Any]]] | None = None,
    ) -> None:
        self._mode = mode
        self._palette = palette
        self._channels: dict[str, NDArray[np.floating[Any]]] = channels or {}
        self._combiner = ChannelCombiner()
        self._mapper = NarrowbandMapper()

    @property
    def name(self) -> str:
        return "Channel Combine"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def set_channels(self, channels: dict[str, NDArray[np.floating[Any]]]) -> None:
        self._channels = channels

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        progress(PipelineProgress(stage=self.stage, current=0, total=1, message="Kanal-Kombination läuft…"))
        result = self._combine()
        if result is not None:
            context.result = result
        progress(PipelineProgress(stage=self.stage, current=1, total=1, message="Kanal-Kombination abgeschlossen"))
        return context

    def _combine(self) -> NDArray[np.float32] | None:
        ch = self._channels
        if not ch or not any(v is not None for v in ch.values()):
            return None
        if self._mode is CombineMode.LRGB:
            return self._combiner.combine_lrgb(
                L=ch.get("L"),
                R=ch.get("R"),
                G=ch.get("G"),
                B=ch.get("B"),
            )
        return self._mapper.map(
            Ha=ch.get("Ha"),
            OIII=ch.get("OIII"),
            SII=ch.get("SII"),
            palette=self._palette,
        )
