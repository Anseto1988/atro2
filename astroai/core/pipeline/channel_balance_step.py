"""ChannelBalanceStep — pipeline step for per-channel background balance."""
from __future__ import annotations

import logging

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.processing.color.channel_balance import ChannelBalanceConfig, ChannelBalancer

__all__ = ["ChannelBalanceStep"]

logger = logging.getLogger(__name__)


class ChannelBalanceStep(PipelineStep):
    """Apply per-channel R/G/B addition-offset background balance."""

    def __init__(self, config: ChannelBalanceConfig | None = None) -> None:
        self._balancer = ChannelBalancer(config)

    @property
    def name(self) -> str:
        return "Kanal-Balance"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        image = context.result if context.result is not None else (
            context.images[0] if context.images else None
        )
        if image is None:
            logger.warning("ChannelBalanceStep: no image in context, skipping")
            return context

        progress(PipelineProgress(
            stage=self.stage, current=0, total=1,
            message="Kanal-Balance anpassen…",
        ))
        context.result = self._balancer.apply(image)
        progress(PipelineProgress(
            stage=self.stage, current=1, total=1,
            message="Kanal-Balance abgeschlossen",
        ))
        return context
