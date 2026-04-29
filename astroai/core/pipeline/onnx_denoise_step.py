"""OnnxDenoiseStep — NAFNet ONNX denoising with statistical fallback."""
from __future__ import annotations

import logging

from astroai.core.onnx_registry import OnnxModelRegistry
from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.inference.backends.nafnet import NAFNetDenoiser
from astroai.processing.denoise.pipeline_step import DenoiseStep

__all__ = ["OnnxDenoiseStep"]

logger = logging.getLogger(__name__)


class OnnxDenoiseStep(PipelineStep):
    """NAFNet ONNX denoising; falls back to DenoiseStep when model unavailable."""

    def __init__(
        self,
        strength: float = 1.0,
        tile_size: int = 512,
        tile_overlap: int = 64,
        registry: OnnxModelRegistry | None = None,
    ) -> None:
        self._strength = strength
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._registry = registry if registry is not None else OnnxModelRegistry()
        self._nafnet: NAFNetDenoiser | None = None

    @property
    def name(self) -> str:
        return "ONNX Denoise (NAFNet)"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    @property
    def active_backend(self) -> str:
        """Returns 'nafnet' when model is cached/available, else 'basic'."""
        return "nafnet" if self._registry.is_available(NAFNetDenoiser.MODEL_NAME) else "basic"

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        if self._registry.is_available(NAFNetDenoiser.MODEL_NAME):
            return self._run_nafnet(context, progress)
        logger.info("NAFNet model not available — falling back to DenoiseStep")
        return self._run_fallback(context, progress)

    def _run_nafnet(
        self,
        context: PipelineContext,
        progress: ProgressCallback,
    ) -> PipelineContext:
        if self._nafnet is None:
            self._nafnet = NAFNetDenoiser(
                strength=self._strength,
                tile_size=self._tile_size,
                tile_overlap=self._tile_overlap,
                registry=self._registry,
            )

        if context.result is not None:
            progress(PipelineProgress(
                stage=self.stage, current=0, total=1,
                message="NAFNet KI-Entrauschung läuft…",
            ))
            context.result = self._nafnet.denoise(context.result)
            progress(PipelineProgress(
                stage=self.stage, current=1, total=1,
                message="NAFNet Entrauschung abgeschlossen",
            ))
        elif context.images:
            total = len(context.images)
            for i, img in enumerate(context.images):
                progress(PipelineProgress(
                    stage=self.stage, current=i, total=total,
                    message=f"NAFNet Denoise {i + 1}/{total}",
                ))
                context.images[i] = self._nafnet.denoise(img)
        return context

    def _run_fallback(
        self,
        context: PipelineContext,
        progress: ProgressCallback,
    ) -> PipelineContext:
        fallback = DenoiseStep(
            strength=self._strength,
            tile_size=self._tile_size,
            tile_overlap=self._tile_overlap,
        )
        return fallback.execute(context, progress)
