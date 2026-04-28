"""Mosaic pipeline step for multi-panel stitching."""

from __future__ import annotations

import logging
from pathlib import Path

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.engine.mosaic.engine import MosaicConfig, MosaicEngine

__all__ = ["MosaicStep"]

logger = logging.getLogger(__name__)

_PANEL_METADATA_KEYS = ("panel_paths", "mosaic_panels")


class MosaicStep(PipelineStep):
    """Pipeline step that stitches multiple panels into a mosaic."""

    def __init__(
        self,
        output_path: Path | None = None,
        blend_mode: str = "linear",
        gradient_correct: bool = True,
        output_scale: float = 1.0,
    ) -> None:
        self._output_path = output_path
        self._config = MosaicConfig(
            blend_mode=blend_mode,
            gradient_correct=gradient_correct,
            output_scale=output_scale,
        )

    @property
    def name(self) -> str:
        return "Mosaic"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.MOSAIC

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        panel_paths = self._resolve_panel_paths(context)
        if not panel_paths:
            logger.warning("MosaicStep: no panel_paths in context metadata, skipping")
            return context

        output_path = self._output_path or Path(
            context.metadata.get("mosaic_output", "mosaic_output.fits")
        )

        n = len(panel_paths)
        progress(PipelineProgress(
            stage=PipelineStage.MOSAIC,
            current=0,
            total=n + 1,
            message=f"Solving {n} mosaic panels",
        ))

        engine = MosaicEngine(
            config=self._config,
            plate_solver=context.metadata.get("plate_solver"),
        )

        ra_hint = context.metadata.get("ra_hint")
        dec_hint = context.metadata.get("dec_hint")

        result_path = engine.stitch(
            panel_paths=panel_paths,
            output_path=output_path,
            ra_hint=ra_hint,
            dec_hint=dec_hint,
        )

        progress(PipelineProgress(
            stage=PipelineStage.MOSAIC,
            current=n + 1,
            total=n + 1,
            message=f"Mosaic written to {result_path}",
        ))

        context.metadata["mosaic_output_path"] = result_path
        return context

    @staticmethod
    def _resolve_panel_paths(context: PipelineContext) -> list[Path]:
        for key in _PANEL_METADATA_KEYS:
            value = context.metadata.get(key)
            if value:
                return [Path(p) for p in value]
        return []
