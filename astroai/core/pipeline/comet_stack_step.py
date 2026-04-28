"""PipelineStep für Dual-Tracking Comet Stacking."""

from __future__ import annotations

import logging
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
from astroai.engine.comet.tracker import CometTracker
from astroai.engine.comet.stacker import CometStacker, TrackingMode

__all__ = ["CometStackStep"]

logger = logging.getLogger(__name__)

_COMET_STAR_STACK_KEY = "comet_star_stack"
_COMET_NUCLEUS_STACK_KEY = "comet_nucleus_stack"
_COMET_POSITIONS_KEY = "comet_positions"


class CometStackStep(PipelineStep):
    """Stackt Kometenaufnahmen mit Dual-Tracking-Alignment.

    Führt gleichzeitig zwei Stacks durch:
      - Stern-Stack: klassisches Alignment (Frames bereits registriert).
      - Kometen-Stack: Alignment auf Kometenkopf.
    Optional: Blend beider Stacks als Hauptresultat.

    Speichert in context.metadata:
      - 'comet_star_stack': ndarray, stern-aligned Stack.
      - 'comet_nucleus_stack': ndarray, kometen-aligned Stack.
      - 'comet_positions': Liste der CometPosition-Objekte.

    Args:
        tracking_mode: 'stars' | 'comet' | 'blend'.
        blend_factor: 0.0 = nur Sterne, 1.0 = nur Komet.
        stack_method: 'mean' | 'median' | 'sigma_clip'.
        min_blob_area: Mindest-Blob-Größe für CometTracker.
        top_fraction: Anteil Top-Pixel für Blob-Erkennung.
        fail_silently: Fehler loggen statt Pipeline-Abbruch.
    """

    def __init__(
        self,
        tracking_mode: TrackingMode = "blend",
        blend_factor: float = 0.5,
        stack_method: str = "sigma_clip",
        min_blob_area: int = 5,
        top_fraction: float = 0.002,
        fail_silently: bool = True,
    ) -> None:
        self._tracking_mode = tracking_mode
        self._blend_factor = blend_factor
        self._tracker = CometTracker(
            min_blob_area=min_blob_area,
            top_fraction=top_fraction,
        )
        self._stacker = CometStacker(stack_method=stack_method)
        self._fail_silently = fail_silently

    @property
    def name(self) -> str:
        return f"Comet Stack ({self._tracking_mode})"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.COMET_STACKING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        if not context.images:
            logger.warning("CometStackStep: keine Frames im Context, übersprungen")
            return context

        n = len(context.images)
        progress(PipelineProgress(
            stage=self.stage,
            current=0,
            total=3,
            message=f"Comet tracking: {n} Frames analysieren…",
        ))

        try:
            positions = self._tracker.track(context.images)
            context.metadata[_COMET_POSITIONS_KEY] = positions

            progress(PipelineProgress(
                stage=self.stage,
                current=1,
                total=3,
                message="Dual-Tracking Stack wird erstellt…",
            ))

            result = self._stacker.stack(
                context.images,
                positions,
                tracking_mode=self._tracking_mode,
                blend_factor=self._blend_factor,
            )

            context.metadata[_COMET_STAR_STACK_KEY] = result.star_stack
            context.metadata[_COMET_NUCLEUS_STACK_KEY] = result.comet_stack

            context.result = self._select_primary_result(result)

            progress(PipelineProgress(
                stage=self.stage,
                current=3,
                total=3,
                message="Comet Stack abgeschlossen",
            ))

        except Exception as exc:
            if self._fail_silently:
                logger.error("CometStackStep fehlgeschlagen (übersprungen): %s", exc)
            else:
                raise

        return context

    def _select_primary_result(self, result: Any) -> NDArray[np.floating[Any]]:
        arr: NDArray[np.floating[Any]]
        if self._tracking_mode == "stars":
            arr = result.star_stack
        elif self._tracking_mode == "comet":
            arr = result.comet_stack
        else:
            arr = result.blend if result.blend is not None else result.comet_stack
        return arr
