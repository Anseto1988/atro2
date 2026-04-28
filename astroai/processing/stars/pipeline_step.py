"""Pipeline step for AI-assisted star removal (FR-3.2)."""

from __future__ import annotations

import logging
from typing import Any, cast

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
from astroai.processing.stars.star_manager import (
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    OnTileProgress,
    StarManager,
)

__all__ = ["StarRemovalStep"]

logger = logging.getLogger(__name__)

_STARNET_MODEL_NAME = "starnet_plus_plus"


class StarRemovalStep(PipelineStep):
    """Remove stars using ONNX StarNet++ model with inpainting fallback."""

    def __init__(
        self,
        detection_sigma: float = 4.0,
        min_star_area: int = 3,
        max_star_area: int = 5000,
        mask_dilation: int = 3,
        reduce_enabled: bool = False,
        reduce_factor: float = 0.5,
        onnx_model_path: str | None = None,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
    ) -> None:
        self._manager = StarManager(
            detection_sigma=detection_sigma,
            min_star_area=min_star_area,
            max_star_area=max_star_area,
            mask_dilation=mask_dilation,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        self._reduce_enabled = reduce_enabled
        self._reduce_factor = float(np.clip(reduce_factor, 0.0, 1.0))
        self._onnx_model_path = onnx_model_path
        self._onnx_session: Any | None = None
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap

    @property
    def name(self) -> str:
        return "Star Removal"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        data = context.result if context.result is not None else (
            context.images[0] if context.images else None
        )
        if data is None:
            return context

        if self._reduce_enabled:
            progress(PipelineProgress(
                stage=self.stage, current=0, total=1, message="Reducing stars\u2026",
            ))
            reduced = self._manager.reduce_stars(data, factor=self._reduce_factor)
            context.result = reduced
            progress(PipelineProgress(
                stage=self.stage, current=1, total=1, message="Star reduction complete",
            ))
            return context

        progress(PipelineProgress(
            stage=self.stage, current=0, total=1, message="Removing stars\u2026",
        ))

        session = self._try_load_onnx()
        if session is not None:
            def _tile_progress(tile_idx: int, total_tiles: int) -> None:
                progress(PipelineProgress(
                    stage=self.stage,
                    current=tile_idx,
                    total=total_tiles,
                    message=f"Removing stars (tile {tile_idx}/{total_tiles})\u2026",
                ))

            starless, star_mask = self._remove_onnx(
                data, session, on_tile_progress=_tile_progress,
            )
            logger.info("Star removal completed via ONNX model")
        else:
            starless, star_mask = self._remove_fallback(data)
            logger.info("Star removal completed via inpainting fallback")

        context.starless_image = starless
        context.star_mask = star_mask
        context.result = starless

        progress(PipelineProgress(
            stage=self.stage, current=1, total=1, message="Star removal complete",
        ))
        return context

    def _try_load_onnx(self) -> Any | None:
        if self._onnx_session is not None:
            return self._onnx_session

        if self._onnx_model_path is not None:
            return self._load_from_path(self._onnx_model_path)

        return self._load_from_registry()

    def _load_from_path(self, path: str) -> Any | None:
        from astroai.core.onnx_registry import OnnxModelRegistry

        session = OnnxModelRegistry().load_from_path(path)
        if session is not None:
            self._onnx_session = session
        return session

    def _load_from_registry(self) -> Any | None:
        try:
            from astroai.core.onnx_registry import OnnxModelRegistry

            registry = OnnxModelRegistry()
            if not registry.is_available(_STARNET_MODEL_NAME):
                logger.debug("Starnet++ model not available in registry")
                return None
            self._onnx_session = registry.get_session(
                _STARNET_MODEL_NAME, fallback_to_dummy=False,
            )
            return self._onnx_session
        except Exception as exc:
            logger.debug("Registry ONNX load failed (%s), using fallback", exc)
            return None

    def _remove_onnx(
        self,
        frame: NDArray[np.floating[Any]],
        session: Any,
        on_tile_progress: OnTileProgress | None = None,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        h, w = frame.shape[:2]
        is_rgb = frame.ndim == 3 and frame.shape[2] == 3

        if StarManager.needs_tiling(h, w):
            starless = StarManager.process_tiled(
                frame,
                lambda tile: self._onnx_infer_tile(tile, session, is_rgb),
                tile_size=self._tile_size,
                overlap=self._tile_overlap,
                on_progress=on_tile_progress,
            )
        else:
            starless = self._onnx_infer_tile(frame, session, is_rgb)

        starless = np.clip(starless.astype(frame.dtype), 0.0, None)
        star_mask = self._derive_star_mask(frame, starless)
        return starless, star_mask

    @staticmethod
    def _onnx_infer_tile(
        tile: NDArray[np.floating[Any]], session: Any, is_rgb: bool,
    ) -> NDArray[np.floating[Any]]:
        if is_rgb:
            inp = tile.transpose(2, 0, 1).astype(np.float32)[np.newaxis]
        else:
            inp = tile.astype(np.float32)[np.newaxis, np.newaxis]

        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: inp})[0].squeeze()

        if is_rgb and out.ndim == 3 and out.shape[0] == 3:
            out = out.transpose(1, 2, 0)

        return cast(NDArray[np.floating[Any]], np.clip(out.astype(tile.dtype), 0.0, None))

    def _remove_fallback(
        self, frame: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        starless, stars_only = self._manager.separate(frame)
        star_mask = self._manager.create_star_mask(frame).astype(np.float32)
        return starless, star_mask

    @staticmethod
    def _derive_star_mask(
        original: NDArray[np.floating[Any]],
        starless: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        diff = np.abs(original.astype(np.float64) - starless.astype(np.float64))
        if diff.ndim == 3:
            diff = diff.mean(axis=-1)
        threshold = diff.mean() + 2.0 * diff.std()
        return cast(NDArray[np.floating[Any]], (diff > threshold).astype(np.float32))
