"""Pipeline step for AI-assisted star removal (FR-3.2)."""

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
    _noop_callback,
)
from astroai.processing.stars.star_manager import StarManager

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
        onnx_model_path: str | None = None,
    ) -> None:
        self._manager = StarManager(
            detection_sigma=detection_sigma,
            min_star_area=min_star_area,
            max_star_area=max_star_area,
            mask_dilation=mask_dilation,
        )
        self._onnx_model_path = onnx_model_path
        self._onnx_session: Any | None = None

    @property
    def name(self) -> str:
        return "Star Removal"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = _noop_callback,
    ) -> PipelineContext:
        data = context.result if context.result is not None else (
            context.images[0] if context.images else None
        )
        if data is None:
            return context

        progress(PipelineProgress(
            stage=self.stage, current=0, total=1, message="Removing stars\u2026",
        ))

        session = self._try_load_onnx()
        if session is not None:
            starless, star_mask = self._remove_onnx(data, session)
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
        try:
            import onnxruntime as ort
            from pathlib import Path

            model_path = Path(path)
            if not model_path.exists():
                logger.debug("ONNX model path does not exist: %s", path)
                return None
            self._onnx_session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            return self._onnx_session
        except Exception as exc:
            logger.debug("Failed to load ONNX from path (%s): %s", path, exc)
            return None

    def _load_from_registry(self) -> Any | None:
        try:
            from astroai.inference.models.downloader import ModelDownloader

            downloader = ModelDownloader()
            if not downloader.is_available(_STARNET_MODEL_NAME):
                logger.debug("Starnet++ model not available in registry")
                return None
            self._onnx_session = downloader.load_onnx_session(
                _STARNET_MODEL_NAME, fallback_to_dummy=False
            )
            return self._onnx_session
        except Exception as exc:
            logger.debug("Registry ONNX load failed (%s), using fallback", exc)
            return None

    def _remove_onnx(
        self, frame: NDArray[np.floating[Any]], session: Any,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        is_rgb = frame.ndim == 3 and frame.shape[2] == 3

        if is_rgb:
            inp = frame.transpose(2, 0, 1).astype(np.float32)[np.newaxis]
        else:
            inp = frame.astype(np.float32)[np.newaxis, np.newaxis]

        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: inp})[0].squeeze()

        if is_rgb and out.ndim == 3 and out.shape[0] == 3:
            out = out.transpose(1, 2, 0)

        starless = np.clip(out.astype(frame.dtype), 0.0, None)
        star_mask = self._derive_star_mask(frame, starless)
        return starless, star_mask

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
        return (diff > threshold).astype(np.float32)
