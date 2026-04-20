"""Pipeline step for deconvolution / AI-based sharpness reconstruction (FR-3.x)."""
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
    _noop_callback,
)
from astroai.processing.deconvolution.deconvolver import Deconvolver

__all__ = ["DeconvolutionStep"]

logger = logging.getLogger(__name__)

_DECONV_MODEL_NAME = "blind_deconv"


class DeconvolutionStep(PipelineStep):
    """Sharpen images via Lucy-Richardson deconvolution; falls back from ONNX when unavailable."""

    def __init__(
        self,
        iterations: int = 10,
        psf_size: int = 5,
        psf_sigma: float = 1.0,
        clip_output: bool = True,
        onnx_model_path: str | None = None,
    ) -> None:
        self._deconvolver = Deconvolver(
            iterations=iterations,
            psf_size=psf_size,
            psf_sigma=psf_sigma,
            clip_output=clip_output,
        )
        self._onnx_model_path = onnx_model_path
        self._onnx_session: Any | None = None

    @property
    def name(self) -> str:
        return "Deconvolution"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = _noop_callback,
    ) -> PipelineContext:
        if context.result is not None:
            progress(PipelineProgress(
                stage=self.stage, current=0, total=1, message="Deconvolution läuft\u2026",
            ))
            context.result = self._process(context.result)
            progress(PipelineProgress(
                stage=self.stage, current=1, total=1, message="Deconvolution abgeschlossen",
            ))
        elif context.images:
            total = len(context.images)
            for i in range(total):
                progress(PipelineProgress(
                    stage=self.stage,
                    current=i,
                    total=total,
                    message=f"Deconvolution {i + 1}/{total}",
                ))
                context.images[i] = self._process(context.images[i])
        return context

    def _process(self, frame: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        session = self._try_load_onnx()
        if session is not None:
            logger.info("Deconvolution via ONNX model")
            return self._deconvolve_onnx(frame, session)
        logger.info("Deconvolution via Lucy-Richardson fallback")
        return self._deconvolver.deconvolve(frame)

    # -- ONNX loading --------------------------------------------------------

    def _try_load_onnx(self) -> Any | None:
        if self._onnx_session is not None:
            return self._onnx_session
        if self._onnx_model_path is not None:
            return self._load_from_path(self._onnx_model_path)
        return self._load_from_registry()

    def _load_from_path(self, path: str) -> Any | None:
        try:
            import onnxruntime as ort
            from pathlib import Path as _Path

            model_path = _Path(path)
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
            if not downloader.is_available(_DECONV_MODEL_NAME):
                logger.debug("Blind deconv model not available in registry")
                return None
            self._onnx_session = downloader.load_onnx_session(
                _DECONV_MODEL_NAME, fallback_to_dummy=False
            )
            return self._onnx_session
        except Exception as exc:
            logger.debug("Registry ONNX load failed (%s), using fallback", exc)
            return None

    # -- ONNX inference -------------------------------------------------------

    def _deconvolve_onnx(
        self,
        frame: NDArray[np.floating[Any]],
        session: Any,
    ) -> NDArray[np.floating[Any]]:
        is_rgb = frame.ndim == 3 and frame.shape[2] == 3
        if is_rgb:
            inp = frame.transpose(2, 0, 1).astype(np.float32)[np.newaxis]
        else:
            inp = frame.astype(np.float32)[np.newaxis, np.newaxis]

        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: inp})[0].squeeze()

        if is_rgb and out.ndim == 3 and out.shape[0] == 3:
            out = out.transpose(1, 2, 0)

        return cast(NDArray[np.floating[Any]], np.clip(out.astype(frame.dtype), 0.0, None))
