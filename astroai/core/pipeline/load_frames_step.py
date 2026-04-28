"""Pipeline step that loads image frames from file paths into the context."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)

__all__ = ["LoadFramesStep"]

logger = logging.getLogger(__name__)


def _load_frame(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".fits", ".fit", ".fts"):
        from astropy.io import fits as _fits
        with _fits.open(str(path)) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"No image data in {path.name}")
            return np.asarray(data, dtype=np.float32)
    from astroai.core.io.raw_io import RAW_EXTENSIONS
    if suffix in RAW_EXTENSIONS:
        from astroai.core.io.raw_io import read_raw
        rgb, _meta = read_raw(path)
        luminance: np.ndarray = np.mean(rgb, axis=2).astype(np.float32)
        return luminance
    from PIL import Image
    img: np.ndarray = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return img


class LoadFramesStep(PipelineStep):
    """Load image files from disk and populate context.images."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self._paths = [Path(p) for p in paths]

    @property
    def name(self) -> str:
        return "Frame-Laden"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.CALIBRATION

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        paths = self._paths
        n = len(paths)
        if n == 0:
            return context

        progress(PipelineProgress(
            stage=self.stage, current=0, total=n,
            message=f"Lade {n} Frames…",
        ))

        frames: list[np.ndarray] = []
        for i, path in enumerate(paths):
            frame = _load_frame(path)
            frames.append(frame)
            progress(PipelineProgress(
                stage=self.stage, current=i + 1, total=n,
                message=f"Frame {i + 1}/{n} geladen: {path.name}",
            ))

        context.images = frames
        context.metadata["loaded_frame_paths"] = [str(p) for p in paths]
        logger.info("LoadFramesStep: %d frames loaded", n)
        return context
