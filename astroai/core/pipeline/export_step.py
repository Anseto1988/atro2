from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroai.core.io.fits_io import ImageMetadata, write_fits
from astroai.core.io.tiff_io import write_tiff32
from astroai.core.io.xisf_io import write_xisf
from astroai.core.pipeline.base import (
    PipelineContext,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    _noop_callback,
)


class ExportFormat(Enum):
    XISF = "xisf"
    TIFF32 = "tiff"
    FITS = "fits"


_EXTENSION_MAP: dict[ExportFormat, str] = {
    ExportFormat.XISF: ".xisf",
    ExportFormat.TIFF32: ".tif",
    ExportFormat.FITS: ".fits",
}


class ExportStep(PipelineStep):
    def __init__(
        self,
        output_dir: str | Path,
        fmt: ExportFormat = ExportFormat.XISF,
        filename: str = "output",
        metadata: ImageMetadata | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._format = fmt
        self._filename = filename
        self._metadata = metadata

    @property
    def name(self) -> str:
        return f"export_{self._format.value}"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.SAVING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = _noop_callback,
    ) -> PipelineContext:
        data = context.result if context.result is not None else (
            context.images[0] if context.images else None
        )
        if data is None:
            raise ValueError("No image data to export")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        ext = _EXTENSION_MAP[self._format]
        output_path = self._output_dir / f"{self._filename}{ext}"

        meta = self._metadata
        if meta is None and "metadata" in context.metadata:
            meta = context.metadata["metadata"]

        if self._format == ExportFormat.XISF:
            write_xisf(output_path, data, meta)
        elif self._format == ExportFormat.TIFF32:
            write_tiff32(output_path, data, meta)
        elif self._format == ExportFormat.FITS:
            write_fits(output_path, data, meta)

        context.metadata["export_path"] = str(output_path)
        return context
