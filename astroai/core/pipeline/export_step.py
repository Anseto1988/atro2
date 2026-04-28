from __future__ import annotations

import logging
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
    noop_callback,
)

logger = logging.getLogger(__name__)


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
        export_starless: bool = False,
        export_star_mask: bool = False,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._format = fmt
        self._filename = filename
        self._metadata = metadata
        self._export_starless = export_starless
        self._export_star_mask = export_star_mask

    @property
    def name(self) -> str:
        return f"export_{self._format.value}"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.SAVING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        data = context.result if context.result is not None else (
            context.images[0] if context.images else None
        )
        if data is None:
            raise ValueError("No image data to export")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        ext = _EXTENSION_MAP[self._format]

        meta = self._metadata
        if meta is None and "metadata" in context.metadata:
            meta = context.metadata["metadata"]

        output_path = self._output_dir / f"{self._filename}{ext}"
        self._write(output_path, data, meta)
        context.metadata["export_path"] = str(output_path)

        if self._export_starless and context.starless_image is not None:
            starless_path = self._output_dir / f"{self._filename}_starless{ext}"
            self._write(starless_path, context.starless_image, meta)
            context.metadata["export_starless_path"] = str(starless_path)

        if self._export_star_mask and context.star_mask is not None:
            mask_path = self._output_dir / f"{self._filename}_starmask{ext}"
            self._write(mask_path, context.star_mask, meta)
            context.metadata["export_starmask_path"] = str(mask_path)

        return context

    def _write(
        self,
        path: Path,
        data: NDArray[np.floating[Any]],
        meta: ImageMetadata | None,
    ) -> None:
        if self._format == ExportFormat.XISF:
            write_xisf(path, data, meta)
        elif self._format == ExportFormat.TIFF32:
            write_tiff32(path, data, meta)
        elif self._format == ExportFormat.FITS:
            write_fits(path, data, meta)
