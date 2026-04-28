"""Pipeline step for photometric color calibration."""
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
from astroai.processing.color.calibrator import (
    CatalogSource,
    SpectralColorCalibrator,
)

__all__ = ["ColorCalibrationStep"]

logger = logging.getLogger(__name__)


class ColorCalibrationStep(PipelineStep):
    """Apply spectrophotometric color calibration using stellar catalog data."""

    def __init__(
        self,
        catalog: CatalogSource = CatalogSource.GAIA_DR3,
        sample_radius_px: int = 8,
        max_iterations: int = 10,
        outlier_sigma: float = 2.5,
    ) -> None:
        self._calibrator = SpectralColorCalibrator(
            catalog=catalog,
            sample_radius_px=sample_radius_px,
            max_iterations=max_iterations,
            outlier_sigma=outlier_sigma,
        )

    @property
    def name(self) -> str:
        return "Farbkalibrierung"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        wcs = context.metadata.get("wcs")
        if wcs is None:
            logger.warning("ColorCalibrationStep: no WCS in context, skipping")
            return context

        data = context.result
        if data is None:
            logger.warning("ColorCalibrationStep: no result image in context, skipping")
            return context

        if data.ndim == 2:
            logger.info("ColorCalibrationStep: grayscale image, skipping color calibration")
            return context

        progress(PipelineProgress(
            stage=self.stage, current=0, total=3,
            message="Sternkatalog abfragen…",
        ))

        catalog_data = context.metadata.get("color_catalog_data")

        progress(PipelineProgress(
            stage=self.stage, current=1, total=3,
            message="Farbkalibrierung berechnen…",
        ))

        calibrated, result = self._calibrator.calibrate(
            data, wcs, catalog_data=catalog_data,
        )

        context.result = calibrated
        context.metadata["color_calibration_result"] = result

        progress(PipelineProgress(
            stage=self.stage, current=3, total=3,
            message=f"Farbkalibrierung abgeschlossen ({result.stars_used} Sterne)",
        ))

        return context
