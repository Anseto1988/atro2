from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


class PipelineStage(Enum):
    LOADING = auto()
    CALIBRATION = auto()
    REGISTRATION = auto()
    STACKING = auto()
    COMET_STACKING = auto()
    DRIZZLE = auto()
    ASTROMETRY = auto()
    PHOTOMETRY = auto()
    MOSAIC = auto()
    PROCESSING = auto()
    SAVING = auto()


@dataclass
class PipelineProgress:
    stage: PipelineStage
    current: int
    total: int
    message: str = ""

    @property
    def fraction(self) -> float:
        return self.current / max(self.total, 1)


ProgressCallback = Callable[[PipelineProgress], None]


def noop_callback(_progress: PipelineProgress) -> None:
    pass


@dataclass
class PipelineContext:
    images: list[NDArray[np.floating[Any]]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    result: NDArray[np.floating[Any]] | None = None
    starless_image: NDArray[np.floating[Any]] | None = None
    star_mask: NDArray[np.floating[Any]] | None = None


class PipelineStep(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    @abstractmethod
    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext: ...


class Pipeline:
    def __init__(self, steps: list[PipelineStep] | None = None) -> None:
        self._steps: list[PipelineStep] = steps or []

    def add(self, step: PipelineStep) -> Pipeline:
        self._steps.append(step)
        return self

    def run(
        self,
        context: PipelineContext,
        progress: ProgressCallback = noop_callback,
    ) -> PipelineContext:
        total = len(self._steps)
        for i, step in enumerate(self._steps):
            progress(PipelineProgress(
                stage=step.stage,
                current=i,
                total=total,
                message=f"Running: {step.name}",
            ))
            context = step.execute(context, progress)
        progress(PipelineProgress(
            stage=PipelineStage.SAVING,
            current=total,
            total=total,
            message="Pipeline complete",
        ))
        return context
