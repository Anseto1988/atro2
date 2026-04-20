from __future__ import annotations

from typing import Any

import numpy as np

from astroai.core.pipeline.base import (
    Pipeline,
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
)


class _DoubleStep(PipelineStep):
    @property
    def name(self) -> str:
        return "double"

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = lambda _: None,
    ) -> PipelineContext:
        context.images = [img * 2 for img in context.images]
        return context


class _AddOneStep(PipelineStep):
    @property
    def name(self) -> str:
        return "add_one"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PROCESSING

    def execute(
        self,
        context: PipelineContext,
        progress: ProgressCallback = lambda _: None,
    ) -> PipelineContext:
        context.images = [img + 1 for img in context.images]
        return context


class TestPipeline:
    def test_empty_pipeline(self) -> None:
        ctx = PipelineContext(images=[np.zeros((5, 5), dtype=np.float32)])
        result = Pipeline().run(ctx)
        assert len(result.images) == 1
        assert result.images[0].sum() == 0.0

    def test_single_step(self) -> None:
        img = np.ones((3, 3), dtype=np.float32)
        ctx = PipelineContext(images=[img])
        result = Pipeline([_DoubleStep()]).run(ctx)
        np.testing.assert_array_equal(result.images[0], np.full((3, 3), 2.0))

    def test_chained_steps(self) -> None:
        img = np.ones((2, 2), dtype=np.float32) * 3
        ctx = PipelineContext(images=[img])
        pipeline = Pipeline().add(_DoubleStep()).add(_AddOneStep())
        result = pipeline.run(ctx)
        np.testing.assert_array_equal(result.images[0], np.full((2, 2), 7.0))

    def test_progress_callback_called(self) -> None:
        events: list[PipelineProgress] = []
        ctx = PipelineContext(images=[np.zeros((2, 2), dtype=np.float32)])
        Pipeline([_DoubleStep(), _AddOneStep()]).run(ctx, progress=events.append)
        assert len(events) == 3
        assert events[0].stage == PipelineStage.PROCESSING
        assert events[0].current == 0
        assert events[-1].message == "Pipeline complete"

    def test_progress_fraction(self) -> None:
        p = PipelineProgress(stage=PipelineStage.LOADING, current=3, total=10)
        assert p.fraction == 0.3

    def test_progress_fraction_zero_total(self) -> None:
        p = PipelineProgress(stage=PipelineStage.LOADING, current=0, total=0)
        assert p.fraction == 0.0
