"""Tests for PipelineWorker."""
from __future__ import annotations

import time

import numpy as np
import pytest

from astroai.core.pipeline.base import (
    Pipeline,
    PipelineContext,
    PipelineProgress,
    PipelineStage,
    PipelineStep,
    ProgressCallback,
    noop_callback,
)
from astroai.core.pipeline.runner import PipelineWorker, _RunnerWorker


class _PassthroughStep(PipelineStep):
    @property
    def name(self) -> str:
        return "Passthrough"

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        progress(PipelineProgress(
            stage=PipelineStage.PROCESSING, current=1, total=1, message="Done"
        ))
        return context


class _ErrorStep(PipelineStep):
    @property
    def name(self) -> str:
        return "Error"

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        raise RuntimeError("deliberate step failure")


class _ResultStep(PipelineStep):
    @property
    def name(self) -> str:
        return "Result"

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        context.result = np.ones((4, 4), dtype=np.float32)
        return context


class _SlowStep(PipelineStep):
    @property
    def name(self) -> str:
        return "Slow"

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        time.sleep(0.05)
        return context


class _CalibrationStep(PipelineStep):
    @property
    def name(self) -> str:
        return "Calibration"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.CALIBRATION

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        progress(PipelineProgress(stage=self.stage, current=1, total=1))
        return context


class _RegistrationStep(PipelineStep):
    @property
    def name(self) -> str:
        return "Registration"

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.REGISTRATION

    def execute(
        self, context: PipelineContext, progress: ProgressCallback = noop_callback
    ) -> PipelineContext:
        progress(PipelineProgress(stage=self.stage, current=1, total=1))
        return context


class TestPipelineWorker:
    def test_not_running_initially(self) -> None:
        worker = PipelineWorker()
        assert worker.is_running is False

    def test_finished_signal_on_success(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_PassthroughStep()])
        context = PipelineContext()
        with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
            worker.start(pipeline, context)
        assert isinstance(blocker.args[0], PipelineContext)

    def test_result_propagated_in_finished_context(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_ResultStep()])
        context = PipelineContext()
        with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
            worker.start(pipeline, context)
        result_ctx: PipelineContext = blocker.args[0]
        assert result_ctx.result is not None

    def test_error_signal_on_step_failure(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_ErrorStep()])
        context = PipelineContext()
        with qtbot.waitSignal(worker.error, timeout=5000) as blocker:
            worker.start(pipeline, context)
        assert "deliberate" in blocker.args[0]

    def test_not_running_after_finish(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_PassthroughStep()])
        context = PipelineContext()
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start(pipeline, context)
        assert worker.is_running is False

    def test_not_running_after_error(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_ErrorStep()])
        context = PipelineContext()
        with qtbot.waitSignal(worker.error, timeout=5000):
            worker.start(pipeline, context)
        assert worker.is_running is False

    def test_progress_signal_emitted(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_PassthroughStep()])
        context = PipelineContext()
        fractions: list[float] = []
        worker.progress.connect(lambda f, _m: fractions.append(f))
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start(pipeline, context)
        assert len(fractions) >= 1

    def test_progress_fraction_between_0_and_1(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_PassthroughStep()])
        context = PipelineContext()
        fractions: list[float] = []
        worker.progress.connect(lambda f, _m: fractions.append(f))
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start(pipeline, context)
        for f in fractions:
            assert 0.0 <= f <= 1.0

    def test_second_start_while_running_is_ignored(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_SlowStep()])
        context = PipelineContext()
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start(pipeline, context)
            assert worker.is_running
            worker.start(pipeline, context)  # must not raise

    def test_empty_pipeline_finishes_successfully(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([])
        context = PipelineContext()
        with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
            worker.start(pipeline, context)
        assert isinstance(blocker.args[0], PipelineContext)

    def test_is_running_true_during_execution(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_SlowStep()])
        context = PipelineContext()
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start(pipeline, context)
            assert worker.is_running is True

    def test_stage_active_emitted_on_stage_entry(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_CalibrationStep()])
        stages: list[str] = []
        worker.stage_active.connect(stages.append)
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start(pipeline, PipelineContext())
        # Pipeline.run emits "Running: X" before calling execute → CALIBRATION stage
        assert "CALIBRATION" in stages

    def test_stage_active_emitted_only_once_per_stage(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """stage_active must not spam the same stage repeatedly."""
        worker = PipelineWorker()

        class _MultiProgressCalibStep(PipelineStep):
            @property
            def name(self) -> str:
                return "Multi"
            @property
            def stage(self) -> PipelineStage:
                return PipelineStage.CALIBRATION
            def execute(self, ctx: PipelineContext, progress: ProgressCallback = noop_callback) -> PipelineContext:
                for i in range(5):
                    progress(PipelineProgress(stage=self.stage, current=i, total=5))
                return ctx

        pipeline = Pipeline([_MultiProgressCalibStep()])
        stages: list[str] = []
        worker.stage_active.connect(stages.append)
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start(pipeline, PipelineContext())
        assert stages.count("CALIBRATION") == 1

    def test_stage_active_emits_all_stages_in_order(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_CalibrationStep(), _RegistrationStep()])
        stages: list[str] = []
        worker.stage_active.connect(stages.append)
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start(pipeline, PipelineContext())
        # Both stages appear and CALIBRATION precedes REGISTRATION
        assert "CALIBRATION" in stages
        assert "REGISTRATION" in stages
        assert stages.index("CALIBRATION") < stages.index("REGISTRATION")

    def test_stage_active_reset_between_runs(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """A second run must re-emit stages that were seen in the first run."""
        worker = PipelineWorker()
        pipeline = Pipeline([_CalibrationStep()])
        run1_stages: list[str] = []
        worker.stage_active.connect(run1_stages.append)
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start(pipeline, PipelineContext())
        worker.stage_active.disconnect()

        run2_stages: list[str] = []
        worker.stage_active.connect(run2_stages.append)
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start(pipeline, PipelineContext())
        assert "CALIBRATION" in run2_stages


class TestPipelineWorkerCancel:
    def test_cancel_before_start_is_noop(self) -> None:
        worker = PipelineWorker()
        worker.cancel()  # must not raise
        assert not worker.is_running

    def test_cancelled_signal_emitted(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()

        class _BlockStep(PipelineStep):
            @property
            def name(self) -> str:
                return "Block"
            def execute(self, ctx: PipelineContext, progress: ProgressCallback = noop_callback) -> PipelineContext:
                time.sleep(0.05)
                return ctx

        pipeline = Pipeline([_BlockStep(), _BlockStep()])
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start(pipeline, PipelineContext())
            worker.cancel()

    def test_cancel_prevents_downstream_steps(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        executed: list[str] = []

        class _TrackStep(PipelineStep):
            def __init__(self, label: str) -> None:
                self._label = label
            @property
            def name(self) -> str:
                return self._label
            def execute(self, ctx: PipelineContext, progress: ProgressCallback = noop_callback) -> PipelineContext:
                executed.append(self._label)
                time.sleep(0.02)
                return ctx

        pipeline = Pipeline([_TrackStep("A"), _TrackStep("B"), _TrackStep("C")])
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start(pipeline, PipelineContext())
            # Cancel immediately — A may run but B/C should not
            worker.cancel()
        # At most "A" ran before cancel was honoured
        assert "C" not in executed

    def test_not_running_after_cancel(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        worker = PipelineWorker()
        pipeline = Pipeline([_SlowStep()])
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start(pipeline, PipelineContext())
            worker.cancel()
        assert not worker.is_running

    def test_cancel_event_cleared_after_run(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """Second run after cancel must complete successfully."""
        worker = PipelineWorker()
        pipeline = Pipeline([_SlowStep()])
        with qtbot.waitSignal(worker.cancelled, timeout=5000):
            worker.start(pipeline, PipelineContext())
            worker.cancel()

        # Second run must succeed, not be instantly cancelled
        with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
            worker.start(Pipeline([_PassthroughStep()]), PipelineContext())
        assert isinstance(blocker.args[0], PipelineContext)


class TestPipelineCancelledError:
    def test_cancel_check_raises_immediately(self) -> None:
        from astroai.core.pipeline.base import PipelineCancelledError

        pipeline = Pipeline([_PassthroughStep()])
        ctx = PipelineContext()
        with pytest.raises(PipelineCancelledError):
            pipeline.run(ctx, cancel_check=lambda: True)

    def test_no_cancel_runs_normally(self) -> None:
        pipeline = Pipeline([_ResultStep()])
        ctx = PipelineContext()
        result = pipeline.run(ctx, cancel_check=lambda: False)
        assert result.result is not None


class TestRunnerWorkerDirect:
    """Direct unit tests for _RunnerWorker, bypassing QThread for coverage."""

    def _make(self, pipeline: Pipeline, cancelled: bool = False) -> _RunnerWorker:
        import threading
        cancel_event = threading.Event()
        if cancelled:
            cancel_event.set()
        return _RunnerWorker(pipeline, PipelineContext(), cancel_event)

    def test_run_emits_finished_on_success(self) -> None:
        worker = self._make(Pipeline([_PassthroughStep()]))
        results: list[object] = []
        worker.finished.connect(results.append)
        worker.run()
        assert len(results) == 1
        assert isinstance(results[0], PipelineContext)

    def test_run_emits_error_on_exception(self) -> None:
        worker = self._make(Pipeline([_ErrorStep()]))
        errors: list[str] = []
        worker.error.connect(errors.append)
        worker.run()
        assert errors and "deliberate" in errors[0]

    def test_run_emits_cancelled_when_pre_cancelled(self) -> None:
        worker = self._make(Pipeline([_PassthroughStep()]), cancelled=True)
        cancelled: list[bool] = []
        worker.cancelled.connect(lambda: cancelled.append(True))
        worker.run()
        assert cancelled == [True]

    def test_emit_progress_relays_signal(self) -> None:
        worker = self._make(Pipeline([_PassthroughStep()]))
        received: list[object] = []
        worker.progress.connect(received.append)
        p = PipelineProgress(stage=PipelineStage.PROCESSING, current=1, total=1)
        worker._emit_progress(p)
        assert received == [p]
