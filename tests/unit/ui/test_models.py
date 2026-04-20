"""Tests for PipelineModel and related data classes."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel, PipelineStep, StepState


class TestPipelineStep:
    def test_initial_state(self) -> None:
        step = PipelineStep("cal", "Kalibrierung")
        assert step.key == "cal"
        assert step.label == "Kalibrierung"
        assert step.state is StepState.PENDING
        assert step.progress == 0.0


class TestPipelineModel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    def test_default_steps(self, model: PipelineModel) -> None:
        steps = model.steps
        assert len(steps) == 5
        assert steps[0].key == "calibrate"
        assert steps[-1].key == "denoise"

    def test_step_by_key(self, model: PipelineModel) -> None:
        assert model.step_by_key("stack") is not None
        assert model.step_by_key("nonexistent") is None

    def test_set_step_state(self, model: PipelineModel) -> None:
        model.set_step_state("calibrate", StepState.ACTIVE)
        step = model.step_by_key("calibrate")
        assert step is not None
        assert step.state is StepState.ACTIVE

    def test_set_step_progress_clamps(self, model: PipelineModel) -> None:
        model.set_step_progress("stack", 1.5)
        step = model.step_by_key("stack")
        assert step is not None
        assert step.progress == 1.0

        model.set_step_progress("stack", -0.3)
        assert step.progress == 0.0

    def test_active_step(self, model: PipelineModel) -> None:
        assert model.active_step() is None
        model.set_step_state("register", StepState.ACTIVE)
        active = model.active_step()
        assert active is not None
        assert active.key == "register"

    def test_reset(self, model: PipelineModel) -> None:
        model.set_step_state("calibrate", StepState.DONE)
        model.set_step_progress("calibrate", 1.0)
        model.reset()
        for step in model.steps:
            assert step.state is StepState.PENDING
            assert step.progress == 0.0

    def test_advance_to(self, model: PipelineModel) -> None:
        model.advance_to("stack")
        steps = model.steps
        assert steps[0].state is StepState.DONE
        assert steps[1].state is StepState.DONE
        assert steps[2].state is StepState.ACTIVE
        assert steps[3].state is StepState.PENDING

    def test_step_changed_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.step_changed, timeout=500):
            model.set_step_state("calibrate", StepState.ACTIVE)

    def test_progress_changed_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.progress_changed, timeout=500):
            model.set_step_progress("calibrate", 0.5)

    def test_pipeline_reset_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.pipeline_reset, timeout=500):
            model.reset()
