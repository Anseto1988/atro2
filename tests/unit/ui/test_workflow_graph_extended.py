"""Extended tests for WorkflowGraph — paint with various states."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSize

from astroai.ui.models import PipelineModel, StepState
from astroai.ui.widgets.workflow_graph import WorkflowGraph


@pytest.fixture()
def model() -> PipelineModel:
    return PipelineModel()


@pytest.fixture()
def graph(qtbot, model: PipelineModel) -> WorkflowGraph:
    w = WorkflowGraph(model)
    w.resize(800, 100)
    qtbot.addWidget(w)
    return w


class TestWorkflowGraphPaint:
    def test_paint_all_pending(self, graph: WorkflowGraph) -> None:
        graph.repaint()

    def test_paint_with_active_step(self, graph: WorkflowGraph, model: PipelineModel) -> None:
        model.set_step_state("calibrate", StepState.ACTIVE)
        graph.repaint()

    def test_paint_active_with_progress(self, graph: WorkflowGraph, model: PipelineModel) -> None:
        model.set_step_state("stack", StepState.ACTIVE)
        model.set_step_progress("stack", 0.5)
        graph.repaint()

    def test_paint_mixed_states(self, graph: WorkflowGraph, model: PipelineModel) -> None:
        model.set_step_state("calibrate", StepState.DONE)
        model.set_step_state("register", StepState.DONE)
        model.set_step_state("stack", StepState.ACTIVE)
        model.set_step_progress("stack", 0.3)
        model.set_step_state("stretch", StepState.PENDING)
        model.set_step_state("denoise", StepState.PENDING)
        graph.repaint()

    def test_paint_with_error_state(self, graph: WorkflowGraph, model: PipelineModel) -> None:
        model.set_step_state("register", StepState.ERROR)
        graph.repaint()

    def test_paint_all_done(self, graph: WorkflowGraph, model: PipelineModel) -> None:
        for step in model.steps:
            model.set_step_state(step.key, StepState.DONE)
        graph.repaint()

    def test_paint_after_reset(self, graph: WorkflowGraph, model: PipelineModel) -> None:
        model.advance_to("stack")
        model.reset()
        graph.repaint()


class TestWorkflowGraphLayout:
    def test_size_hint_reasonable(self, graph: WorkflowGraph) -> None:
        hint = graph.sizeHint()
        assert hint.width() >= 200
        assert hint.height() >= 48 + 24

    def test_total_width(self, graph: WorkflowGraph) -> None:
        tw = graph._total_width()
        assert tw > 0
        n = len(graph._model.steps)
        assert tw == n * 120 + (n - 1) * 24

    def test_minimum_height(self, graph: WorkflowGraph) -> None:
        assert graph.minimumHeight() >= 48 + 24

    def test_minimum_width(self, graph: WorkflowGraph) -> None:
        assert graph.minimumWidth() >= 200


class TestWorkflowGraphPaintForced:
    """Uses show()+grab() to force synchronous paintEvent execution."""

    @pytest.fixture()
    def shown_graph(self, qtbot, model):
        w = WorkflowGraph(model)
        w.resize(800, 100)
        qtbot.addWidget(w)
        w.show()
        return w

    def test_grab_all_pending(self, shown_graph):
        shown_graph.grab()

    def test_grab_with_disabled_step(self, shown_graph, model):
        model.set_step_state("calibrate", StepState.DISABLED)
        shown_graph.grab()

    def test_grab_active_with_progress(self, shown_graph, model):
        model.set_step_state("stack", StepState.ACTIVE)
        model.set_step_progress("stack", 0.6)
        shown_graph.grab()

    def test_grab_active_zero_progress(self, shown_graph, model):
        model.set_step_state("stack", StepState.ACTIVE)
        model.set_step_progress("stack", 0.0)
        shown_graph.grab()

    def test_grab_error_state(self, shown_graph, model):
        model.set_step_state("register", StepState.ERROR)
        shown_graph.grab()

    def test_grab_all_done(self, shown_graph, model):
        for step in model.steps:
            model.set_step_state(step.key, StepState.DONE)
        shown_graph.grab()

    def test_grab_empty_steps_no_crash(self, qtbot, model, monkeypatch):
        monkeypatch.setattr(type(model), "steps", property(lambda self: []))
        w = WorkflowGraph(model)
        w.resize(800, 100)
        qtbot.addWidget(w)
        w.show()
        w.grab()


class TestWorkflowGraphSignals:
    def test_step_changed_triggers_update(self, graph: WorkflowGraph, model: PipelineModel) -> None:
        model.set_step_state("calibrate", StepState.ACTIVE)
        graph.repaint()

    def test_reset_triggers_update(self, graph: WorkflowGraph, model: PipelineModel) -> None:
        model.advance_to("denoise")
        model.reset()
        graph.repaint()

    def test_accessible_name(self, graph: WorkflowGraph) -> None:
        assert graph.accessibleName() == "Pipeline-Workflow"

    def test_tooltip_present(self, graph: WorkflowGraph) -> None:
        assert "Pipeline" in graph.toolTip() or "Verarbeitungs" in graph.toolTip()
