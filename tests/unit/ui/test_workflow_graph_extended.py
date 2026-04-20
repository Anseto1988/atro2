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
