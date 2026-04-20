"""Tests for custom AstroAI widgets."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.ui.models import PipelineModel, StepState
from astroai.ui.widgets.histogram_widget import HistogramWidget
from astroai.ui.widgets.image_viewer import ImageViewer
from astroai.ui.widgets.progress_widget import ProgressWidget
from astroai.ui.widgets.workflow_graph import WorkflowGraph


class TestImageViewer:
    @pytest.fixture()
    def viewer(self, qtbot) -> ImageViewer:  # type: ignore[no-untyped-def]
        w = ImageViewer()
        qtbot.addWidget(w)
        return w

    def test_initial_state(self, viewer: ImageViewer) -> None:
        assert viewer._raw_data is None
        assert viewer.zoom_level == 1.0

    def test_set_image_data(self, viewer: ImageViewer) -> None:
        data = np.random.rand(100, 200).astype(np.float32)
        viewer.set_image_data(data)
        assert viewer._raw_data is not None
        assert viewer._width == 200
        assert viewer._height == 100

    def test_set_image_3d_reduces(self, viewer: ImageViewer) -> None:
        data = np.random.rand(3, 100, 200).astype(np.float32)
        viewer.set_image_data(data)
        assert viewer._raw_data is not None
        assert viewer._raw_data.ndim == 2

    def test_clear(self, viewer: ImageViewer) -> None:
        viewer.set_image_data(np.zeros((10, 10), dtype=np.float32))
        viewer.clear()
        assert viewer._raw_data is None

    def test_set_zoom(self, viewer: ImageViewer) -> None:
        viewer.set_zoom(2.5)
        assert viewer.zoom_level == 2.5

    def test_zoom_clamps(self, viewer: ImageViewer) -> None:
        viewer.set_zoom(100.0)
        assert viewer.zoom_level == 50.0
        viewer.set_zoom(0.001)
        assert viewer.zoom_level == 0.05

    def test_zoom_signal(self, viewer: ImageViewer, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(viewer.zoom_changed, timeout=500):
            viewer.set_zoom(3.0)


class TestHistogramWidget:
    @pytest.fixture()
    def histogram(self, qtbot) -> HistogramWidget:  # type: ignore[no-untyped-def]
        w = HistogramWidget()
        qtbot.addWidget(w)
        return w

    def test_initial_state(self, histogram: HistogramWidget) -> None:
        assert histogram._bins is None

    def test_set_image_data(self, histogram: HistogramWidget) -> None:
        data = np.random.rand(50, 50).astype(np.float32)
        histogram.set_image_data(data)
        assert histogram._bins is not None
        assert len(histogram._bins) == 256

    def test_constant_image_clears(self, histogram: HistogramWidget) -> None:
        data = np.ones((10, 10), dtype=np.float32)
        histogram.set_image_data(data)
        assert histogram._bins is None

    def test_clear(self, histogram: HistogramWidget) -> None:
        histogram.set_image_data(np.random.rand(10, 10).astype(np.float32))
        histogram.clear()
        assert histogram._bins is None


class TestWorkflowGraph:
    @pytest.fixture()
    def graph(self, qtbot) -> WorkflowGraph:  # type: ignore[no-untyped-def]
        model = PipelineModel()
        w = WorkflowGraph(model)
        qtbot.addWidget(w)
        return w

    def test_has_model(self, graph: WorkflowGraph) -> None:
        assert graph._model is not None

    def test_updates_on_step_change(self, graph: WorkflowGraph) -> None:
        graph._model.set_step_state("calibrate", StepState.ACTIVE)
        step = graph._model.step_by_key("calibrate")
        assert step is not None
        assert step.state is StepState.ACTIVE

    def test_size_hint(self, graph: WorkflowGraph) -> None:
        hint = graph.sizeHint()
        assert hint.width() >= 200
        assert hint.height() > 0


class TestProgressWidget:
    @pytest.fixture()
    def progress(self, qtbot) -> ProgressWidget:  # type: ignore[no-untyped-def]
        w = ProgressWidget()
        qtbot.addWidget(w)
        return w

    def test_initial_state(self, progress: ProgressWidget) -> None:
        assert progress._label.text() == "Bereit"
        assert progress._bar.value() == 0
        assert not progress._cancel_btn.isEnabled()

    def test_set_progress(self, progress: ProgressWidget) -> None:
        progress.set_progress(0.5)
        assert progress._bar.value() == 500

    def test_set_status(self, progress: ProgressWidget) -> None:
        progress.set_status("Kalibrierung...")
        assert progress._label.text() == "Kalibrierung..."

    def test_set_cancellable(self, progress: ProgressWidget) -> None:
        progress.set_cancellable(True)
        assert progress._cancel_btn.isEnabled()

    def test_reset(self, progress: ProgressWidget) -> None:
        progress.set_progress(0.7)
        progress.set_status("Test")
        progress.set_cancellable(True)
        progress.reset()
        assert progress._label.text() == "Bereit"
        assert progress._bar.value() == 0
        assert not progress._cancel_btn.isEnabled()

    def test_cancel_signal(self, progress: ProgressWidget, qtbot) -> None:  # type: ignore[no-untyped-def]
        progress.set_cancellable(True)
        with qtbot.waitSignal(progress.cancel_requested, timeout=500):
            progress._cancel_btn.click()

    def test_indeterminate(self, progress: ProgressWidget) -> None:
        progress.set_indeterminate()
        assert progress._bar.maximum() == 0

    def test_determinate(self, progress: ProgressWidget) -> None:
        progress.set_indeterminate()
        progress.set_determinate()
        assert progress._bar.maximum() == 1000
