from __future__ import annotations

import pytest

from astroai.core.processing_history import HistoryEntry, ProcessingHistory


@pytest.fixture()
def history(qapp):
    return ProcessingHistory(max_depth=5)


class TestHistoryEntry:
    def test_frozen_dataclass(self) -> None:
        entry = HistoryEntry(step_name="Stretch", params={"a": 1})
        assert entry.step_name == "Stretch"
        assert entry.params == {"a": 1}
        assert entry.timestamp > 0
        with pytest.raises(AttributeError):
            entry.step_name = "other"  # type: ignore[misc]


class TestProcessingHistoryInit:
    def test_default_max_depth(self, qapp) -> None:
        h = ProcessingHistory()
        assert h.max_depth == 20

    def test_custom_max_depth(self, history: ProcessingHistory) -> None:
        assert history.max_depth == 5

    def test_min_depth_clamped(self, qapp) -> None:
        h = ProcessingHistory(max_depth=0)
        assert h.max_depth == 1

    def test_empty_state(self, history: ProcessingHistory) -> None:
        assert not history.can_undo
        assert not history.can_redo
        assert history.undo_count == 0
        assert history.redo_count == 0
        assert history.undo_step_name is None
        assert history.redo_step_name is None


class TestPush:
    def test_push_single(self, history: ProcessingHistory) -> None:
        history.push("Stretch", {"target": 0.25})
        assert history.can_undo
        assert history.undo_count == 1
        assert history.undo_step_name == "Stretch"

    def test_push_clears_redo(self, history: ProcessingHistory) -> None:
        history.push("Stretch", {"target": 0.25})
        history.push("Curves", {"points": [(0, 0), (1, 1)]})
        history.undo()
        assert history.can_redo
        history.push("Denoise", {"strength": 0.5})
        assert not history.can_redo

    def test_push_copies_params(self, history: ProcessingHistory) -> None:
        params = {"key": [1, 2, 3]}
        history.push("Step", params)
        params["key"].append(4)
        assert history.peek_undo().params["key"] == [1, 2, 3]


class TestMaxDepth:
    def test_lru_drop(self, history: ProcessingHistory) -> None:
        for i in range(8):
            history.push(f"Step-{i}", {"i": i})
        assert history.undo_count == 5
        assert history.peek_undo().params["i"] == 7
        oldest = history.entries()[0]
        assert oldest.params["i"] == 3

    def test_set_max_depth_shrinks(self, history: ProcessingHistory) -> None:
        for i in range(5):
            history.push(f"Step-{i}", {"i": i})
        history.max_depth = 2
        assert history.undo_count == 2
        assert history.entries()[0].params["i"] == 3


class TestUndo:
    def test_undo_returns_entry(self, history: ProcessingHistory) -> None:
        history.push("Stretch", {"v": 1})
        entry = history.undo()
        assert entry is not None
        assert entry.step_name == "Stretch"

    def test_undo_empty(self, history: ProcessingHistory) -> None:
        assert history.undo() is None

    def test_undo_moves_to_redo(self, history: ProcessingHistory) -> None:
        history.push("A", {})
        history.push("B", {})
        history.undo()
        assert history.can_redo
        assert history.redo_step_name == "B"
        assert history.undo_count == 1

    def test_multiple_undo(self, history: ProcessingHistory) -> None:
        history.push("A", {})
        history.push("B", {})
        history.push("C", {})
        history.undo()
        history.undo()
        history.undo()
        assert not history.can_undo
        assert history.redo_count == 3


class TestRedo:
    def test_redo_returns_entry(self, history: ProcessingHistory) -> None:
        history.push("Stretch", {"v": 1})
        history.undo()
        entry = history.redo()
        assert entry is not None
        assert entry.step_name == "Stretch"

    def test_redo_empty(self, history: ProcessingHistory) -> None:
        assert history.redo() is None

    def test_redo_moves_back_to_undo(self, history: ProcessingHistory) -> None:
        history.push("A", {})
        history.undo()
        history.redo()
        assert history.can_undo
        assert not history.can_redo
        assert history.undo_step_name == "A"


class TestClear:
    def test_clear(self, history: ProcessingHistory) -> None:
        history.push("A", {})
        history.push("B", {})
        history.undo()
        history.clear()
        assert not history.can_undo
        assert not history.can_redo
        assert history.undo_count == 0
        assert history.redo_count == 0


class TestEntries:
    def test_entries_returns_copy(self, history: ProcessingHistory) -> None:
        history.push("A", {})
        entries = history.entries()
        entries.clear()
        assert history.undo_count == 1


class TestPeek:
    def test_peek_undo(self, history: ProcessingHistory) -> None:
        history.push("A", {"x": 1})
        assert history.peek_undo().step_name == "A"
        assert history.undo_count == 1  # peek does not pop

    def test_peek_redo(self, history: ProcessingHistory) -> None:
        history.push("A", {})
        history.undo()
        assert history.peek_redo().step_name == "A"
        assert history.redo_count == 1


class TestSignals:
    def test_push_emits_signal(self, history: ProcessingHistory) -> None:
        calls: list[bool] = []
        history.history_changed.connect(lambda: calls.append(True))
        history.push("X", {})
        assert len(calls) == 1

    def test_undo_emits_signal(self, history: ProcessingHistory) -> None:
        history.push("X", {})
        calls: list[bool] = []
        history.history_changed.connect(lambda: calls.append(True))
        history.undo()
        assert len(calls) == 1

    def test_redo_emits_signal(self, history: ProcessingHistory) -> None:
        history.push("X", {})
        history.undo()
        calls: list[bool] = []
        history.history_changed.connect(lambda: calls.append(True))
        history.redo()
        assert len(calls) == 1

    def test_clear_emits_signal(self, history: ProcessingHistory) -> None:
        calls: list[bool] = []
        history.history_changed.connect(lambda: calls.append(True))
        history.clear()
        assert len(calls) == 1
