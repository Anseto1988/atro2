"""Undo/Redo history for pipeline processing steps (parameter snapshots only)."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class HistoryEntry:
    step_name: str
    params: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class ProcessingHistory(QObject):
    history_changed = Signal()

    DEFAULT_MAX_DEPTH = 20

    def __init__(
        self,
        max_depth: int = DEFAULT_MAX_DEPTH,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._max_depth = max(1, max_depth)
        self._undo_stack: list[HistoryEntry] = []
        self._redo_stack: list[HistoryEntry] = []

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        self._max_depth = max(1, value)
        self._enforce_limit()
        self.history_changed.emit()

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_step_name(self) -> str | None:
        return self._undo_stack[-1].step_name if self._undo_stack else None

    @property
    def redo_step_name(self) -> str | None:
        return self._redo_stack[-1].step_name if self._redo_stack else None

    @property
    def undo_count(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        return len(self._redo_stack)

    def push(self, step_name: str, params: dict[str, Any]) -> None:
        entry = HistoryEntry(step_name=step_name, params=copy.deepcopy(params))
        self._undo_stack.append(entry)
        self._redo_stack.clear()
        self._enforce_limit()
        self.history_changed.emit()

    def undo(self) -> HistoryEntry | None:
        if not self._undo_stack:
            return None
        entry = self._undo_stack.pop()
        self._redo_stack.append(entry)
        self.history_changed.emit()
        return entry

    def redo(self) -> HistoryEntry | None:
        if not self._redo_stack:
            return None
        entry = self._redo_stack.pop()
        self._undo_stack.append(entry)
        self.history_changed.emit()
        return entry

    def peek_undo(self) -> HistoryEntry | None:
        return self._undo_stack[-1] if self._undo_stack else None

    def peek_redo(self) -> HistoryEntry | None:
        return self._redo_stack[-1] if self._redo_stack else None

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.history_changed.emit()

    def entries(self) -> list[HistoryEntry]:
        return list(self._undo_stack)

    def _enforce_limit(self) -> None:
        while len(self._undo_stack) > self._max_depth:
            self._undo_stack.pop(0)
