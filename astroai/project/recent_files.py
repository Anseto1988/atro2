"""Manage list of recently opened project files."""
from __future__ import annotations

import json
from pathlib import Path

_MAX_RECENT = 10


def _config_path() -> Path:
    import sys

    if sys.platform == "win32":
        base = Path.home() / "AppData" / "Local" / "AstroAI"
    else:
        base = Path.home() / ".config" / "astroai"
    return base / "recent_projects.json"


class RecentProjects:
    def __init__(self, config_path: Path | None = None) -> None:
        self._path = config_path or _config_path()
        self._entries: list[str] = self._load()

    @property
    def entries(self) -> list[str]:
        return list(self._entries)

    def add(self, path: Path | str) -> None:
        resolved = str(Path(path).resolve())
        if resolved in self._entries:
            self._entries.remove(resolved)
        self._entries.insert(0, resolved)
        self._entries = self._entries[:_MAX_RECENT]
        self._persist()

    def remove(self, path: Path | str) -> None:
        resolved = str(Path(path).resolve())
        if resolved in self._entries:
            self._entries.remove(resolved)
            self._persist()

    def clear(self) -> None:
        self._entries.clear()
        self._persist()

    def _load(self) -> list[str]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(e) for e in data[:_MAX_RECENT]]
        except (OSError, json.JSONDecodeError):
            pass
        return []

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._entries, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
