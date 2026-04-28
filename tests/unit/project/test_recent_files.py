"""Tests for RecentProjects manager."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from astroai.project.recent_files import RecentProjects, _config_path


@pytest.fixture
def recent(tmp_path: Path) -> RecentProjects:
    return RecentProjects(config_path=tmp_path / "recent.json")


class TestRecentProjects:
    def test_initially_empty(self, recent: RecentProjects):
        assert recent.entries == []

    def test_add_single(self, recent: RecentProjects, tmp_path: Path):
        recent.add(tmp_path / "a.astroai")
        assert len(recent.entries) == 1

    def test_add_moves_to_front(self, recent: RecentProjects, tmp_path: Path):
        recent.add(tmp_path / "a.astroai")
        recent.add(tmp_path / "b.astroai")
        recent.add(tmp_path / "a.astroai")
        assert len(recent.entries) == 2
        assert "a.astroai" in recent.entries[0]

    def test_max_entries_enforced(self, recent: RecentProjects, tmp_path: Path):
        for i in range(15):
            recent.add(tmp_path / f"proj_{i}.astroai")
        assert len(recent.entries) == 10

    def test_remove(self, recent: RecentProjects, tmp_path: Path):
        recent.add(tmp_path / "a.astroai")
        recent.remove(tmp_path / "a.astroai")
        assert recent.entries == []

    def test_clear(self, recent: RecentProjects, tmp_path: Path):
        recent.add(tmp_path / "a.astroai")
        recent.add(tmp_path / "b.astroai")
        recent.clear()
        assert recent.entries == []

    def test_persistence_across_instances(self, tmp_path: Path):
        cfg = tmp_path / "recent.json"
        r1 = RecentProjects(config_path=cfg)
        r1.add(tmp_path / "x.astroai")
        r2 = RecentProjects(config_path=cfg)
        assert len(r2.entries) == 1

    def test_load_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        """_load silently returns [] on invalid JSON (lines 54-56)."""
        cfg = tmp_path / "recent.json"
        cfg.write_text("{not valid json}", encoding="utf-8")
        r = RecentProjects(config_path=cfg)
        assert r.entries == []

    def test_persist_oserror_swallowed(self, tmp_path: Path) -> None:
        """_persist OSError is silently ignored (lines 62-63)."""
        cfg = tmp_path / "recent.json"
        r = RecentProjects(config_path=cfg)
        with patch.object(type(cfg), "write_text", side_effect=OSError("no space")):
            r.add(tmp_path / "a.astroai")  # should not raise

    def test_config_path_non_windows(self) -> None:
        """_config_path returns ~/.config/astroai path on non-Windows (line 16)."""
        with patch("sys.platform", "linux"):
            path = _config_path()
        assert ".config" in str(path) and "astroai" in str(path)
