"""Tests for RecentProjects manager."""
from __future__ import annotations

from pathlib import Path

import pytest
from astroai.project.recent_files import RecentProjects


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
