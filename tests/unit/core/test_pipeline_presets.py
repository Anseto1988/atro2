"""Tests for PipelinePreset and PresetManager."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from astroai.core.pipeline.presets import (
    PipelinePreset,
    PresetManager,
    _preset_dir,
    _safe_name,
)


@pytest.fixture
def manager(tmp_path: Path) -> PresetManager:
    return PresetManager(preset_dir=tmp_path / "presets")


class TestSafeName:
    def test_alphanumeric_unchanged(self) -> None:
        assert _safe_name("Deepsky") == "Deepsky"

    def test_spaces_preserved(self) -> None:
        assert " " in _safe_name("Deep Sky")

    def test_slashes_replaced(self) -> None:
        name = _safe_name("Ha/OIII")
        assert "/" not in name

    def test_length_capped(self) -> None:
        assert len(_safe_name("x" * 100)) <= 64


class TestPipelinePreset:
    def test_round_trip_dict(self) -> None:
        preset = PipelinePreset(
            name="Test",
            description="desc",
            config={"stacking_method": "sigma_clip", "denoise_strength": 1.5},
        )
        data = preset.to_dict()
        restored = PipelinePreset.from_dict(data)
        assert restored.name == "Test"
        assert restored.description == "desc"
        assert restored.config["stacking_method"] == "sigma_clip"
        assert restored.config["denoise_strength"] == pytest.approx(1.5)

    def test_from_dict_missing_keys_use_defaults(self) -> None:
        preset = PipelinePreset.from_dict({})
        assert preset.name == ""
        assert preset.config == {}

    def test_to_dict_structure(self) -> None:
        preset = PipelinePreset(name="N", config={"k": "v"})
        d = preset.to_dict()
        assert set(d.keys()) == {"name", "description", "config"}


class TestPresetManager:
    def test_list_names_empty_when_no_dir(self, manager: PresetManager) -> None:
        assert manager.list_names() == []

    def test_save_and_list(self, manager: PresetManager) -> None:
        manager.save(PipelinePreset(name="Alpha", config={}))
        assert "Alpha" in manager.list_names()

    def test_save_creates_json_file(self, manager: PresetManager, tmp_path: Path) -> None:
        manager.save(PipelinePreset(name="Test", config={"k": 1}))
        files = list((tmp_path / "presets").glob("*.json"))
        assert len(files) == 1

    def test_load_roundtrip(self, manager: PresetManager) -> None:
        original = PipelinePreset(name="Beta", description="d", config={"sigma": 2.5})
        manager.save(original)
        loaded = manager.load("Beta")
        assert loaded.name == "Beta"
        assert loaded.description == "d"
        assert loaded.config["sigma"] == pytest.approx(2.5)

    def test_load_missing_raises_file_not_found(self, manager: PresetManager) -> None:
        with pytest.raises(FileNotFoundError):
            manager.load("NonExistent")

    def test_delete_removes_file(self, manager: PresetManager) -> None:
        manager.save(PipelinePreset(name="ToDelete", config={}))
        assert manager.exists("ToDelete")
        manager.delete("ToDelete")
        assert not manager.exists("ToDelete")

    def test_delete_nonexistent_is_noop(self, manager: PresetManager) -> None:
        manager.delete("ghost")  # should not raise

    def test_exists_false_for_unknown(self, manager: PresetManager) -> None:
        assert not manager.exists("nope")

    def test_save_overwrite(self, manager: PresetManager) -> None:
        manager.save(PipelinePreset(name="P", config={"v": 1}))
        manager.save(PipelinePreset(name="P", config={"v": 2}))
        loaded = manager.load("P")
        assert loaded.config["v"] == 2

    def test_list_names_sorted(self, manager: PresetManager) -> None:
        for name in ("Zeta", "Alpha", "Mu"):
            manager.save(PipelinePreset(name=name, config={}))
        names = manager.list_names()
        assert names == sorted(names)

    def test_preset_dir_non_windows(self) -> None:
        with patch("sys.platform", "linux"):
            path = _preset_dir()
        assert ".config" in str(path) and "presets" in str(path)

    def test_preset_dir_windows(self) -> None:
        """Cover line 45: Windows branch uses AppData/Local/AstroAI."""
        with patch("sys.platform", "win32"):
            path = _preset_dir()
        assert "AppData" in str(path) and "AstroAI" in str(path) and "presets" in str(path)


class TestPresetModelIntegration:
    class _MockModel:
        def __init__(self) -> None:
            self.stacking_method = "sigma_clip"
            self.denoise_strength = 1.0
            self.stretch_target_background = 0.25

    def test_capture_reads_model_attrs(self, manager: PresetManager) -> None:
        model = self._MockModel()
        preset = manager.capture_from_model("Snap", model, description="snapshot")
        assert preset.config["stacking_method"] == "sigma_clip"
        assert preset.config["denoise_strength"] == pytest.approx(1.0)
        assert preset.description == "snapshot"

    def test_capture_skips_missing_attrs(self, manager: PresetManager) -> None:
        model = object()
        preset = manager.capture_from_model("Empty", model)
        assert preset.config == {}

    def test_apply_sets_model_attrs(self, manager: PresetManager) -> None:
        model = self._MockModel()
        preset = PipelinePreset(
            name="P",
            config={"stacking_method": "mean", "denoise_strength": 0.5},
        )
        manager.apply_to_model(preset, model)
        assert model.stacking_method == "mean"
        assert model.denoise_strength == pytest.approx(0.5)

    def test_apply_skips_unknown_keys(self, manager: PresetManager) -> None:
        model = self._MockModel()
        preset = PipelinePreset(name="P", config={"nonexistent_key": 42})
        manager.apply_to_model(preset, model)  # should not raise

    def test_apply_skips_read_only_attribute(self, manager: PresetManager) -> None:
        """Cover lines 129-130: setattr raises AttributeError or TypeError — silently skipped."""

        class _ReadOnlyModel:
            @property
            def stacking_method(self) -> str:
                return "sigma_clip"

        model = _ReadOnlyModel()
        preset = PipelinePreset(name="P", config={"stacking_method": "mean"})
        manager.apply_to_model(preset, model)  # must not raise

    def test_save_capture_load_apply_roundtrip(self, manager: PresetManager) -> None:
        model = self._MockModel()
        model.stacking_method = "median"
        preset = manager.capture_from_model("Round", model)
        manager.save(preset)
        loaded = manager.load("Round")
        target = self._MockModel()
        manager.apply_to_model(loaded, target)
        assert target.stacking_method == "median"
