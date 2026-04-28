"""Tests for built-in pipeline preset library."""
from __future__ import annotations

from pathlib import Path

import pytest

from astroai.core.pipeline.builtin_presets import (
    BUILTIN_PRESET_NAMES,
    BUILTIN_PRESETS,
    install_builtin_presets,
)
from astroai.core.pipeline.presets import PresetManager


@pytest.fixture
def manager(tmp_path: Path) -> PresetManager:
    return PresetManager(preset_dir=tmp_path / "presets")


class TestBuiltinPresetDefinitions:
    def test_exactly_four_presets(self) -> None:
        assert len(BUILTIN_PRESETS) == 4

    def test_all_names_non_empty(self) -> None:
        for p in BUILTIN_PRESETS:
            assert p.name.strip(), f"Preset hat leeren Namen: {p!r}"

    def test_all_presets_have_stacking_method(self) -> None:
        for p in BUILTIN_PRESETS:
            assert "stacking_method" in p.config, f"Fehlendes stacking_method in {p.name!r}"

    def test_all_presets_have_descriptions(self) -> None:
        for p in BUILTIN_PRESETS:
            assert p.description.strip(), f"Fehlende Beschreibung in {p.name!r}"

    def test_builtin_preset_names_constant_matches(self) -> None:
        assert set(BUILTIN_PRESET_NAMES) == {p.name for p in BUILTIN_PRESETS}

    def test_expected_preset_names_present(self) -> None:
        names = set(BUILTIN_PRESET_NAMES)
        assert "Deepsky LRGB" in names
        assert "Narrowband SHO" in names
        assert "Narrowband HOO" in names
        assert "Planetarisch" in names

    def test_deepsky_lrgb_uses_sigma_clip(self) -> None:
        preset = next(p for p in BUILTIN_PRESETS if p.name == "Deepsky LRGB")
        assert preset.config["stacking_method"] == "sigma_clip"

    def test_deepsky_lrgb_linked_channels(self) -> None:
        preset = next(p for p in BUILTIN_PRESETS if p.name == "Deepsky LRGB")
        assert preset.config["stretch_linked_channels"] is True

    def test_narrowband_sho_unlinked_channels(self) -> None:
        preset = next(p for p in BUILTIN_PRESETS if p.name == "Narrowband SHO")
        assert preset.config["stretch_linked_channels"] is False

    def test_narrowband_hoo_uses_median(self) -> None:
        preset = next(p for p in BUILTIN_PRESETS if p.name == "Narrowband HOO")
        assert preset.config["stacking_method"] == "median"

    def test_planetary_drizzle_enabled(self) -> None:
        preset = next(p for p in BUILTIN_PRESETS if p.name == "Planetarisch")
        assert preset.config["drizzle_enabled"] is True

    def test_planetary_frame_selection_enabled(self) -> None:
        preset = next(p for p in BUILTIN_PRESETS if p.name == "Planetarisch")
        assert preset.config["frame_selection_enabled"] is True

    def test_planetary_no_background_removal(self) -> None:
        preset = next(p for p in BUILTIN_PRESETS if p.name == "Planetarisch")
        assert preset.config["background_removal_enabled"] is False


class TestInstallBuiltinPresets:
    def test_installs_all_to_empty_manager(self, manager: PresetManager) -> None:
        count = install_builtin_presets(manager)
        assert count == 4

    def test_all_presets_accessible_after_install(self, manager: PresetManager) -> None:
        install_builtin_presets(manager)
        for name in BUILTIN_PRESET_NAMES:
            assert manager.exists(name), f"Preset nach Installation nicht gefunden: {name!r}"

    def test_skips_existing_preset(self, manager: PresetManager) -> None:
        manager.save(BUILTIN_PRESETS[0])
        count = install_builtin_presets(manager)
        assert count == 3

    def test_idempotent_second_call_returns_zero(self, manager: PresetManager) -> None:
        install_builtin_presets(manager)
        count = install_builtin_presets(manager)
        assert count == 0

    def test_installed_preset_roundtrip(self, manager: PresetManager) -> None:
        install_builtin_presets(manager)
        loaded = manager.load("Deepsky LRGB")
        assert loaded.config["stacking_method"] == "sigma_clip"
        assert loaded.description

    def test_install_creates_preset_dir(self, tmp_path: Path) -> None:
        subdir = tmp_path / "nested" / "presets"
        mgr = PresetManager(preset_dir=subdir)
        install_builtin_presets(mgr)
        assert subdir.is_dir()
