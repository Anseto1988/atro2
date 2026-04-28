"""Named pipeline configuration presets — save/load/apply to PipelineModel."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PRESET_EXTENSION = ".json"
_MAX_NAME_LEN = 64

# Keys captured from PipelineModel for preset persistence
_MODEL_KEYS: tuple[str, ...] = (
    "stacking_method",
    "stacking_sigma_low",
    "stacking_sigma_high",
    "stretch_target_background",
    "stretch_shadow_clipping_sigmas",
    "stretch_linked_channels",
    "denoise_strength",
    "denoise_tile_size",
    "denoise_tile_overlap",
    "background_removal_enabled",
    "background_removal_method",
    "background_removal_tile_size",
    "background_removal_preserve_median",
    "drizzle_enabled",
    "drizzle_drop_size",
    "drizzle_scale",
    "drizzle_pixfrac",
    "frame_selection_enabled",
    "frame_selection_min_score",
    "comet_stack_enabled",
    "comet_tracking_mode",
    "synthetic_flat_enabled",
    "deconvolution_enabled",
    "starless_enabled",
)


def _preset_dir() -> Path:
    import sys

    if sys.platform == "win32":
        base = Path.home() / "AppData" / "Local" / "AstroAI"
    else:
        base = Path.home() / ".config" / "astroai"
    return base / "presets"


def _safe_name(name: str) -> str:
    """Strip characters that are unsafe for filenames."""
    safe = "".join(c if c.isalnum() or c in " _-." else "_" for c in name)
    return safe[:_MAX_NAME_LEN].strip()


@dataclass
class PipelinePreset:
    name: str
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "config": self.config}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PipelinePreset:
        return PipelinePreset(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            config=dict(data.get("config", {})),
        )


class PresetManager:
    def __init__(self, preset_dir: Path | None = None) -> None:
        self._dir = preset_dir or _preset_dir()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def list_names(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        return sorted(
            p.stem for p in self._dir.iterdir() if p.suffix == _PRESET_EXTENSION
        )

    def save(self, preset: PipelinePreset) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{_safe_name(preset.name)}{_PRESET_EXTENSION}"
        path.write_text(json.dumps(preset.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, name: str) -> PipelinePreset:
        path = self._dir / f"{_safe_name(name)}{_PRESET_EXTENSION}"
        if not path.exists():
            raise FileNotFoundError(f"Preset nicht gefunden: {name!r}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return PipelinePreset.from_dict(data)

    def delete(self, name: str) -> None:
        path = self._dir / f"{_safe_name(name)}{_PRESET_EXTENSION}"
        if path.exists():
            path.unlink()

    def exists(self, name: str) -> bool:
        return (self._dir / f"{_safe_name(name)}{_PRESET_EXTENSION}").exists()

    # ------------------------------------------------------------------
    # Model integration
    # ------------------------------------------------------------------

    def capture_from_model(self, name: str, model: object, description: str = "") -> PipelinePreset:
        """Create a preset by reading the current values from *model*."""
        config: dict[str, Any] = {}
        for key in _MODEL_KEYS:
            try:
                config[key] = getattr(model, key)
            except AttributeError:
                pass
        return PipelinePreset(name=name, description=description, config=config)

    def apply_to_model(self, preset: PipelinePreset, model: object) -> None:
        """Write all config values from *preset* onto *model* (silently skips unknown keys)."""
        for key, value in preset.config.items():
            try:
                setattr(model, key, value)
            except (AttributeError, TypeError):
                pass
