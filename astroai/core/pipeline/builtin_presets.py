"""Built-in factory pipeline presets shipped with AstroAI Suite."""
from __future__ import annotations

from astroai.core.pipeline.presets import PipelinePreset, PresetManager

BUILTIN_PRESETS: list[PipelinePreset] = [
    PipelinePreset(
        name="Deepsky LRGB",
        description="Klassisches Deepsky-Preset für LRGB-Aufnahmen mit Sigma-Clipping und moderatem Denoising.",
        config={
            "stacking_method": "sigma_clip",
            "stacking_sigma_low": 3.0,
            "stacking_sigma_high": 3.0,
            "stretch_target_background": 0.20,
            "stretch_shadow_clipping_sigmas": 1.5,
            "stretch_linked_channels": True,
            "denoise_strength": 0.5,
            "denoise_tile_size": 128,
            "denoise_tile_overlap": 16,
            "background_removal_enabled": True,
            "background_removal_method": "grid",
            "background_removal_tile_size": 64,
            "background_removal_preserve_median": True,
            "drizzle_enabled": False,
            "drizzle_drop_size": 1.0,
            "drizzle_scale": 2.0,
            "drizzle_pixfrac": 1.0,
            "frame_selection_enabled": False,
            "frame_selection_min_score": 0.5,
            "comet_stack_enabled": False,
            "comet_tracking_mode": "stars",
            "synthetic_flat_enabled": False,
            "deconvolution_enabled": False,
            "starless_enabled": False,
        },
    ),
    PipelinePreset(
        name="Narrowband SHO",
        description="Hubble-Palette (SII-Ha-OIII) mit Sigma-Clipping und aggressivem Hintergrundabzug.",
        config={
            "stacking_method": "sigma_clip",
            "stacking_sigma_low": 2.5,
            "stacking_sigma_high": 2.5,
            "stretch_target_background": 0.15,
            "stretch_shadow_clipping_sigmas": 2.0,
            "stretch_linked_channels": False,
            "denoise_strength": 0.7,
            "denoise_tile_size": 128,
            "denoise_tile_overlap": 16,
            "background_removal_enabled": True,
            "background_removal_method": "grid",
            "background_removal_tile_size": 64,
            "background_removal_preserve_median": False,
            "drizzle_enabled": False,
            "drizzle_drop_size": 1.0,
            "drizzle_scale": 2.0,
            "drizzle_pixfrac": 1.0,
            "frame_selection_enabled": True,
            "frame_selection_min_score": 0.6,
            "comet_stack_enabled": False,
            "comet_tracking_mode": "stars",
            "synthetic_flat_enabled": False,
            "deconvolution_enabled": True,
            "starless_enabled": False,
        },
    ),
    PipelinePreset(
        name="Narrowband HOO",
        description="HOO-Palette (Ha-OIII-OIII) mit Median-Stacking und sanftem Stretch.",
        config={
            "stacking_method": "median",
            "stacking_sigma_low": 3.0,
            "stacking_sigma_high": 3.0,
            "stretch_target_background": 0.18,
            "stretch_shadow_clipping_sigmas": 1.5,
            "stretch_linked_channels": False,
            "denoise_strength": 0.6,
            "denoise_tile_size": 128,
            "denoise_tile_overlap": 16,
            "background_removal_enabled": True,
            "background_removal_method": "grid",
            "background_removal_tile_size": 64,
            "background_removal_preserve_median": True,
            "drizzle_enabled": False,
            "drizzle_drop_size": 1.0,
            "drizzle_scale": 2.0,
            "drizzle_pixfrac": 1.0,
            "frame_selection_enabled": True,
            "frame_selection_min_score": 0.55,
            "comet_stack_enabled": False,
            "comet_tracking_mode": "stars",
            "synthetic_flat_enabled": False,
            "deconvolution_enabled": False,
            "starless_enabled": False,
        },
    ),
    PipelinePreset(
        name="Planetarisch",
        description="Planetenaufnahmen: Lucky-Imaging-Selektion, Drizzle 2×, kein Hintergrundabzug.",
        config={
            "stacking_method": "mean",
            "stacking_sigma_low": 3.0,
            "stacking_sigma_high": 3.0,
            "stretch_target_background": 0.10,
            "stretch_shadow_clipping_sigmas": 1.0,
            "stretch_linked_channels": True,
            "denoise_strength": 0.3,
            "denoise_tile_size": 64,
            "denoise_tile_overlap": 8,
            "background_removal_enabled": False,
            "background_removal_method": "grid",
            "background_removal_tile_size": 64,
            "background_removal_preserve_median": True,
            "drizzle_enabled": True,
            "drizzle_drop_size": 0.5,
            "drizzle_scale": 2.0,
            "drizzle_pixfrac": 0.5,
            "frame_selection_enabled": True,
            "frame_selection_min_score": 0.7,
            "comet_stack_enabled": False,
            "comet_tracking_mode": "stars",
            "synthetic_flat_enabled": False,
            "deconvolution_enabled": True,
            "starless_enabled": False,
        },
    ),
]

BUILTIN_PRESET_NAMES: tuple[str, ...] = tuple(p.name for p in BUILTIN_PRESETS)


def install_builtin_presets(manager: PresetManager) -> int:
    """Save each builtin preset not yet present in *manager*. Returns installed count."""
    installed = 0
    for preset in BUILTIN_PRESETS:
        if not manager.exists(preset.name):
            manager.save(preset)
            installed += 1
    return installed
