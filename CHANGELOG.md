# Changelog — AstroAI Suite

All notable changes are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.8.0-alpha] — 2026-04-29

### Added
- **F-LocalContrastEnhancement: CLAHE-basierte lokale Kontrastverbesserung (VER-303)**
  - `astroai/processing/contrast/clahe.py` — `CLAHEConfig` (clip_limit, tile_size, n_bins, channel_mode: luminance/each/grayscale), `CLAHEEnhancer` (Kachel-Histogramm mit Clip+Umverteilung und bilinearer Interpolation, pure numpy), `CLAHEStep` (PipelineStep)
  - `astroai/ui/widgets/clahe_panel.py` — Qt-Panel: clip_limit (1.0–10.0), tile_size (8–512), channel_mode ComboBox, Reset-Button
  - `astroai/ui/models.py` — `clahe_*` Properties + `clahe_config_changed` Signal
  - `astroai/core/pipeline/builder.py` — `CLAHEStep`-Integration nach `MTFStep`
  - 61 Tests in `tests/unit/processing/test_clahe.py` + 17 in `tests/unit/ui/test_clahe_panel.py`
- **F-BackgroundNeutralization: Farbliche Hintergrundneutralisierung (VER-420)**
  - `astroai/processing/color/background_neutralizer.py` — `BackgroundNeutralizationConfig` (sample_mode auto/manual, target_background 0.0–0.3, roi, sample_percentile), `BackgroundNeutralizer` (Auto-Modus sampelt 2%-Perzentil je Kanal; manueller Modus nutzt ROI-Median; Shift auf Zielwert per Kanal, geclippt auf [0,1]), `BackgroundNeutralizationStep` (PipelineStep nach WhiteBalanceStep, vor AsinHStep; speichert Background-Schätzung in `context.metadata["background_neutralization_estimate"]`)
  - `astroai/ui/widgets/background_neutralization_panel.py` — Qt-Panel mit Modus-Combo (Auto/Manuell), Target-Spinner (0.0–0.3), ROI-Eingabe (nur im manuellen Modus aktiv), Reset-Button; Signal `bg_neutralization_changed(BackgroundNeutralizationConfig)`
  - `astroai/ui/models.py` — `bg_neutralization_enabled`, `bg_neutralization_sample_mode`, `bg_neutralization_target`, `bg_neutralization_roi` Properties + `background_neutralization_config_changed` Signal; Snapshot/Restore-Support; `bg_neutralization`-Step in DEFAULT_STEPS
  - `astroai/core/pipeline/builder.py` — `BackgroundNeutralizationStep`-Integration zwischen `WhiteBalanceStep` und `AsinHStep` bei `bg_neutralization_enabled`
  - 48 Tests in `tests/unit/processing/test_background_neutralization.py` (Config-Validation, Auto-Modus, Manueller-Modus, Estimate, PipelineStep)

- **F-MTFStretch: Midtone Transfer Function / Histogram Transformation Stretch (VER-419)**
  - `astroai/processing/stretch/mtf_stretch.py` — `MidtoneTransferConfig` (midpoint 0.001–0.499, shadows_clipping 0–0.1, highlights 0.98–1.0), `MidtoneTransferFunction` (MTF-Formel `f(x)=((m-1)x)/((2m-1)x-m)` mit x=0→0 / x=1→1 Edge-Cases, `apply()`, statisches `compute_midpoint_from_background()`, statisches `estimate_background()`), `MTFStep` (PipelineStep)
  - `astroai/ui/widgets/mtf_stretch_panel.py` — Qt-Panel mit 3 Schiebereglern (Mittelpunkt, Schatten-Clipping, Lichter), Auto-BTF-Button (emittiert `auto_btf_requested`, `set_auto_btf_midpoint(float)` Slot), Signal `mtf_changed(MidtoneTransferConfig)`
  - `astroai/ui/models.py` — `mtf_enabled`, `mtf_midpoint`, `mtf_shadows_clipping`, `mtf_highlights` Properties + `mtf_stretch_config_changed` Signal; Snapshot/Restore-Support
  - `astroai/core/pipeline/builder.py` — `MTFStep`-Integration nach `AsinHStep`
  - 62 Tests in `tests/unit/processing/test_mtf_stretch.py` + 17 in `tests/unit/ui/test_mtf_panel.py`

---

## [2.7.0-alpha] — 2026-04-29

### Added
- **F-SelectiveColorSaturation: HSV-basierte selektive Farbsättigung (VER-400)**
  - `astroai/processing/color/saturation.py` — `SaturationAdjustment` mit triangularer Hue-Gewichtung für 7 Farbbereiche (Reds, Oranges, Yellows, Greens, Cyans, Blues, Purples) + globalem Multiplikator
  - `SaturationConfig` Dataclass mit `is_identity()`-Check und `as_dict()` für alle Hue-Ranges
  - `SaturationStep` Pipeline-Integration (`PipelineStage.PROCESSING`)
  - `astroai/ui/widgets/saturation_panel.py` — Qt-Panel mit 8 Schiebereglern und `saturation_changed(SaturationConfig)`-Signal
  - `astroai/ui/models.py` — `saturation_*` Properties (enabled, global, 7 Hue-Ranges)
  - `astroai/core/pipeline/builder.py` — `SaturationStep`-Integration bei `saturation_enabled`
  - 40+ Tests in `tests/unit/processing/test_saturation.py` (Config, Identity, Global, PerRange, EdgeCases, HsvRoundtrip, Step)
- **F-NoiseEstimator: MAD-basierte Sky-Noise-Analyse mit Auto-Detect (VER-396)**
  - `astroai/core/noise_estimator.py` — `NoiseEstimator` mit iterativem Sigma-Clipping (MAD-basiert), SNR-Schätzung (dB), Noise-Level-Prozentsatz und automatischem Stärke-Mapping [0.2 → 1.0]
  - `NoiseEstimate` Frozen-Dataclass: `sky_sigma`, `snr_db`, `noise_level_pct`, `suggested_strength`
  - `astroai/ui/widgets/denoise_panel.py` — Auto-Detect-Button, `apply_estimate()` Slot, Noise-Info-Label (`σ=… SNR=…dB`)
  - Unterstützt Mono- (H×W) und Multi-Channel- (H×W×C) Bilder; BT.601 Luminanz-Konvertierung
  - 48 Tests in `tests/unit/core/test_noise_estimator.py` + 7 AutoDetect-Tests in `tests/unit/ui/test_denoise_panel.py`

- **F-AdaptiveDenoise: Automatische Rauschstärke via NoiseEstimator (VER-303)**
  - `astroai/core/pipeline/adaptive_denoise_step.py` — `AdaptiveDenoiseStep`: schätzt Rauschen per MAD-Sigma-Clipping, delegiert an `DenoiseStep` mit `suggested_strength`; speichert Schätzung in `context.metadata["estimated_denoise_strength"]`
  - `astroai/ui/widgets/denoise_panel.py` — Adaptive-Checkbox aktiviert `AdaptiveDenoiseStep` statt manuellem Stärke-Slider; `strength_spin` deaktiviert bei aktiviertem Adaptive-Modus
  - 34 Tests in `tests/unit/core/test_adaptive_denoise_step.py` + 10 in `tests/unit/ui/test_denoise_panel.py` + 4 in `tests/unit/core/test_pipeline_builder.py`
- **F-WhiteBalance: Manueller Per-Kanal R/G/B Weißabgleich (VER-303)**
  - `astroai/processing/color/white_balance.py` — `WhiteBalanceConfig` (Validierung, `is_identity`, `as_dict`), `WhiteBalanceAdjustment` (Kanal-Multiplikation, Graustufen-Passthrough, Clipping), `WhiteBalanceStep` (PipelineStep)
  - `astroai/ui/widgets/white_balance_panel.py` — Qt-Panel mit 3 Schiebereglern (0.1–5.0), Reset-Button, Signal `white_balance_changed(WhiteBalanceConfig)`
  - `astroai/ui/models.py` — `white_balance_enabled` + `wb_red/green/blue` Properties
  - `astroai/core/pipeline/builder.py` — `WhiteBalanceStep`-Integration nach `SaturationStep`
  - 43 Tests in `tests/unit/processing/test_white_balance.py` + 17 in `tests/unit/ui/test_white_balance_panel.py`
- **F-AsinHStretch: Arcsinh-Stretch (Sternphotometrie-erhaltend) (VER-303)**
  - `astroai/processing/stretch/asinh_stretcher.py` — `AsinHConfig` (stretch_factor, black_point, linked_channels), `AsinHStretcher` (arcsinh(β·x)/arcsinh(β), β<1e-9 → linearer Passthrough, linked/unlinked RGB), `AsinHStep` (PipelineStep)
  - `astroai/ui/widgets/asinh_stretch_panel.py` — Qt-Panel mit stretch_factor (0.001–1000), black_point (0–0.5), linked Checkbox, Signal `asinh_stretch_changed(AsinHConfig)`
  - `astroai/ui/models.py` — `asinh_enabled`, `asinh_stretch_factor`, `asinh_black_point`, `asinh_linked` Properties
  - `astroai/core/pipeline/builder.py` — `AsinHStep`-Integration nach `WhiteBalanceStep`
  - 49 Tests in `tests/unit/processing/test_asinh_stretcher.py` + 18 in `tests/unit/ui/test_asinh_stretch_panel.py`

### Stats
- **+270 neue Tests** (+67 AsinHStretch auf ~203 vorherige)
- **8 neue Dateien**, ~2759 Insertionen

---

## [2.6.0-alpha] — 2026-04-29

### Added
- **F-OnnxRegistry: Zentrales ONNX-Modell-Registry mit Lazy Loading (VER-363)**
  - `astroai/core/onnx_registry.py` — thread-safe Registry, Lazy-Loading, automatische Geräte-Auswahl (CUDA/MPS/CPU), TTL-basiertes Cache-Eviction
  - Integration in `FrameScorer`, `Denoiser`, `StarManager` für einheitliche Modell-Verwaltung
  - 513 neue Tests in `tests/unit/core/test_onnx_registry.py`
- **F-StarNetTiling: StarNet++ Kachel-Verarbeitung für große Bilder (VER-364)**
  - `astroai/processing/stars/star_manager.py` — überlappende Tile-Verarbeitung mit konfigurierbarer Tile-Größe und Overlap, Blend-Masken für nahtlose Übergänge
  - Unterstützung für beliebig große Bilder ohne GPU-OOM; fallback auf Direktverarbeitung bei kleinen Bildern
  - 101 neue Tests in `tests/unit/processing/test_stars.py`
- **F-CatalogCache: SQLite-gestützter Stern-Katalog-Cache mit TTL (VER-365)**
  - `astroai/core/catalog_cache.py` — persistenter SQLite-Cache für GAIA/2MASS-Katalogabfragen, TTL-Eviction, Geolookup-Deduplizierung
  - Reduziert API-Requests bei wiederholten Abfragen in gleicher Himmelsgegend
  - 222 neue Tests in `tests/unit/core/test_catalog_cache.py`
- **F-LivePreview: Debounced Live-Preview-Engine mit Diff-Modus (VER-366)**
  - `astroai/ui/preview_worker.py` — QThread-basierter Preview-Worker, konfigurierbares Debounce-Delay, Diff-Modus zeigt Vorher/Nachher-Vergleich
  - Integration in StretchPanel, DenoisePanel, BackgroundRemovalPanel, CurvesPanel
  - 559 neue Tests in `tests/unit/ui/test_preview_worker.py`
- **F-ProcessingHistory: Undo/Redo für Processing-Schritte**
  - `astroai/core/processing_history.py` — Stack-basierte Undo/Redo-History mit konfigurierbarer Tiefe (Standard: 20)
  - Integriert in MainWindow (Ctrl+Z / Ctrl+Shift+Z)
  - 191 neue Tests in `tests/unit/core/test_processing_history.py`
- **F-Sharpening: Schärfungs-Modul mit UnsharpMask + Pipeline-Integration**
  - `astroai/processing/sharpening/unsharp_mask.py` — Gaussian-basiertes UnsharpMask, radius/amount/threshold-Parameter mit Validierung [0, 0.5]
  - `astroai/processing/sharpening/pipeline_step.py` — `SharpeningStep` für Pipeline-Integration
  - `astroai/ui/widgets/sharpening_panel.py` — Qt-Widget mit Slidern und Live-Preview-Signalen
  - PipelineBuilder + PipelineModel Sharpening-Support
  - 39 Tests in `tests/unit/processing/test_sharpening.py` + `tests/unit/ui/test_sharpening_panel.py`

### Fixed
- **fix(test-suite): Qt teardown crash bei ~96% im vollen 3348-Test-Run (VER-361)**
  - `conftest.py` session-scoped QApplication-Fixture verhindert vorzeitigen Teardown
  - Alle 3348 Tests laufen jetzt sauber durch

### Stats
- **+168 neue Tests** (39 Sharpening + 129 Core/UI)
- **Gesamt-Testanzahl:** ~3348+
- **9 neue Dateien**, 3723+ Insertionen

---

## [2.4.0-alpha] — 2026-04-28

### Added
- **F-CoverageImprovement: LiveHistogramView Test-Abdeckung von 79% auf 100%**
  - `_HistogramCanvas`-Fixture in `test_live_histogram.py` um `w.show()` ergänzt — `repaint()` triggert jetzt tatsächlich `paintEvent`; paintEvent-Zeilen sowie `_draw_channel` vollständig abgedeckt
  - Gesamtabdeckung aller Tests verbessert von 96% auf 97%
- **F-CoverageImprovement: SplitCompareView Test-Abdeckung von 77% auf 93%**
  - 10 neue Tests in `TestSplitCompareViewZoom` und `TestSplitCompareViewMouseAndKey`
  - WheelEvent-Tests (Zoom in/out), Maus-Drag-Tests (press/release pan-drag), Tasten-Tests (Links/Rechts/Hoch/Runter-Pfeile), Tile-Cache-Hit-Test
- **F-CoverageImprovement: CurvesPanel Test-Abdeckung von 43% auf 93%**
  - Neues `tests/unit/ui/test_curves_panel.py` mit 23 Tests für `CurveEditor` und `CurvesPanel`
  - `CurveEditor`-Tests: `get_points`, `set_points`, `reset`, `_plot_rect`, `_to_widget/_from_widget`, `_nearest_point_idx`, Linksklick-Hinzufügen, Rechtsklick-Entfernen (Mittel- und Endpunkt), `mouseReleaseEvent`, Max-Punkte-Grenze
  - `CurvesPanel`-Tests: Reset aller Kanäle, Channel-Wechsel, `_on_points_changed` für alle 4 Kanäle (RGB, R, G, B)
  - Deprecation-Warnings durch Verwendung von `qtbot.mouseClick` statt manueller `QMouseEvent`-Konstruktion beseitigt
- **F-ImportFolderAction: Ordner-Import (rekursiv) für Light-Frames**
  - Frames importieren > "Ordner importieren..." (Ctrl+Shift+F) — `QFileDialog.getExistingDirectory`, dann `Path.rglob("*")` nach `.fits/.fit/.fts`; delegiert an `_add_light_frames()` (DRY)
  - Zeigt Status "Keine FITS-Dateien gefunden in: <Ordnername>" wenn Ordner leer ist
  - `ShortcutsDialog` um "Ctrl+Shift+F" ergänzt
  - 4 neue Tests in `TestMainWindowImportFrames` (cancelled, finds 2 files, recursive subdir, empty folder message)
- **F-CalibStatusLabel: Kalibrierungs-Status in der Statusleiste**
  - Permanente `QLabel` `_calib_status_label` in der Statusleiste zeigt Anzahl konfigurierter Dark/Flat/Bias-Frames (Format: "Kalib.: 5D / 3F / 0B" bzw. "Kalib.: —" wenn leer)
  - `_refresh_calib_status()` Hilfsmethode liest `project.calibration.{dark,flat,bias}_frames` und aktualisiert das Label
  - Wird automatisch aufgerufen nach: Dark/Flat/Bias-Import, Auto-Match-Kalibrierung, Projekt-Laden
  - Label steht links von "Zoom: …" und "Lizenz"-Badge
  - 6 neue Tests in `TestCalibStatusLabel` in `tests/unit/ui/test_main_window.py`
- **F-FrameNotesField: Freitext-Notizen auf einzelnen Frames**
  - `FrameEntry` erhält neues optionales Feld `notes: str = ""` (rückwärtskompatibel — `from_dict` füllt fehlende Schlüssel mit Default)
  - `FrameListPanel`: Dateiname-Zelle zeigt Präfix `"* "` wenn Notiz vorhanden; Tooltip enthält Notiz ("Notiz: …")
  - Kontextmenü erhält "Notiz bearbeiten…" (aktiv nur bei Einzelauswahl) → `QInputDialog.getText` → speichert gekürzten Text in `entry.notes`; aktualisiert Tabelle sofort
  - Abbrechen des Dialogs lässt `entry.notes` unverändert; leere Eingabe löscht die Notiz
  - `_edit_notes(row)` schützt gegen out-of-range-Index ohne Exception
  - 6 neue Tests in `TestFrameListPanelNotes`; 3 neue Tests in `TestAstroProject` für Notes-Roundtrip und Backward-Compat
- **F-FramePreviewOnClick: Frame-Vorschau aus Kontextmenü**
  - `FrameListPanel` erhält neues Signal `preview_requested = Signal(str)` (absoluter Dateipfad)
  - Kontextmenü enthält neuen Eintrag "Vorschau anzeigen" (erster Eintrag, mit Trennlinie) — aktiv nur wenn genau 1 Zeile markiert ist; mehrfachauswahl deaktiviert die Aktion
  - Auslösen emittiert `preview_requested` mit `entry.path` des gewählten Frames
  - `MainWindow._connect_signals()` verbindet `preview_requested` → `_on_frame_preview_requested(path)`
  - `_on_frame_preview_requested` delegiert an `self._file_loader.load(Path(path))` — lädt das FITS-File direkt in den Viewer
  - 4 neue Tests in `TestFrameListPanelPreviewSignal` in `tests/unit/ui/test_widgets.py`
- **F-SavePreviewImage: Vorschau-Bild als PNG/JPEG/TIFF speichern**
  - Bearbeiten-Menü erhält "Vorschau-Bild speichern..." (Ctrl+Shift+P); aktiviert sobald ein Bild geladen ist
  - `_on_save_preview_image()` Slot öffnet `QFileDialog.getSaveFileName` mit Filter PNG/JPEG/TIFF; speichert via `QImage.save()`; zeigt Fehler-QMessageBox bei Schreibfehler
  - `render_full_qimage()` aus `ImageViewer` wiederverwendet (DRY) — gleiche min/max-Normalisierung wie Clipboard-Aktion
  - Shortcut "Ctrl+Shift+P" in `ShortcutsDialog` ergänzt
  - 6 neue Tests in `TestSavePreviewImage` in `tests/unit/ui/test_main_window.py`
- **F-ShortcutsDialog: Tastaturkürzel-Referenz**
  - `ShortcutsDialog(parent)` in `astroai/ui/widgets/shortcuts_dialog.py` — modaler QDialog mit gruppierten Shortcut-Tabellen (5 Abschnitte: Projekt, Frames & Kalibrierung, Ansicht, Pipeline, Hilfe)
  - Jeder Abschnitt als `QGroupBox` mit read-only `QTableWidget` (Funktion / Kürzel), Alternating Row Colors, keine Bearbeitung, kein Fokus
  - `WA_DeleteOnClose` gesetzt; Schließen-Button via `QDialogButtonBox`
  - Hilfe-Menü erhält "Tastaturkürzel..." (Ctrl+?) → `_on_show_shortcuts` Slot öffnet Dialog (nicht-modal)
  - `_SECTIONS` als öffentliche Konstante für Testbarkeit
  - 13 neue Tests in `tests/unit/ui/test_shortcuts_dialog.py`
- **F-DragDropFrameImport: Drag & Drop FITS Import in Frame-Liste**
  - `FrameListPanel` akzeptiert jetzt Drag-&-Drop von FITS-Dateien (`.fits`, `.fit`, `.fts`) direkt aus dem Datei-Manager
  - `files_dropped = Signal(list)` emittiert absolute Dateipfade aller gedropten FITS-Dateien
  - `dragEnterEvent` akzeptiert nur wenn mindestens eine URL mit FITS-Endung vorhanden; verwirft alles andere (PNG, TXT, etc.)
  - `dropEvent` filtert URLs auf FITS-Endungen, emittiert `files_dropped` mit validen Pfaden, ignoriert leere Listen
  - `_FITS_SUFFIXES: frozenset[str]` als Klassen-Konstante für konsistente Endungen-Prüfung
  - `MainWindow` verbindet `files_dropped` mit `_on_frames_dropped(paths)` Slot — delegiert an `_add_light_frames()`
  - `_add_light_frames()` extrahiert die DRY-Logik aus `_on_import_lights`: Duplikat-Check, `_enrich_fits_entry`, `refresh()`, Status-Bar-Meldung
  - 7 neue Tests in `TestFrameListPanelDragDrop` in `tests/unit/ui/test_widgets.py`
- **F-SmartCalibUI: Smart-Kalibrierungs-Scanner-Panel (FR-2.3)**
  - `SmartCalibPanel` in `astroai/ui/widgets/smart_calib_panel.py` — Dock-Widget mit Verzeichnis-Picker, optionalem Rekursiv-Scan, Scan-Button und Ergebnistabelle (Typ/Anzahl für Dark/Flat/Bias/Light/Unbekannt)
  - Auto-Match-Button ruft `build_calibration_library` → `batch_match` → `suggest_calibration_config` auf und schreibt das Ergebnis direkt in `AstroProject.calibration`; markiert Projekt als dirty via `touch()`
  - Coverage-Anzeige zeigt `dark_coverage` und `flat_coverage` als Prozentwerte nach erfolgreichem Match
  - Fehlermeldungen bei ungültigem Verzeichnis, fehlendem Projekt oder fehlenden Light-Frames
  - `MainWindow._setup_docks()` integriert das Panel als "Smart-Kalibrierung"-Dock im linken Bereich; `SmartCalibPanel(lambda: self._project)` liefert immer das aktuelle Projekt
  - Return-Typen von `build_calibration_library` und `suggest_calibration_config` präzisiert (von `object` auf konkrete Typen via `TYPE_CHECKING`); alle mypy-Errors bereinigt
  - 26 neue Tests in `tests/unit/ui/test_smart_calib_panel.py`
- **F-BuiltinPresets: Eingebaute Pipeline-Preset-Bibliothek**
  - `astroai/core/pipeline/builtin_presets.py` liefert 4 fertige Presets: `Deepsky LRGB` (Sigma-Clipping, verknüpfte Kanäle, Hintergrundabzug), `Narrowband SHO` (Hubble-Palette, entkoppelte Kanäle, Dekonvolution), `Narrowband HOO` (Median-Stacking, sanfter Stretch) und `Planetarisch` (Lucky-Imaging-Selektion, Drizzle 2×, kein Hintergrundabzug)
  - `BUILTIN_PRESETS: list[PipelinePreset]` und `BUILTIN_PRESET_NAMES: tuple[str, ...]` als Konstanten exportiert; `install_builtin_presets(manager) -> int` installiert fehlende Presets (idempotent, gibt Anzahl installierter Presets zurück)
  - `MainWindow._setup_presets()` ruft `install_builtin_presets` beim Start auf (vor `_setup_menus()`), sodass die 4 Presets beim ersten App-Start automatisch angelegt werden
  - `_rebuild_preset_menu()` trennt eingebaute von Benutzer-Presets durch eine `QMenu`-Trennlinie; eingebaute Presets erscheinen oben
  - `BUILTIN_PRESET_NAMES` und `install_builtin_presets` aus `astroai.core.pipeline` exportiert
  - 19 neue Tests in `tests/unit/core/test_builtin_presets.py`: Preset-Definitionen, Konfigurationswerte, `install_builtin_presets`-Idempotenz und Verzeichniserstellung
- **F-ProjectSummary: Project Statistics Overview**
  - `compute_summary(project) -> ProjectSummary` in `astroai/project/summary.py` — aggregates: total/selected frame counts, total exposure (s), exposure groups, quality score stats (mean/min/max), temperature range
  - `ProjectSummary.total_exposure_hms` property formats seconds to human-readable string (e.g. `1h 23m 45s`)
  - `ExposureGroup` dataclass: `exposure_s`, `count`; groups sorted by exposure value
  - Hilfe menu gains "Projektübersicht..." (Ctrl+I) — shows `QMessageBox.information` with formatted project stats (HTML bold labels, exposure groups, quality percentages, temperature range)
  - `ProjectSummary`, `ExposureGroup`, `compute_summary` exported from `astroai.project`
  - 18 new tests in `tests/unit/project/test_summary.py`
- **F-PresetUI: Pipeline Preset Menu Integration**
  - Pipeline menu gains "Preset speichern..." (opens `QInputDialog` for name → `capture_from_model` + `save`) and "Preset laden" submenu (auto-populated from saved presets via `_rebuild_preset_menu()`)
  - `_on_save_preset()` slot captures current `PipelineModel` state and writes to `~/.config/astroai/presets/`; status bar confirms name
  - `_on_load_preset(name)` slot loads and applies preset to `PipelineModel`; shows `QMessageBox.warning` if file no longer exists and rebuilds menu
  - `PresetManager` instance stored on `MainWindow` as `_preset_manager`
- **F-ProjectValidation: Pre-Run Project Validator**
  - `validate_project(project) -> ValidationResult` in `astroai/project/validator.py` — runs structured checks with typed `ValidationIssue` (level: `"error"` / `"warning"` / `"info"`, machine-readable `code`)
  - Checks: no frames (`NO_FRAMES`), all deselected (`NO_SELECTED_FRAMES`), missing light files (`MISSING_LIGHT_FILES`), missing dark/flat/bias calibration files (`MISSING_{DARK/FLAT/BIAS}_FILES`), output directory missing (`OUTPUT_DIR_MISSING`)
  - `ValidationResult` properties: `has_errors`, `has_warnings`, `errors`, `warnings`, `summary()`
  - `_validate_before_run()` helper in `MainWindow` — calls `validate_project` before `_on_run_full_pipeline`; shows `QMessageBox.warning` listing all error messages and aborts if errors found
  - `ValidationIssue`, `ValidationResult`, `validate_project` exported from `astroai.project`
  - 17 new tests in `tests/unit/project/test_validator.py`
- **F-PipelinePreset: Named Pipeline Configuration Presets**
  - `PipelinePreset` dataclass in `astroai/core/pipeline/presets.py`: `name`, `description`, `config` dict; `to_dict()` / `from_dict()` for JSON persistence
  - `PresetManager` class: `save()`, `load()`, `delete()`, `exists()`, `list_names()`; stores presets as JSON files in `~/.config/astroai/presets/` (Win: `%LOCALAPPDATA%/AstroAI/presets/`)
  - `capture_from_model(name, model)` — snapshots 24 `PipelineModel` properties (stacking, stretch, denoise, background removal, drizzle, frame selection, comet, synflat, deconvolution, starless) into a `PipelinePreset`
  - `apply_to_model(preset, model)` — applies preset config to any model via `setattr`; silently skips unknown keys for forward compatibility
  - `_safe_name()` strips filesystem-unsafe characters; names capped at 64 chars
  - Both `PipelinePreset` and `PresetManager` exported from `astroai.core.pipeline`
  - 23 new tests in `tests/unit/core/test_pipeline_presets.py` covering persistence, round-trip, model integration, sorting, and platform paths
- **F-FrameExportStats: Export Frame Statistics to CSV**
  - `export_frame_stats(frames, dest) -> int` in `astroai/core/io/frame_stats_export.py` — writes a CSV with columns `filename`, `path`, `exposure_s`, `gain_iso`, `temperature_c`, `quality_score`, `selected`; silently skips non-`FrameEntry` objects; creates parent directory if missing; returns row count
  - File menu action "Frames-Statistik exportieren..." (Ctrl+Shift+E) opens a save dialog and calls `export_frame_stats` on the current project's `input_frames`; status bar shows exported row count and filename
  - 14 new tests in `tests/unit/core/test_frame_stats_export.py`
- **F-CalibScan: Calibration Frame Directory Scanner (FR-2.3)**
  - `scan_directory(directory, *, recursive=False) -> list[ScannedFrame]` — scans a directory for FITS files (`.fits`, `.fit`, `.fts`) and classifies each by `IMAGETYP` header keyword (also checks `FRAME` fallback); silently skips unreadable files and non-FITS files
  - `_classify_imagetyp(imagetyp)` — normalizes and maps raw keyword values to `"dark"`, `"flat"`, `"bias"`, `"light"`, or `"unknown"` (case-insensitive, handles common variants: `Dark Frame`, `Flat Field`, `Flatfield`, `BiasFrame`, `offset`, `zero`, etc.)
  - `partition_by_type(frames) -> dict[str, list[ScannedFrame]]` — groups scanned frames by type for downstream use
  - `build_calibration_library(frames, load_data=False) -> CalibrationLibrary` — converts pre-scanned frames directly to a `CalibrationLibrary`; light/unknown frames are excluded; `load_data=True` eagerly loads pixel arrays
  - `ScannedFrame` dataclass: `path`, `frame_type`, `metadata`; exported from `astroai.core.calibration`
  - 27 new tests in `tests/unit/core/test_calib_scanner.py` covering classification, recursive scanning, corrupt file skipping, mixed-type directories, and `build_calibration_library`
- **F-CalibMatchBatch: Smart Calibration Batch Matching (FR-2.3)**
  - `batch_match(lights, library) -> BatchMatchResult` — matches every light frame to its optimal dark/flat from a `CalibrationLibrary` in one call; bias slot reserved as `None` (future extension)
  - `BatchMatchResult` dataclass with `matches: list[FrameMatchResult]`, `coverage`, `dark_coverage`, `flat_coverage` properties (all [0.0, 1.0])
  - `FrameMatchResult` dataclass — per-frame result linking `light_path`, `dark`, `flat`, `bias`
  - `suggest_calibration_config(result) -> CalibrationConfig` — collapses `BatchMatchResult` into a deduplicated `CalibrationConfig` (ready to assign to `AstroProject.calibration`); preserves insertion order; filters out `None` matches
  - `batch_match`, `BatchMatchResult`, `FrameMatchResult`, `suggest_calibration_config` exported from `astroai.core.calibration.__init__`
  - 12 new tests split across `TestBatchMatch` (8) and `TestSuggestCalibrationConfig` (4) in `test_calibration.py`
- **F-CopyImageToClipboard: Copy Displayed Image to Clipboard**
  - `ImageViewer.render_full_qimage()` renders the full `_raw_data` array to a `QImage` (Grayscale8) with min/max normalization; returns `None` when no image is loaded
  - "Bearbeiten" menu added to menu bar (between Datei and Ansicht) with "Bild kopieren" action (Ctrl+C); enabled when an image is loaded
  - `_on_copy_image()` slot calls `render_full_qimage()` and pushes the result to `QApplication.clipboard().setImage()`; updates status bar with confirmation message
  - 4 new `ImageViewer` tests; 5 new `TestCopyImageToClipboard` tests
- **F-FrameQualityThreshold: Quality Threshold Auto-Selection**
  - `QDoubleSpinBox` (0–100 %, step 5) + "Anwenden" button added to `FrameListPanel` header area for threshold-based frame rejection
  - `apply_quality_threshold(min_pct)` public method: sets `entry.selected = score >= threshold` for all scored frames; unscored frames are untouched; emits `selection_changed` only for rows that actually changed
  - `_on_apply_threshold()` slot reads spinbox value and delegates to `apply_quality_threshold()`
  - 9 new unit tests in `TestFrameListPanel`
- **F-FrameFilterBar: Frame List Text Filter**
  - `QLineEdit` with placeholder "Frames filtern…" and built-in clear button added above the frame table in `FrameListPanel`
  - `_on_filter_changed(text)` updates `_filter_text` and calls `_apply_filter()`
  - `_apply_filter()` calls `QTableWidget.setRowHidden(row, not visible)` for each row — hides non-matching entries without changing row indices or disrupting signal emission
  - Filter is case-insensitive substring match on filename (basename only)
  - Filter state persists across `refresh()` calls — re-applied automatically after `_repopulate_table()`
  - 7 new unit tests in `TestFrameListPanel`
- **F-AutoStretch: One-Click Preview Auto-Stretch**
  - `_apply_auto_stretch(data)` module-level function: clips to [p0.5, p99.5] percentiles, normalizes to [0, 1] float32; no-op when hi == lo; does not modify source array
  - View menu action "Auto-Stretch Vorschau" (Ctrl+Shift+A, checkable); enabled when an image is loaded
  - `_display_image_data(img)` helper centralizes viewer/histogram/stats push — applies stretch when `_auto_stretch=True`; used by `_on_image_loaded`, `_on_pipeline_finished`, and the toggle slot
  - `_on_toggle_auto_stretch(checked)`: toggles flag and immediately re-renders current image with/without stretch
  - 9 new tests in `TestAutoStretch`
- **F-ProjectDirty: Unsaved Changes Tracking**
  - `AstroProject.is_dirty` property backed by `_dirty` instance attribute set in `__post_init__`; does NOT appear in `to_dict()` / `asdict()` so it never bleeds into serialized files
  - `AstroProject.touch()` now also sets `_dirty = True` alongside updating `modified_at`
  - `AstroProject.mark_clean()` clears dirty flag; called after successful `_save_project()`
  - `MainWindow._update_title()` calls `setWindowModified(is_dirty)` with `[*]` placeholder in title — Qt auto-shows `*` when modified
  - Frame selection changes, frame removes, and session note edits now all call `self._project.touch()` + `_update_title()`
  - `closeEvent` prompts user (Save/Discard/Cancel) only when `event.spontaneous()` — programmatic closes (e.g. test teardown) skip the dialog
  - `_maybe_discard_changes()` helper reused by `_on_new_project`, `_on_open_project`, and `closeEvent`
  - 7 new `TestAstroProjectDirty` tests in `test_project_file.py`; 10 new `TestMainWindowProjectDirty` tests in `test_main_window.py`
- **F-ZoomStatusBar: Permanent Zoom Indicator**
  - Zoom level shown as a permanent `QLabel` widget in the status bar right side (e.g. "Zoom: 150%") — always visible, never overwritten by transient status messages
  - `SplitCompareView.zoom_changed = Signal(float)` added; emitted on wheel zoom, keyboard +/−, and `fit_to_view()`; `zoom_level` property added
  - `MainWindow._on_zoom_changed` now updates `_zoom_label` directly instead of embedding in the message string
  - Both `ImageViewer` and `SplitCompareView` zoom events feed the same label — label reflects whichever view is active
  - `_on_pixel_hovered` simplified: no longer needs to split `|`-separated zoom text out of the status message
  - 7 new tests in `TestSplitCompareViewZoom`; `test_on_zoom_changed` updated to check the permanent label
- **F-FrameSort: Sortable Frame List Columns**
  - `FrameListPanel` columns (Dateiname, Belichtung, Qualität, Ausgewählt) are now sortable by clicking the header — first click ascending, second click descending, switching columns resets to ascending
  - `_on_header_clicked(col)` sorts `_entries` list in-place so all existing signal/row-index logic remains correct without any mapping overhead
  - `refresh()` re-applies the active sort when called with new entries (e.g. after quality scoring updates)
  - Visual sort indicator shown on column header via `setSortIndicatorShown(True)` / `setSortIndicator()`
  - `None` values for exposure/quality sort before any real value (treated as −1) to keep unscored/unexposed frames grouped
  - 10 new unit tests in `TestFrameListPanel`
- **F-AnnotationConfig: Annotation Layer Persistence**
  - `AnnotationConfig` dataclass added to `project_file.py` (`show_dso`, `show_stars`, `show_boundaries`, `show_grid`) with defaults matching prior panel behavior
  - 4 new `PipelineModel` properties + `annotation_config_changed` signal; annotation panel toggles write to model, model sync restores overlay state on project load
  - `_sync_annotation_from_model()` in `MainWindow` applies model state to both panel checkboxes and annotation overlay; called on project load and `annotation_config_changed`
  - Full bidirectional `.astroai` project sync — annotation visibility settings now survive project save/reload
- **F-FullPipelineRun: Multi-Frame Stack & Process**
  - `LoadFramesStep` (`astroai/core/pipeline/load_frames_step.py`) — `CALIBRATION`-stage step that loads image files (FITS/TIFF/PNG) from a path list into `context.images`; per-frame progress; stores `loaded_frame_paths` metadata
  - `PipelineBuilder.build_full_pipeline(model, frame_paths)` — builds end-to-end pipeline: `LoadFramesStep` → calibration (frame selection, synthetic flat) → registration → stacking → processing
  - "Pipeline → Stack && Process" (Ctrl+Shift+R) menu action in `MainWindow`; enabled when project has light frames; runs full pipeline on all `project.input_frames`; result displayed after completion
  - `_on_import_lights` now enables the Stack & Process action immediately after import
  - **Bugfix (type safety)**: `PipelineBuilder` now resolves `model.comet_tracking_mode` via `_TRACKING_MODE` lookup dict (removes `# type: ignore[arg-type]`); uses `TrackingMode` literal type
- **F-RegistrationStep + F-StackingStep: Full Registration/Stacking Pipeline Steps**
  - `RegistrationStep` (`astroai/engine/registration/pipeline_step.py`) — `REGISTRATION`-stage pipeline step that aligns all frames to a reference using phase correlation via `FrameAligner`; configurable `upsample_factor` and `reference_frame_index`; stores metadata `registration_reference_index` + `registration_frames_aligned`
  - `StackingStep` (`astroai/engine/stacking/pipeline_step.py`) — `STACKING`-stage pipeline step that combines context frames via `FrameStacker`; supports `mean`, `median`, and `sigma_clip` (configurable `sigma_low`/`sigma_high`); stores metadata `stacking_method` + `stacking_frame_count`
  - `PipelineBuilder.build_stacking_pipeline(model)` — combines `RegistrationStep` + `StackingStep` into a single pipeline reading registration/stacking config from `PipelineModel`
  - `PipelineBuilder.build_registration_pipeline(model)` — standalone registration-only pipeline
  - `RegistrationPanel` + `StackingPanel` — UI dock widgets for upsample-factor/reference-frame and stacking-method/sigma controls; bidirectional sync with `PipelineModel`
  - 5 new `PipelineModel` properties (`registration_upsample_factor`, `registration_reference_frame_index`, `stacking_method`, `stacking_sigma_low`, `stacking_sigma_high`) + signals `registration_config_changed` / `stacking_config_changed`; full bidirectional `.astroai` sync (`RegistrationConfig` + `StackingConfig` existed since v1 but were never synced to the model)
  - `MainWindow` docks "Registrierung" + "Stacking" wired; `_sync_model_to_project()` / `_sync_project_to_model()` extended for both configs
  - **Bugfix**: Removed `PipelineBuilder` re-export from `astroai.core.pipeline.__init__` to eliminate circular import (`background.pipeline_step` → `core.pipeline` package → `builder` → `background.pipeline_step`)
- **F-PipelineRunner: Background Pipeline Execution**
  - `PipelineWorker` (`astroai/core/pipeline/runner.py`) — QThread-based executor that runs a `Pipeline` in a background thread; signals: `finished(PipelineContext)`, `error(str)`, `progress(float, str)`
  - `MainWindow` wired: "Pipeline → Ausführen" menu action (Ctrl+R), enabled when image loaded; builds processing pipeline via `PipelineBuilder`, creates `PipelineContext`, starts `PipelineWorker`, displays result image on completion
  - `MainWindow` now stores `_current_image` on each file-load so the run action has input data
  - Exported from `astroai.core.pipeline` package
- **F-PipelineBuilder: Pipeline Configuration Bridge**
  - `PipelineBuilder` — factory that reads `PipelineModel` state and instantiates a fully-configured `Pipeline` with concrete step objects
  - `build_calibration_pipeline(model)`: builds pre-stacking pipeline (optional `FrameSelectionStep` + `SyntheticFlatStep`)
  - `build_processing_pipeline(model)`: builds post-stacking processing pipeline in correct execution order — drizzle, mosaic, channel combine, comet stacking, background removal, stretch, color calibration, denoise, deconvolution, starless — all with model-derived parameters
  - `build_export_step(model, output_dir)`: creates a correctly configured `ExportStep`
  - Closes the previously-noted gap where all UI config was stored but never wired to actual step execution
  - Exported from `astroai.core.pipeline` package
- **F-StarProcessing: Star Detection + Reduction UI + Project Persistence**
  - `StarProcessingPanel` — UI dock with star size reduction toggle (factor 0–100%), and advanced star detection controls (σ-threshold, min/max area, mask dilation)
  - `star_processing_config_changed` signal + 6 model properties (`star_reduce_enabled`, `star_reduce_factor`, `star_detection_sigma`, `star_min_area`, `star_max_area`, `star_mask_dilation`) with full bidirectional `.astroai` sync (`StarProcessingConfig` existed but was never persisted)
  - `StarRemovalStep` extended: when `reduce_enabled=True`, calls `StarManager.reduce_stars()` (shrinks stars by `factor`) instead of full removal — activates previously dead `reduce_stars()` code path
- **F-DenoisePanel: Denoising UI + Project Persistence**
  - `DenoisePanel` — UI dock with strength, tile-size, and tile-overlap controls for NAFNet AI denoising
  - `denoise_config_changed` signal + model properties (`denoise_strength`, `denoise_tile_size`, `denoise_tile_overlap`) wired to full bidirectional `.astroai` sync (config dataclass existed since v1.x but was never persisted)
- **F-StretchPanel: Histogram Stretch UI + Project Persistence**
  - `StretchPanel` — UI dock with target-background, shadow-sigma, and linked-channels controls
  - `stretch_config_changed` signal + model properties (`stretch_target_background`, `stretch_shadow_clipping_sigmas`, `stretch_linked_channels`) wired to full bidirectional `.astroai` sync (same latent persistence bug fixed)
- **F-BackgroundRemoval: Background Gradient Removal UI Integration**
  - `BackgroundRemovalPanel` — UI dock with enable checkbox, method (RBF/Polynomial) selector, tile-size control, and preserve-median toggle
  - `BackgroundRemovalConfig` — project-file dataclass for `.astroai` persistence; bidirectional sync in `MainWindow`
  - Wires the existing `BackgroundRemovalStep` / `GradientRemover` engine into the full PipelineModel + workflow graph
- **F-FrameSelect: AI Frame Selection / Sub-Frame Rejection**
  - `FrameSelectionStep` — `CALIBRATION`-stage pipeline step; scores each frame via `FrameScorer` (HFR, roundness, cloud), filters below `min_score`, safety-net keeps at least `1 - max_rejected_fraction` of frames
  - `FrameSelectionPanel` — UI dock with enable checkbox, min-score and max-rejected-fraction controls
  - `FrameSelectionConfig` — project-file dataclass for `.astroai` persistence; bidirectional sync in `MainWindow`
- **F-SynFlat: Synthetic Flat Frame Generation**
  - `SyntheticFlatGenerator` — models illumination vignetting via tile-sampled RBF/polynomial surface fitting, median-combines multiple frames, applies Gaussian smoothing, normalises to peak = 1.0
  - `SyntheticFlatStep` — `CALIBRATION`-stage pipeline step that generates and applies the flat
  - `SyntheticFlatPanel` — UI dock widget with enable checkbox, tile-size and smoothing-sigma controls
  - `SyntheticFlatConfig` + `CometStackConfig` — project-file dataclasses for `.astroai` persistence
- **Full bidirectional project persistence** — `MainWindow._sync_model_to_project()` / `_sync_project_to_model()` synchronises all optional feature configs (drizzle, mosaic, channel_combine, color_calibration, deconvolution, starless, synthetic_flat, comet_stack, frame_selection) on save/load

### Added (continued)
- **F-ExportPanel: Export Configuration UI + Pipeline Integration**
  - `ExportPanel` — UI dock with output directory picker (native file dialog), filename stem input, and format selector (FITS/TIFF32/XISF); bidirectional sync with `PipelineModel.export_config_changed`
  - 3 new `PipelineModel` properties (`output_path`, `output_format`, `output_filename`) + `export_config_changed` signal; full bidirectional `.astroai` sync via `output_path`/`output_format` project fields
  - `MainWindow` "Export" dock wired; `_on_run_full_pipeline` now appends `ExportStep` to the full pipeline when `output_path` is configured, saving result to disk automatically after Stack & Process
  - **Bugfix**: `PipelineModel.export_config()` was using `_starless_format` as the export format (wrong coupling); now uses the dedicated `_output_format` field
  - 28 new tests: 15 model tests (`TestPipelineModelExport`) + 13 widget tests (`TestExportPanel`)
- **F-LiveWorkflowTracking: Real-Time WorkflowGraph Step Highlighting**
  - `PipelineWorker.stage_active = Signal(str)` — new signal that emits `PipelineStage.name` exactly once per stage transition (deduplication via `_last_stage` guard); reset on each `start()` so second runs re-emit all stages
  - `_STAGE_TO_STEP_KEY` mapping in `MainWindow` (CALIBRATION → "calibrate", REGISTRATION → "register", STACKING → "stack", PROCESSING → "stretch", SAVING → "export", …)
  - `_on_pipeline_stage_active(stage_name)` calls `self._pipeline.advance_to(step_key)` — WorkflowGraph nodes transition PENDING → DONE/ACTIVE as the pipeline executes
  - `_on_pipeline_finished` marks the last ACTIVE step as DONE; `_on_pipeline_error` marks it as ERROR — WorkflowGraph always reaches a terminal state
  - 5 new `PipelineWorker` tests: signal emitted on stage entry, deduplication, correct order in multi-stage pipeline, reset between runs
- **F-FrameListPanel: Light-Frame Browser with Quality Scores**
  - `FrameListPanel` — dock widget with a read-only `QTableWidget` showing all loaded light frames; columns: Filename (basename + full-path tooltip), Belichtung (s), Qualität (% or —), Ausgewählt (✓/✗)
  - Count label: "N Frame(s) — M ausgewählt — K bewertet" updates on every refresh
  - `MainWindow` wired: "Light-Frames" dock on left area; `refresh()` called after `_on_import_lights`, after project load (`_sync_project_to_model`), and after pipeline completion
  - **Frame score writeback**: on `_on_pipeline_finished`, `frame_scores` from `FrameSelectionStep` metadata are written back to `project.input_frames[i].quality_score` and the panel is refreshed — quality scores become visible immediately after a full pipeline run
  - 13 new widget tests (`TestFrameListPanel`)
- **F-IntegrationE2E: Full Pipeline + Annotation Persistence Integration Tests**
  - `TestFullPipelineWithProductionSteps` (5 tests) — E2E coverage for `LoadFramesStep → RegistrationStep → StackingStep → StretchStep` using production step implementations with real FITS files on disk; validates result shape, [0,1] range, all step metadata keys (`loaded_frame_paths`, `registration_frames_aligned`, `stacking_method`, etc.), and progress stage emission
  - `TestAnnotationPersistenceE2E` (5 tests) — E2E roundtrip of `AnnotationConfig` through `ProjectSerializer`; covers all 4 flags, combined project with registration/stacking config, and legacy files missing the `annotation` key

### Added (continued)
- **F-ImageStats: Per-Kanal Bildstatistik-Dock**
  - `ImageStatsWidget` — Dock-Widget mit QTableWidget; zeigt Mittelwert, Std, Min, Max pro Kanal (R/G/B für Farbbilder, L für Mono)
  - Wired in `MainWindow._setup_docks()` + alle 3 `set_image_data()`-Aufrufe (Datei laden, Komet-Preview, Pipeline-Ergebnis)
  - 12 neue `TestImageStatsWidget` Widget-Tests
- **F-LiveHistogram: RGB+Luminanz Echtzeit-Histogramm (externer Beitrag)**
  - `live_histogram_view.py` — `HistogramView` (QWidget) mit `HistogramWorker` (QThread-basiert); zeichnet R/G/B- und Luminanz-Kurven überlagert; Log-Skala-Toggle
  - `app.py` verwendet jetzt `HistogramView` statt `HistogramWidget` im "Histogramm"-Dock
- **F-ToneCurves: Tonkurven-Pipeline-Schritt (externer Beitrag)**
  - `CurvesStep` (`astroai/processing/curves/pipeline_step.py`) — PROCESSING-Stage-Schritt; CubicSpline-LUT-basierte Tonkurvenanpassung für RGB-Gesamtkurve und R/G/B-Einzelkurven
  - `CurvesPanel` — UI-Dock mit interaktivem Kurven-Editor (Drag-Kontrollpunkte, Channel-Auswahl)
  - "curves"-Schritt in `PipelineModel.DEFAULT_STEPS` eingefügt (zwischen stretch und background_removal)
  - **Bugfix (CubicSpline-Linearität)**: `_build_lut()` verwendete `bc_type='clamped'` (erzwingt Ableitung=0 → S-Kurve); korrigiert auf `bc_type='not-a-knot'`; Testfehler `test_identity_lut_preserves_values` behoben
- **F-WcsHoverCoords: RA/Dec-Koordinaten im Statusbar**
  - `_wcs_adapter` auf `MainWindow` gespeichert; gesetzt wenn `set_wcs_solution()` ein gültiges WCS liefert; zurückgesetzt auf `None` bei Clear
  - `_on_pixel_hovered()` erweitert: wenn `_wcs_adapter` nicht None, `pixel_to_world()` aufgerufen und `RA ... Dec ...` an Statusbar-Nachricht angehängt
  - Fehlerbehandlung: `except Exception: pass` — kein Crash bei WCS-Fehlern
  - 6 neue Tests in `TestMainWindowSync` (Adapter initial None, hover ohne/mit WCS, None-Result, clear, Curves-Roundtrip)
- **TestCurvesConfigPersistenceE2E** — 5 neue E2E-Tests für CurvesConfig Roundtrip (enabled, defaults, rgb_points, per-channel, legacy file ohne curves-key)
- **F-SessionNotes: Session-Notizen Persistence**
  - `SessionNotesPanel` — UI dock widget with a plain-text `QTextEdit`; placeholder "Notizen zur Beobachtungsnacht..."
  - Wired to `ProjectMetadata.description` (already serialized via `asdict()` since v1.0) — `text_changed` signal writes immediately; `_sync_model_to_project()` flushes on save; `_load_project()` / `_on_new_project()` restore on load/reset
  - No `PipelineModel` or new dataclass fields required — `description: str = ""` existed in `ProjectMetadata` but had no UI
  - 10 new `TestSessionNotesPanel` widget tests + 5 new `TestSessionNotesPersistenceE2E` integration tests

- **F-FrameExposureImport + F-TotalExposure: FITS-Metadaten bei Import + Gesamt-Integrationszeit**
  - `_enrich_fits_entry(entry)` — liest FITS-Header beim Light-Frame-Import und befüllt `FrameEntry.exposure`, `gain_iso`, `temperature` aus EXPTIME/EXPOSURE, GAIN, CCD-TEMP/CCD_TEMP; silent no-op bei Fehler oder Nicht-FITS-Dateien
  - `FrameListPanel._refresh_count_label()` zeigt jetzt Gesamt-Integrationszeit der ausgewählten Frames: `"N Frame(s) — M ausgewählt — Xh MMm — K bewertet"`; Zeit nur bei `exposure is not None` und `selected=True`; `_format_exposure()` formatiert Sekunden → `"Xs"`, `"Xm Ys"`, `"Xh YYm"`
  - 10 neue `_enrich_fits_entry` Tests + 7 neue FrameListPanel-Expositionstests; total 2439 passed
- **F-FrameContextMenu: Rechtsklick-Kontextmenü im FrameListPanel**
  - Rechtsklick auf `FrameListPanel` öffnet Kontextmenü: "Alle auswählen", "Alle abwählen", "Auswahl umkehren", Trennlinie, "N Frame(s) entfernen" (nur wenn Tabellenzeilen selektiert)
  - Public-API: `select_all()`, `deselect_all()`, `invert_selection()` — direkt testbar ohne Menü-Simulation
  - Bulk-Ops mutieren alle `_entries` in-place und emittieren `selection_changed` nur für geänderte Zeilen; `_refresh_count_label()` einmalig am Ende
  - `remove_requested = Signal(list)` — emittiert sortierte Liste ausgewählter Zeilenindizes; `MainWindow._on_frames_remove_requested()` löscht rückwärts aus `project.input_frames` und refresht Panel + Button-State
  - `_table.SelectionMode` auf `ExtendedSelection` gesetzt (Ctrl/Shift-Mehrfachauswahl)
  - 12 neue `TestFrameListPanel` Bulk-Tests + 3 neue `TestMainWindowFrameSelection` Remove-Tests; total 2423 passed
- **F-FITSMetadata: FITS-Header-Metadaten-Dock**
  - `FileLoader` erweitert: FITS-Header wird zusammen mit dem Bild geladen; `header_loaded = Signal(object)` emittiert `dict[str, str] | None` (None für Nicht-FITS-Dateien) kurz vor `image_loaded`
  - `_extract_fits_header()` — extrahiert 16 definierte FITS-Schlüssel (OBJECT, DATE-OBS, EXPTIME, FILTER, TELESCOP, INSTRUME, FOCALLEN, XPIXSZ, RA, DEC, GAIN, CCD-TEMP, XBINNING, YBINNING, NAXIS1, NAXIS2); Aliase für alternative Schreibweisen (z.B. OBJCTRA/RA)
  - `FITSMetadataPanel` — QTableWidget-Dock mit deutschen Bezeichnungen (Ziel, Belichtung, Filter, Teleskop, Kamera …); zeigt Platzhalter wenn kein FITS geladen; `set_header(dict | None)` + `clear()`
  - `MainWindow`: "FITS-Metadaten"-Dock (rechts); `header_loaded` → `_fits_metadata.set_header()`; `_fits_metadata.clear()` bei neuem Projekt
  - `_LoadWorker.finished` jetzt 3-Tupel `(ndarray, filename, header | None)`; alle 7 betroffenen Unit-Tests aktualisiert
  - 13 neue Tests: `TestFITSHeaderExtraction` (3) + `TestFITSMetadataPanel` (10); total 2411 passed
- **F-ManualFrameReject: Interaktive Frame-Auswahl im FrameListPanel**
  - `FrameListPanel` upgraded from read-only to interactive: double-click any row to toggle its `selected` state
  - New `selection_changed = Signal(int, bool)` signal emits `(frame_index, new_selected)` after each toggle
  - `_entries` stored as a reference to the passed list — toggling mutates the project's `FrameEntry` in-place
  - Count label and "Ausgewählt" cell update immediately after each toggle; tooltip explains double-click
  - `MainWindow._on_frame_selection_changed(idx, selected)`: belt-and-suspenders project update + re-evaluates `_stack_run_act.setEnabled(any selected frames exist)` to prevent running with zero selected frames
  - **Bugfix**: `_on_run_full_pipeline` now filters `e.selected` when building `frame_paths` — previously ran all imported frames regardless of AI rejection; friendly message shown when none selected
  - 8 new `FrameListPanel` toggle tests + 4 new `TestMainWindowFrameSelection` tests; total 2398 passed
- **F-PreviewCompare: Vorher/Nachher Split-View**
  - `SplitCompareView` (`astroai/ui/widgets/split_compare_view.py`) — draggable vertical split widget; left half renders "Vorher" (before) image, right half renders "Nachher" (after) image; same tile-based lazy rendering, zoom, and pan as `ImageViewer`
  - Split handle: golden divider line with directional triangles at midpoint; drag within ±8 px to adjust split (clamped 5%–95%); cursor changes to `SplitHCursor` on hover
  - `MainWindow._view_stack` (`QStackedWidget`) wraps existing viewer container (page 0) and `SplitCompareView` (page 1); overlays unaffected
  - `MainWindow._before_image` — stores current image snapshot before `_on_run_pipeline()` starts
  - After pipeline finishes with a result, compare view is auto-populated and shown (Ctrl+D to toggle back)
  - "Ansicht → Vorher/Nachher Vergleich" (Ctrl+D) checkable menu action; enabled only when compare data available; `_reset_compare_state()` clears on new/load project
  - "An Fenster anpassen" (F) now routes to whichever view is active via `_on_fit_to_view()`
  - 21 new widget tests (`TestSplitCompareView*`) + 10 new `TestMainWindowCompareView` MainWindow tests; total 2387 passed
- **F-PipelineCancel: Cooperative Pipeline Cancellation**
  - `PipelineCancelledError(Exception)` — new exception raised by `Pipeline.run()` when cancel requested between steps
  - `CancelCheck = Callable[[], bool]` type alias + `noop_cancel()` default; `Pipeline.run()` gains optional `cancel_check` parameter — checked before each step
  - `PipelineWorker.cancel()` — sets a `threading.Event`; `start()` and `_cleanup()` clear it so second runs always start fresh
  - `PipelineWorker.cancelled = Signal()` — emitted when `PipelineCancelledError` is caught in `_RunnerWorker`
  - `MainWindow._cancel_act` — "Abbrechen" QAction with Escape shortcut; enabled when pipeline runs, disabled on finish/cancel/error
  - `_on_cancel_pipeline()` calls `self._pipeline_worker.cancel()`; `_on_pipeline_cancelled()` re-enables run actions + updates status bar
  - Exports: `CancelCheck`, `PipelineCancelledError`, `noop_cancel` added to `astroai.core.pipeline.__init__`
  - 7 new tests: `TestPipelineWorkerCancel` (5) + `TestPipelineCancelledError` (2); total suite 2356 passed

### Fixed
- `SpectralColorCalibrator._catalog_to_pixels()` returned a 2-tuple instead of the declared 3-tuple when catalog was empty (`calibrator.py:275`)
- **Duplicate registration sync** — `_sync_model_to_project()` and `_sync_project_to_model()` in `app.py` wrote `registration.upsample_factor` / `registration.reference_frame_index` twice (dead duplicate at lines 700-701 / 776-778); removed extra assignments
- **Silent GPU fallback** — `calibrate_frame()` swallowed GPU init errors with bare `except Exception: pass`; now logs `WARNING` via `logger.warning()` so operators can diagnose missing CUDA/MPS support

### Changed
- Version bump from `0.1.0-alpha` to `2.3.0-alpha` in `pyproject.toml` and `astroai/__init__.py`

### Quality
- 2349 unit + integration tests, all passing (3 skipped: `@gpu`/`@benchmark`)
- mypy: 0 errors across 137 source files
- ruff: 0 errors
- Coverage: 99% (21 genuinely untestable lines remaining)

---

## [2.2.0-alpha] — 2026-04-28

### Added
- **F-Comet: Comet Stacking (Dual-Tracking)**
  - `CometTracker` — AI-powered comet nucleus detection via difference imaging
  - `CometStacker` — dual-stack algorithm producing simultaneous star-aligned and comet-aligned outputs
  - `CometStackStep` — pipeline integration storing `comet_star_stack` and `comet_nucleus_stack` in context
  - `CometStackPanel` — UI dock with tracking-mode radio buttons and blend-factor slider

---

## [2.1.0-alpha] — 2026-04-28

### Added
- **F-5: Photometrische Farbkalibrierung (SPCC)**
  - `SpectralColorCalibrator` — queries GAIA DR3 or 2MASS, measures star RGB fluxes, fits correction matrix
  - `GAIACatalogClient` + `AAVSOCatalogClient` — TAP/Vizier catalog adapters
  - `PhotometryEngine` — aperture photometry with LoG star detection and GAIA magnitude calibration
  - `ColorCalibrationStep` — PROCESSING-stage pipeline step
  - `ColorCalibrationPanel` — UI dock with catalog selection and sample-radius controls
  - `PhotometryPanel` — UI dock showing matched stars, calibration R², and residuals

---

## [2.0.0-alpha] — 2026-04-28

### Added
- **F-3: Drizzle Super-Resolution** — WCS sub-pixel alignment, configurable drop-size/pixfrac/scale
- **F-4: Mosaic Assembly** — multi-panel WCS stitching with overlap detection and gradient correction
- **F-1: Plate Solving** — ASTAP integration, WCS overlay, DSO/star annotation overlay
- **F-2: GPU-Accelerated Calibration** — CUDA/MPS/CPU dark+flat correction with benchmark widget

---

## [1.1.0-alpha] — legacy

### Added
- Neural Frame Scoring (HFR, roundness, cloud detection)
- GPU-accelerated denoising (NAFNet)
- Starless separation and star reduction
- LRGB / Narrowband channel combination
- Richardson-Lucy deconvolution
- Background extraction and gradient removal
- Intelligent histogram stretch
- Freemium licensing with offline activation
- Project persistence (`.astroai` project files)
