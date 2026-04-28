# Changelog — AstroAI Suite

All notable changes are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.3.0-alpha] — unreleased

### Added
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
