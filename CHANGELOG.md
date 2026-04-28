# Changelog — AstroAI Suite

All notable changes are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.3.0-alpha] — unreleased

### Added
- **F-SynFlat: Synthetic Flat Frame Generation**
  - `SyntheticFlatGenerator` — models illumination vignetting via tile-sampled RBF/polynomial surface fitting, median-combines multiple frames, applies Gaussian smoothing, normalises to peak = 1.0
  - `SyntheticFlatStep` — `CALIBRATION`-stage pipeline step that generates and applies the flat
  - `SyntheticFlatPanel` — UI dock widget with enable checkbox, tile-size and smoothing-sigma controls
  - `SyntheticFlatConfig` + `CometStackConfig` — project-file dataclasses for `.astroai` persistence
- **Full bidirectional project persistence** — `MainWindow._sync_model_to_project()` / `_sync_project_to_model()` synchronises all optional feature configs (drizzle, mosaic, channel_combine, color_calibration, deconvolution, starless, synthetic_flat, comet_stack) on save/load

### Fixed
- `SpectralColorCalibrator._catalog_to_pixels()` returned a 2-tuple instead of the declared 3-tuple when catalog was empty (`calibrator.py:275`)

### Changed
- Version bump from `0.1.0-alpha` to `2.3.0-alpha` in `pyproject.toml` and `astroai/__init__.py`

### Quality
- 1900 unit tests, all passing (3 skipped: `@gpu`/`@benchmark`)
- mypy: 0 errors across 130 source files
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
