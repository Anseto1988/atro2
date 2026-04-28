# AstroAI Suite

[![CI](https://github.com/Anseto1988/atro2/actions/workflows/ci.yml/badge.svg)](https://github.com/Anseto1988/atro2/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/Anseto1988/atro2/branch/main/graph/badge.svg)](https://codecov.io/gh/Anseto1988/atro2)

AI-powered astrophotography processing suite.

**Version:** 2.5.0-alpha

## Feature Overview

| ID | Feature | Since |
|----|---------|-------|
| F-1 | Plate Solving (ASTAP + astrometry.net fallback) | v2.0.0-alpha |
| F-2 | GPU-Accelerated Calibration (Dark/Flat, CUDA/MPS/CPU) | v2.0.0-alpha |
| F-3 | Drizzle Super-Resolution (WCS sub-pixel stacking) | v2.0.0-alpha |
| F-4 | Mosaic Assembly (multi-panel WCS stitching) | v2.0.0-alpha |
| F-5 | Photometric Color Calibration / SPCC (GAIA DR3 / 2MASS) | v2.1.0-alpha |
| F-Comet | Comet Stacking — Dual-Tracking (star + nucleus aligned) | v2.2.0-alpha |
| F-SynFlat | Synthetic Flat Generation via RBF/Polynomial surface fit | v2.3.0-alpha |
| F-CalibScan | Calibration library scanner with auto-partition | v2.4.0-alpha |
| F-PipelinePreset | Pipeline preset save/load + 4 built-in presets | v2.4.0-alpha |
| F-SmartCalibUI | Smart Calibration panel with auto-match and coverage view | v2.4.0-alpha |
| F-ProjectValidation | Project validation with typed issues and severity | v2.4.0-alpha |
| F-RawDSLR | DSLR RAW ingestion (Canon CR2, Nikon NEF, Sony ARW) | v2.5.0-alpha |
| F-AIStarAlignment | AI-powered star alignment engine | v2.5.0-alpha |

## What's New in v2.5

### F-RawDSLR: DSLR RAW Support
- Native RAW ingestion via **rawpy** — no external converter required
- Supported cameras: Canon (CR2/CR3), Nikon (NEF), Sony (ARW), and any libraw-compatible format
- `RawIO.load()` auto-demosaics and normalises to float32 FITS-compatible arrays
- `LoadFramesStep` transparently handles mixed FITS + RAW frame lists
- `FrameListPanel` and `Loader` updated with RAW workflow support
- Metadata (EXIF ISO, exposure, focal length) extracted and stored in frame registry

### F-AIStarAlignment: AI-Powered Star Alignment
- `StarAligner` engine: keypoint-based AI feature matching for robust alignment under heavy distortion
- Affine + homography registration with RANSAC outlier rejection
- Replaces legacy centroid-only approach — works on crowded fields and wide-angle setups
- Integrated into the registration pipeline stage; fully backwards-compatible

### CI/CD Pipeline
- GitHub Actions workflow (`.github/workflows/ci.yml`): lint → mypy → pytest-cov
- Coverage gate: **≥ 95%** enforced on every pull request
- Automatic badge update on `main` merges

## What's New in v2.4

### Calibration & Frame Management
- **F-CalibScan:** `scan_directory` / `partition_by_type` / `build_calibration_library` — auto-organises Dark, Flat, Bias from any folder structure
- **F-CalibMatchBatch:** `batch_match` / `suggest_calibration_config` — bulk frame-to-calibration matching with typed `BatchMatchResult`
- **F-SmartCalibUI:** `SmartCalibPanel` dock with auto-match button and visual coverage display
- **F-DragDropFrameImport:** Drag-and-drop FITS import directly into `FrameListPanel`
- **F-ImportFolderAction:** Recursive folder import via `Ctrl+Shift+F`

### Pipeline Presets
- **F-PipelinePreset:** `PipelinePreset` dataclass + `PresetManager` with JSON persistence
- **F-BuiltinPresets:** 4 built-in presets — *Deepsky LRGB*, *Narrowband SHO*, *Narrowband HOO*, *Planetarisch*
- **F-PresetUI:** Pipeline menu → Save Preset / Load Preset / preset selector

### Project & Workflow
- **F-ProjectValidation:** `validate_project` with `ValidationIssue` / `ValidationResult` typed output
- **F-ProjectSummary:** `compute_summary` → `ProjectSummary` + `ExposureGroup` aggregates
- **F-FrameExportStats:** CSV export of per-frame quality metrics (`Ctrl+Shift+E`)
- **F-SavePreviewImage:** Save current preview as PNG / JPEG / TIFF (`Ctrl+Shift+P`)
- **F-FramePreviewOnClick:** `preview_requested` signal + context menu entry per frame
- **F-FrameNotesField:** Per-frame notes with `QInputDialog` tooltip persistence
- **F-ShortcutsDialog:** Grouped keyboard shortcut reference dialog (`Ctrl+?`)
- **F-CalibStatusLabel:** Live calibration status label in the status bar

### Code Quality (v2.4)
- 2 751 tests (3 skipped) — all passing
- 0 mypy errors across 161 source files
- ~97% overall coverage

## What's New in v2.3

### F-SynFlat: Synthetic Flat Frame Generation
- Models the optical vignetting pattern from science light frames via tile-sampled RBF / polynomial surface fitting — no dedicated flat frames required
- Median-combines multi-frame models to suppress astronomical signal contamination
- Gaussian smoothing pass suppresses tile-grid artefacts before normalisation
- `SyntheticFlatGenerator` + `SyntheticFlatStep` pipeline integration (CALIBRATION stage)
- `SyntheticFlatPanel` UI dock with enable/disable toggle, tile-size and smoothing-sigma controls
- Full project persistence: `SyntheticFlatConfig` stored in `.astroai` project file

### Full Project State Persistence
- `MainWindow._sync_model_to_project()` / `_sync_project_to_model()` — bidirectional sync for all optional features (drizzle, mosaic, channel_combine, color_calibration, deconvolution, starless, synthetic_flat, comet_stack)
- Project load/save now restores the complete UI state of all configurable panels

### Code Quality (v2.3)
- 1 900 unit tests — all passing
- 0 mypy errors across 130 source files
- 0 ruff lint errors
- ~100% coverage on all new modules

## What's New in v2.2

### F-Comet: Comet Stacking (Dual-Tracking)
- AI-powered comet nucleus detection via difference imaging (`CometTracker`)
- Dual-Tracking: simultaneous star-aligned and comet-nucleus-aligned stacks from the same frame set
- Configurable tracking mode: `stars` / `comet` / `blend` (adjustable blend factor)
- `CometStackStep` pipeline integration — stores `comet_star_stack` and `comet_nucleus_stack` in context

## What's New in v2.1

### F-5: Photometric Color Calibration (SPCC)
- Spectrophotometric color calibration matching observed star colors against GAIA DR3 or 2MASS catalogs
- Full 3×3 correction matrix with residual RMS diagnostics and per-channel white balance output
- `SpectralColorCalibrator` + `ColorCalibrationStep` pipeline integration
- `ColorCalibrationPanel` UI widget with catalog selection and minimum-star-count controls

## What's New in v2.0

### F-3: Drizzle Super-Resolution
- WCS-based sub-pixel alignment using plate-solving output from F-1
- Configurable drop-size (0.5 / 0.7 / 1.0), pixfrac and scale parameters
- `DrizzleEngine` + `DrizzleStep` pipeline integration after stacking
- `DrizzlePanel` UI widget for drop-size and pixfrac configuration

### F-4: Mosaic Assembly
- Multi-panel WCS-based stitching with automatic overlap detection
- Linear/average blend modes with gradient correction
- `MosaicEngine` + `MosaicStep` pipeline integration
- `MosaicPanel` UI widget for panel management and blend configuration

### F-1: Plate Solving (ASTAP)
- Automatic astrometric plate solving via ASTAP with astrometry.net fallback
- RA/Dec sky coordinate overlay on the image viewer
- Deep-sky object and star annotation overlay with local catalogs
- `AstrometryStep` pipeline integration — WCS solution stored in pipeline context

### F-2: GPU-Accelerated Calibration
- `GPUCalibrationEngine`: CUDA / MPS / CPU fallback via PyTorch
- Batch calibration with Dark/Flat tensor reuse for maximum throughput
- `CalibrationBenchmarkWidget`: real-time GPU vs. CPU speedup metrics
- `CalibrationWorker`: non-blocking QThread with benchmark signal emission

## Features

- FITS / XISF / RAW image loading (Canon CR2, Nikon NEF, Sony ARW, DSLR via rawpy)
- AI-based frame scoring (HFR, roundness, cloud detection) and stacking
- GPU-accelerated denoising (NAFNet, CUDA/MPS/CPU)
- **Plate Solving** (ASTAP) with sky overlay and annotation
- **GPU-Accelerated Calibration** (Dark/Flat correction) with benchmark metrics
- **Drizzle Super-Resolution** (WCS sub-pixel stacking, configurable drop-size)
- **Mosaic Assembly** (multi-panel WCS stitching with gradient correction)
- **Photometric Color Calibration** (SPCC via GAIA DR3 / 2MASS stellar catalogs)
- **Comet Stacking** (Dual-Tracking: simultaneous star- and comet-aligned stacks)
- **Synthetic Flat Generation** (vignetting model from light frames when real flats are unavailable)
- **DSLR RAW Support** (Canon/Nikon/Sony — CR2/NEF/ARW — auto-demosaic to float32)
- **AI Star Alignment** (keypoint-based engine with RANSAC homography)
- **Calibration Library Scanner** (auto-partition + smart matching)
- **Pipeline Presets** (save/load + 4 built-in deep-sky and planetary presets)
- **Smart Calibration UI** (auto-match panel with visual coverage display)
- **Project Validation** (typed issue reporting before pipeline execution)
- Starless separation and star reduction
- LRGB / Narrowband channel combination
- Deconvolution (Richardson-Lucy)
- Background extraction and gradient removal
- Intelligent histogram stretch
- Dark-theme GUI with histogram, workflow graph, and progress tracking
- Project persistence (save/open `.astroai` project files)
- Keyboard shortcut reference dialog (`Ctrl+?`)
- Freemium licensing with offline activation

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| UI Framework | PyQt6 |
| Numerical core | NumPy, SciPy |
| GPU compute | PyTorch (CUDA / MPS / CPU fallback) |
| FITS I/O | astropy |
| RAW decode | rawpy (libraw) |
| Plate solving | ASTAP binary + astrometry.net HTTP |
| Packaging | Poetry + PyInstaller |
| Testing | pytest + pytest-cov (≥ 95% gate) |
| Static analysis | mypy (strict) + ruff |
| CI/CD | GitHub Actions |

## CI/CD Status

The CI pipeline runs on every push and pull request to `main`:

1. **Lint** — `ruff check astroai/`
2. **Type-check** — `mypy astroai/` (zero-error policy)
3. **Unit tests** — `pytest tests/unit/ --cov=astroai --cov-fail-under=95`
4. **Integration tests** — `pytest tests/integration/`
5. **Coverage badge** — uploaded to Codecov on `main`

## Releases

| Version | Tag | Key Features |
|---------|-----|-------------|
| v1.1.0-alpha | [v1.1.0-alpha](https://github.com/Anseto1988/atro2/releases/tag/v1.1.0-alpha) | E3–E6, Security S-01–S-04, Licensing API |
| v2.0.0-alpha | [v2.0.0-alpha](https://github.com/Anseto1988/atro2/releases/tag/v2.0.0-alpha) | F-1 Plate Solving, F-2 GPU-Calibration, F-3 Drizzle, F-4 Mosaic |
| v2.1.0-alpha | [v2.1.0-alpha](https://github.com/Anseto1988/atro2/releases/tag/v2.1.0-alpha) | F-5 Photometric Color Calibration (SPCC) |
| v2.2.0-alpha | [v2.2.0-alpha](https://github.com/Anseto1988/atro2/releases/tag/v2.2.0-alpha) | F-Comet Dual-Tracking Comet Stacking (1 278 tests) |
| v2.3.0-alpha | [v2.3.0-alpha](https://github.com/Anseto1988/atro2/releases/tag/v2.3.0-alpha) | F-SynFlat Synthetic Flat, full project persistence (1 900 tests) |
| v2.4.0-alpha | [v2.4.0-alpha](https://github.com/Anseto1988/atro2/releases/tag/v2.4.0-alpha) | CalibScan, PipelinePresets, SmartCalibUI, ProjectValidation (2 751 tests) |
| v2.5.0-alpha | _(current)_ | F-RawDSLR DSLR RAW, F-AIStarAlignment, CI/CD pipeline (≥ 95% coverage gate) |

## Requirements

- Python 3.12+
- CUDA-capable GPU or Apple Silicon (recommended; CPU fallback available)
- ASTAP binary for plate solving (optional; auto-detectable from PATH or `ASTAP_BINARY_PATH`)
- libraw / rawpy for DSLR RAW support (`pip install rawpy`)

## Installation (from source)

```bash
pip install poetry
poetry install
```

## Running

```bash
poetry run astroai
```

## Plate Solving Setup

Install ASTAP from https://www.hnsky.org/astap.htm and ensure it is on your PATH,
or set the `ASTAP_BINARY_PATH` environment variable to the binary location.

Download a star catalog (H18 recommended for wide fields):

```bash
python scripts/download_astap.py
```

## GPU Calibration Benchmark

```bash
python benchmarks/calibration_gpu_bench.py
```

## Pre-built Bundles

Download the latest release from
[GitHub Releases](https://github.com/Anseto1988/atro2/releases/tag/v2.5.0-alpha):

| Platform | Archive |
|----------|---------|
| Windows  | [`astroai-2.5.0-alpha-win.zip`](https://github.com/Anseto1988/atro2/releases/download/v2.5.0-alpha/astroai-2.5.0-alpha-win.zip) |
| Linux    | [`astroai-2.5.0-alpha-linux.tar.gz`](https://github.com/Anseto1988/atro2/releases/download/v2.5.0-alpha/astroai-2.5.0-alpha-linux.tar.gz) |

**Older releases:**

| Version | Windows | Linux |
|---------|---------|-------|
| v2.4.0-alpha | [`win.zip`](https://github.com/Anseto1988/atro2/releases/download/v2.4.0-alpha/astroai-2.4.0-alpha-win.zip) | [`linux.tar.gz`](https://github.com/Anseto1988/atro2/releases/download/v2.4.0-alpha/astroai-2.4.0-alpha-linux.tar.gz) |
| v2.3.0-alpha | [`win.zip`](https://github.com/Anseto1988/atro2/releases/download/v2.3.0-alpha/astroai-2.3.0-alpha-win.zip) | [`linux.tar.gz`](https://github.com/Anseto1988/atro2/releases/download/v2.3.0-alpha/astroai-2.3.0-alpha-linux.tar.gz) |
| v2.2.0-alpha | [`win.zip`](https://github.com/Anseto1988/atro2/releases/download/v2.2.0-alpha/astroai-2.2.0-alpha-win.zip) | [`linux.tar.gz`](https://github.com/Anseto1988/atro2/releases/download/v2.2.0-alpha/astroai-2.2.0-alpha-linux.tar.gz) |
| v2.1.0-alpha | [`win.zip`](https://github.com/Anseto1988/atro2/releases/download/v2.1.0-alpha/astroai-2.1.0-alpha-win.zip) | [`linux.tar.gz`](https://github.com/Anseto1988/atro2/releases/download/v2.1.0-alpha/astroai-2.1.0-alpha-linux.tar.gz) |
| v2.0.0-alpha | [`win.zip`](https://github.com/Anseto1988/atro2/releases/download/v2.0.0-alpha/astroai-2.0.0-alpha-win.zip) | [`linux.tar.gz`](https://github.com/Anseto1988/atro2/releases/download/v2.0.0-alpha/astroai-2.0.0-alpha-linux.tar.gz) |

### Windows

1. Extract the `.zip` archive
2. Run `AstroAI/AstroAI.exe`

### Linux

```bash
tar xzf astroai-2.5.0-alpha-linux.tar.gz
./AstroAI/AstroAI
```

## Development

```bash
poetry install --with dev
poetry run pytest tests/unit/ -v
poetry run pytest tests/integration/ -v
poetry run ruff check astroai/
poetry run mypy astroai/
```

### Running GPU calibration tests

```bash
poetry run pytest tests/unit/core/test_gpu_calibration.py -v
```

### Running astrometry tests (no ASTAP required — subprocess mocked)

```bash
poetry run pytest tests/unit/astrometry/ tests/integration/test_astrometry_e2e.py -v
```

### Running RAW ingestion tests

```bash
poetry run pytest tests/unit/core/test_raw_io.py tests/unit/core/test_load_frames_step.py -v
```

## Building Bundles Locally

```bash
poetry run pyinstaller scripts/astroai.spec --noconfirm
```

Output is written to `dist/AstroAI/`. See `scripts/BUILD.md` for full instructions.

## License

Proprietary — all rights reserved.
