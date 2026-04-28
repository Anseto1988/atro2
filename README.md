# AstroAI Suite

AI-powered astrophotography processing suite.

**Version:** 2.2.0-alpha

## What's New in v2.2

### F-Comet: Comet Stacking (Dual-Tracking)
- AI-powered comet nucleus detection via difference imaging (`CometTracker`)
- Dual-Tracking: simultaneous star-aligned and comet-nucleus-aligned stacks from the same frame set
- Configurable tracking mode: `stars` / `comet` / `blend` (adjustable blend factor)
- `CometStackStep` pipeline integration — stores `comet_star_stack` and `comet_nucleus_stack` in context
- _Screenshot: Dual-Stack output comparison — coming with the v2.2.0-alpha release_

## What's New in v2.1

### F-5: Photometrische Farbkalibrierung (SPCC)
- Spectrophotometric color calibration matching observed star colors against GAIA DR3 or 2MASS catalogs
- Full 3×3 correction matrix with residual RMS diagnostics and per-channel white balance output
- `SpectralColorCalibrator` + `ColorCalibrationStep` pipeline integration
- `ColorCalibrationPanel` UI widget with catalog selection and minimum-star-count controls
- _Screenshot: Before/After color calibration — coming with the v2.1.0-alpha release_

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

- FITS / XISF / RAW image loading with metadata extraction
- AI-based frame scoring (HFR, roundness, cloud detection) and stacking
- GPU-accelerated denoising (NAFNet, CUDA/MPS/CPU)
- Plate solving (ASTAP) with sky overlay and annotation
- GPU-accelerated calibration (Dark/Flat correction) with benchmark metrics
- **Drizzle Super-Resolution** (WCS sub-pixel stacking, configurable drop-size)
- **Mosaic Assembly** (multi-panel WCS stitching with gradient correction)
- **Photometric Color Calibration** (SPCC via GAIA DR3 / 2MASS stellar catalogs)
- **Comet Stacking** (Dual-Tracking: simultaneous star- and comet-aligned stacks)
- Starless separation and star reduction
- LRGB / Narrowband channel combination
- Deconvolution (Richardson-Lucy)
- Background extraction and gradient removal
- Intelligent histogram stretch
- Dark-theme GUI with histogram, workflow graph, and progress tracking
- Project persistence (save/open `.astroai` project files)
- Freemium licensing with offline activation

## Releases

| Version | Tag | Features |
|---------|-----|----------|
| v1.1.0-alpha | [v1.1.0-alpha](https://github.com/Anseto1988/atro2/releases/tag/v1.1.0-alpha) | E3–E6, Security S-01–S-04, Licensing API |
| v2.0.0-alpha | [v2.0.0-alpha](https://github.com/Anseto1988/atro2/releases/tag/v2.0.0-alpha) | F-1 Plate Solving, F-2 GPU-Kalibrierung, F-3 Drizzle Super-Resolution, F-4 Mosaic-Assembly |
| v2.1.0-alpha | [v2.1.0-alpha](https://github.com/Anseto1988/atro2/releases/tag/v2.1.0-alpha) | F-5 Photometrische Farbkalibrierung (SPCC) |
| v2.2.0-alpha | [v2.2.0-alpha](https://github.com/Anseto1988/atro2/releases/tag/v2.2.0-alpha) | F-Comet Dual-Tracking Comet Stacking, Test-Suite 1278 Tests |

## Requirements

- Python 3.12+
- CUDA-capable GPU or Apple Silicon (recommended; CPU fallback available)
- ASTAP binary for plate solving (optional; auto-detectable from PATH or `ASTAP_BINARY_PATH`)

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
[GitHub Releases](https://github.com/Anseto1988/atro2/releases/tag/v2.2.0-alpha):

| Platform | Archive |
|----------|---------|
| Windows  | [`astroai-2.2.0-alpha-win.zip`](https://github.com/Anseto1988/atro2/releases/download/v2.2.0-alpha/astroai-2.2.0-alpha-win.zip) |
| Linux    | [`astroai-2.2.0-alpha-linux.tar.gz`](https://github.com/Anseto1988/atro2/releases/download/v2.2.0-alpha/astroai-2.2.0-alpha-linux.tar.gz) |

**Older releases:**

| Version | Windows | Linux |
|---------|---------|-------|
| v2.1.0-alpha | [`win.zip`](https://github.com/Anseto1988/atro2/releases/download/v2.1.0-alpha/astroai-2.1.0-alpha-win.zip) | [`linux.tar.gz`](https://github.com/Anseto1988/atro2/releases/download/v2.1.0-alpha/astroai-2.1.0-alpha-linux.tar.gz) |
| v2.0.0-alpha | [`win.zip`](https://github.com/Anseto1988/atro2/releases/download/v2.0.0-alpha/astroai-2.0.0-alpha-win.zip) | [`linux.tar.gz`](https://github.com/Anseto1988/atro2/releases/download/v2.0.0-alpha/astroai-2.0.0-alpha-linux.tar.gz) |

### Windows

1. Extract the `.zip` archive
2. Run `AstroAI/AstroAI.exe`

### Linux

```bash
tar xzf astroai-2.2.0-alpha-linux.tar.gz
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

## Building Bundles Locally

```bash
poetry run pyinstaller scripts/astroai.spec --noconfirm
```

Output is written to `dist/AstroAI/`. See `scripts/BUILD.md` for full instructions.

## License

Proprietary — all rights reserved.
