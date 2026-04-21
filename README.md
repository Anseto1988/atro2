# AstroAI Suite

AI-powered astrophotography processing suite.

**Version:** 2.0.0-alpha

## What's New in v2.0

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
- Starless separation and star reduction
- LRGB / Narrowband channel combination
- Deconvolution (Richardson-Lucy)
- Background extraction and gradient removal
- Intelligent histogram stretch
- Dark-theme GUI with histogram, workflow graph, and progress tracking
- Project persistence (save/open `.astroai` project files)
- Freemium licensing with offline activation

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

Download the latest bundle from the
[GitHub Actions artifacts](../../actions/workflows/build.yml):

| Platform | Archive |
|----------|---------|
| Windows  | `astroai-2.0.0-alpha-win.zip` |
| Linux    | `astroai-2.0.0-alpha-linux.tar.gz` |

### Windows

1. Extract `astroai-2.0.0-alpha-win.zip`
2. Run `AstroAI/AstroAI.exe`

### Linux

```bash
tar xzf astroai-2.0.0-alpha-linux.tar.gz
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
