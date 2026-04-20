# AstroAI Suite

AI-powered astrophotography processing suite.

**Version:** 0.1.0-alpha (pre-alpha)

## Features

- FITS/XISF/RAW image loading
- AI-based frame scoring and stacking
- GPU-accelerated denoising and star management
- Dark-theme GUI with histogram, workflow graph, and progress tracking

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended for inference)

## Installation (from source)

```bash
pip install poetry
poetry install
```

## Running

```bash
poetry run astroai
```

## Pre-built Bundles

Download the latest pre-alpha bundle from the
[GitHub Actions artifacts](../../actions/workflows/build.yml):

| Platform | Archive |
|----------|---------|
| Windows  | `astroai-0.1.0-alpha-win.zip` |
| Linux    | `astroai-0.1.0-alpha-linux.tar.gz` |
| macOS    | `astroai-0.1.0-alpha-macos.tar.gz` |

### Windows

1. Extract `astroai-0.1.0-alpha-win.zip`
2. Run `AstroAI/AstroAI.exe`

### Linux

```bash
tar xzf astroai-0.1.0-alpha-linux.tar.gz
./AstroAI/AstroAI
```

### macOS

```bash
tar xzf astroai-0.1.0-alpha-macos.tar.gz
open AstroAI.app
```

## Development

```bash
poetry install --with dev
poetry run pytest tests/unit/ -v
poetry run ruff check astroai/
poetry run mypy astroai/
```

## Building Bundles Locally

```bash
poetry run pyinstaller scripts/astroai.spec --noconfirm
```

Output is written to `dist/AstroAI/`.

## License

Proprietary — all rights reserved.
