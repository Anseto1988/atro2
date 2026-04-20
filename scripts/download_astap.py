#!/usr/bin/env python3
"""Download ASTAP binaries for bundling into the AstroAI package.

Usage:
    python scripts/download_astap.py                 # current platform only
    python scripts/download_astap.py --all            # all platforms
    python scripts/download_astap.py --platform linux-x86_64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from astroai.engine.platesolving.astap_binary import (
    _PLATFORM_SPECS,
    _bundled_path,
    _detect_platform_key,
    download_astap,
)

_BUNDLE_DIR = Path(__file__).parent.parent / "astroai" / "engine" / "platesolving" / "bin"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Download ASTAP binaries for bundling")
    parser.add_argument("--all", action="store_true", help="Download for all platforms")
    parser.add_argument("--platform", choices=list(_PLATFORM_SPECS.keys()), help="Target platform")
    args = parser.parse_args()

    if args.all:
        platforms = list(_PLATFORM_SPECS.keys())
    elif args.platform:
        platforms = [args.platform]
    else:
        platforms = [_detect_platform_key()]

    for plat in platforms:
        spec = _PLATFORM_SPECS[plat]
        target_dir = _BUNDLE_DIR / spec.key
        logging.info("Downloading ASTAP for %s -> %s", plat, target_dir)
        try:
            path = download_astap(target_dir)
            logging.info("OK: %s", path)
        except Exception as e:
            logging.error("FAILED for %s: %s", plat, e)
            sys.exit(1)


if __name__ == "__main__":
    main()
