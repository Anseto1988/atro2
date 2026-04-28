from __future__ import annotations

import os
import sys

# Set offscreen platform BEFORE any Qt import to prevent display crashes and
# ensure headless test runs (CI/WSL) never try to connect to X11/Wayland.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


def _probe_gpu() -> bool:
    try:
        import torch
    except Exception:
        return False
    for device_str in ("cuda", "mps"):
        if device_str == "cuda" and not torch.cuda.is_available():
            continue
        if device_str == "mps" and not torch.backends.mps.is_available():
            continue
        try:
            torch.zeros(1, device=torch.device(device_str))
            return True
        except Exception:
            continue
    return False


HAS_GPU = _probe_gpu()

requires_gpu = pytest.mark.skipif(
    not HAS_GPU, reason="No usable GPU (CUDA CC mismatch or unavailable)"
)


@pytest.fixture(scope="session")
def qapp_args() -> list[str]:
    """CLI args passed to QApplication — consumed by pytest-qt's qapp fixture."""
    return [sys.argv[0]]


@pytest.fixture(scope="session")
def qapp(qapp_args: list[str]):  # type: ignore[override]
    """Session-scoped QApplication singleton.

    Overrides pytest-qt's built-in fixture to keep one QApplication alive for
    the entire session.  This prevents PySide6 teardown segfaults that appear
    at ~96% completion when pytest-cov's atexit hooks race Qt's cleanup.

    Key properties:
    - quitOnLastWindowClosed=False  → no premature exit mid-suite
    - processEvents() at teardown   → flush pending Qt events cleanly
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(qapp_args)

    app.setQuitOnLastWindowClosed(False)

    yield app

    # Flush pending events before Python GC runs Qt object destructors.
    app.processEvents()
