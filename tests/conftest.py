from __future__ import annotations

import torch
import pytest


def _probe_gpu() -> bool:
    """Check if a GPU device is actually usable (not just reported as available)."""
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

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="No usable GPU (CUDA CC mismatch or unavailable)")
