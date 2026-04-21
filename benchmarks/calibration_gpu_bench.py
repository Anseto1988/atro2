"""GPU vs CPU calibration benchmark. Run directly: python benchmarks/calibration_gpu_bench.py"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from astroai.core.calibration.calibrate import calibrate_frame
from astroai.core.calibration.gpu_engine import GPUCalibrationEngine
from astroai.core.calibration.matcher import CalibrationFrame, CalibrationLibrary
from astroai.core.io.fits_io import ImageMetadata

FRAME_COUNT = 200
H, W = 4096, 4096
MIN_SPEEDUP = 3.0  # regression guard


def _make_lib(rng: np.random.Generator, meta: ImageMetadata) -> CalibrationLibrary:
    dark = (rng.random((H, W), dtype=np.float32) * 0.1).astype(np.float32)
    flat = (rng.random((H, W), dtype=np.float32) + 0.5).astype(np.float32)
    df = CalibrationFrame(path=Path("bench_dark.fits"), metadata=meta, data=dark)
    ff = CalibrationFrame(path=Path("bench_flat.fits"), metadata=meta, data=flat)
    return CalibrationLibrary(darks=[df], flats=[ff], bias=[])


def _bench_cpu(frames: list[np.ndarray], meta: ImageMetadata, lib: CalibrationLibrary) -> float:
    t0 = time.perf_counter()
    for f in frames:
        calibrate_frame(f, meta, lib, use_gpu=False)
    return time.perf_counter() - t0


def _bench_gpu(
    frames: list[np.ndarray],
    meta: ImageMetadata,
    lib: CalibrationLibrary,
    engine: GPUCalibrationEngine,
) -> float:
    # warmup
    engine.calibrate_batch_gpu(frames[:2], meta, lib)
    t0 = time.perf_counter()
    engine.calibrate_batch_gpu(frames, meta, lib)
    return time.perf_counter() - t0


def main() -> None:
    rng = np.random.default_rng(0)
    meta = ImageMetadata(exposure=120.0, gain_iso=800, temperature=-10.0, width=W, height=H)
    lib = _make_lib(rng, meta)
    frames = [rng.random((H, W), dtype=np.float32) for _ in range(FRAME_COUNT)]

    engine = GPUCalibrationEngine()
    device = engine.device_type
    print(f"Device: {device}  |  Frames: {FRAME_COUNT}  |  Resolution: {H}x{W}")

    cpu_sec = _bench_cpu(frames, meta, lib)
    cpu_fps = FRAME_COUNT / cpu_sec
    print(f"CPU: {cpu_fps:.2f} frames/s  ({cpu_sec:.2f}s total)")

    gpu_sec = _bench_gpu(frames, meta, lib, engine)
    gpu_fps = FRAME_COUNT / gpu_sec
    speedup = cpu_sec / gpu_sec
    print(f"GPU ({device}): {gpu_fps:.2f} frames/s  ({gpu_sec:.2f}s total)")
    print(f"Speedup: {speedup:.2f}x")

    results = {
        "device": device,
        "frame_count": FRAME_COUNT,
        "resolution": f"{H}x{W}",
        "cpu_fps": round(cpu_fps, 3),
        "gpu_fps": round(gpu_fps, 3),
        "speedup": round(speedup, 3),
        "regression_guard_min": MIN_SPEEDUP,
        "regression_guard_passed": speedup >= MIN_SPEEDUP,
    }
    out = Path("benchmark_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out}")

    if device == "cpu":
        print("Note: GPU not available – speedup measures CPU-to-CPU (batch vs. sequential).")
    elif speedup < MIN_SPEEDUP:
        print(f"WARNING: Speedup {speedup:.2f}x below regression guard {MIN_SPEEDUP}x!")
        raise SystemExit(1)
    else:
        print(f"OK: Speedup {speedup:.2f}x >= {MIN_SPEEDUP}x guard.")


if __name__ == "__main__":
    main()
