"""Unit-Tests für CometTracker."""
from __future__ import annotations

import numpy as np
import pytest

from astroai.engine.comet.tracker import CometPosition, CometTracker


def _blank_frames(n: int = 4, h: int = 64, w: int = 64) -> list[np.ndarray]:
    return [np.zeros((h, w), dtype=np.float32) for _ in range(n)]


def _frames_with_comet(
    positions: list[tuple[int, int]],
    h: int = 64,
    w: int = 64,
    comet_brightness: float = 1.0,
    star_brightness: float = 0.1,
) -> list[np.ndarray]:
    """Erstellt synthetische Frames: Sterne gleich + Kometenkopf wandernd."""
    rng = np.random.default_rng(42)
    star_frame = np.zeros((h, w), dtype=np.float64)
    for _ in range(10):
        sy, sx = rng.integers(0, h), rng.integers(0, w)
        star_frame[sy, sx] = star_brightness

    frames = []
    for py, px in positions:
        f = star_frame.copy()
        # Setze Gauss-Blob für Kometenkopf
        yy, xx = np.ogrid[:h, :w]
        blob = comet_brightness * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / 4.0)
        f += blob
        frames.append(f.astype(np.float32))
    return frames


class TestCometPosition:
    def test_repr(self) -> None:
        p = CometPosition(10.5, 20.3, confidence=0.9)
        assert "10.50" in repr(p)
        assert "20.30" in repr(p)


class TestCometTracker:
    def test_raises_on_empty(self) -> None:
        tracker = CometTracker()
        with pytest.raises(ValueError, match="Keine Frames"):
            tracker.track([])

    def test_returns_one_position_per_frame(self) -> None:
        tracker = CometTracker()
        frames = _blank_frames(5)
        # Füge hell-Pixel hinzu damit Tracker etwas findet
        for f in frames:
            f[32, 32] = 1.0
        positions = tracker.track(frames)
        assert len(positions) == 5

    def test_detects_moving_comet(self) -> None:
        comet_coords = [(20, 20), (20, 22), (20, 24), (20, 26)]
        frames = _frames_with_comet(comet_coords, h=64, w=64)
        tracker = CometTracker(min_blob_area=1, top_fraction=0.005)
        positions = tracker.track(frames)
        assert len(positions) == 4
        # Komet bewegt sich in X-Richtung → x-Koordinaten sollten zunehmen
        xs = [p.x for p in positions]
        assert xs[3] > xs[0], f"Erwartet x-Drift: {xs}"

    def test_grayscale_input(self) -> None:
        frames = _blank_frames(2, h=32, w=32)
        frames[0][10, 10] = 1.0
        frames[1][10, 12] = 1.0
        tracker = CometTracker()
        positions = tracker.track(frames)
        assert len(positions) == 2
        assert all(isinstance(p, CometPosition) for p in positions)

    def test_rgb_input(self) -> None:
        frames = [np.zeros((32, 32, 3), dtype=np.float32) for _ in range(2)]
        frames[0][15, 15, :] = [1.0, 1.0, 1.0]
        frames[1][15, 17, :] = [1.0, 1.0, 1.0]
        tracker = CometTracker(min_blob_area=1)
        positions = tracker.track(frames)
        assert len(positions) == 2

    def test_confidence_in_range(self) -> None:
        comet_coords = [(30, 30), (30, 32)]
        frames = _frames_with_comet(comet_coords)
        tracker = CometTracker()
        positions = tracker.track(frames)
        for p in positions:
            assert 0.0 <= p.confidence <= 1.0
