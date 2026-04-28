"""Unit tests for GradientRemover."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from astroai.processing.background.extractor import BackgroundExtractor
from astroai.processing.background.gradient_remover import GradientRemover


def _flat(value: float = 100.0, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def _gradient(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Ramp gradient: left=0, right=50."""
    w = shape[1]
    row = np.linspace(0.0, 50.0, w, dtype=np.float32)
    return np.tile(row, (shape[0], 1))


class TestGradientRemoverBasic:
    def test_output_shape_preserved(self) -> None:
        remover = GradientRemover()
        img = np.random.default_rng(0).random((128, 128)).astype(np.float32)
        out = remover.remove(img)
        assert out.shape == img.shape

    def test_output_dtype_preserved_float32(self) -> None:
        remover = GradientRemover()
        img = np.ones((64, 64), dtype=np.float32) * 200.0
        out = remover.remove(img)
        assert out.dtype == np.float32

    def test_output_dtype_preserved_float64(self) -> None:
        remover = GradientRemover()
        img = np.ones((64, 64), dtype=np.float64) * 200.0
        out = remover.remove(img)
        assert out.dtype == np.float64

    def test_flat_image_preserve_median(self) -> None:
        """Flat image: background ≈ image → result ≈ 0 + original_median."""
        remover = GradientRemover(preserve_median=True, clip_negative=False)
        img = _flat(100.0)
        out = remover.remove(img)
        # Result should be close to original_median (100.0) since bg ≈ 100
        assert float(np.mean(out)) == pytest.approx(100.0, abs=5.0)

    def test_clip_negative_removes_negative_values(self) -> None:
        extractor = MagicMock(spec=BackgroundExtractor)
        # Background larger than image → result would be negative
        extractor.extract.return_value = np.full((32, 32), 200.0, dtype=np.float64)
        remover = GradientRemover(extractor=extractor, preserve_median=False, clip_negative=True)
        img = np.full((32, 32), 100.0, dtype=np.float32)
        out = remover.remove(img)
        assert float(out.min()) >= 0.0

    def test_clip_negative_false_allows_negatives(self) -> None:
        extractor = MagicMock(spec=BackgroundExtractor)
        extractor.extract.return_value = np.full((32, 32), 200.0, dtype=np.float64)
        remover = GradientRemover(extractor=extractor, preserve_median=False, clip_negative=False)
        img = np.full((32, 32), 100.0, dtype=np.float32)
        out = remover.remove(img)
        assert float(out.min()) < 0.0

    def test_preserve_median_false_does_not_add_offset(self) -> None:
        extractor = MagicMock(spec=BackgroundExtractor)
        extractor.extract.return_value = np.zeros((32, 32), dtype=np.float64)
        remover = GradientRemover(extractor=extractor, preserve_median=False, clip_negative=False)
        img = np.full((32, 32), 50.0, dtype=np.float32)
        out = remover.remove(img)
        # background=0, no median added → result = 50.0
        assert float(np.mean(out)) == pytest.approx(50.0, abs=0.1)


class TestGradientRemoverMockedExtractor:
    def _make_remover(
        self,
        bg_value: float,
        preserve_median: bool = True,
        clip_negative: bool = True,
    ) -> tuple[GradientRemover, MagicMock]:
        extractor = MagicMock(spec=BackgroundExtractor)
        extractor.extract.return_value = np.full((32, 32), bg_value, dtype=np.float64)
        remover = GradientRemover(
            extractor=extractor,
            preserve_median=preserve_median,
            clip_negative=clip_negative,
        )
        return remover, extractor

    def test_extractor_called_once(self) -> None:
        remover, extractor = self._make_remover(10.0)
        img = np.ones((32, 32), dtype=np.float32) * 50.0
        remover.remove(img)
        extractor.extract.assert_called_once()

    def test_zero_background_adds_median(self) -> None:
        # bg=0, preserve_median=True → result = (img - 0) + median(img) = 2 * img
        remover, _ = self._make_remover(0.0, preserve_median=True, clip_negative=False)
        img = np.full((32, 32), 80.0, dtype=np.float32)
        out = remover.remove(img)
        assert float(np.mean(out)) == pytest.approx(160.0, abs=0.1)

    def test_uniform_background_removed(self) -> None:
        remover, _ = self._make_remover(30.0, preserve_median=False, clip_negative=False)
        img = np.full((32, 32), 100.0, dtype=np.float32)
        out = remover.remove(img)
        assert float(np.mean(out)) == pytest.approx(70.0, abs=0.1)


class TestGradientRemoverBatch:
    def test_remove_batch_returns_same_count(self) -> None:
        remover = GradientRemover()
        frames = [np.ones((32, 32), dtype=np.float32) * i for i in range(1, 5)]
        results = remover.remove_batch(frames)
        assert len(results) == 4

    def test_remove_batch_shapes_preserved(self) -> None:
        remover = GradientRemover()
        frames = [np.ones((32, 32), dtype=np.float32)] * 3
        results = remover.remove_batch(frames)
        for r in results:
            assert r.shape == (32, 32)


class TestGradientRemoverExtractBackground:
    def test_extract_background_returns_correct_shape(self) -> None:
        extractor = MagicMock(spec=BackgroundExtractor)
        extractor.extract.return_value = np.full((32, 32), 50.0, dtype=np.float64)
        remover = GradientRemover(extractor=extractor)
        img = np.ones((32, 32), dtype=np.float32) * 100.0
        bg = remover.extract_background(img)
        assert bg.shape == img.shape
        assert bg.dtype == img.dtype

    def test_extractor_property(self) -> None:
        ext = BackgroundExtractor()
        remover = GradientRemover(extractor=ext)
        assert remover.extractor is ext
