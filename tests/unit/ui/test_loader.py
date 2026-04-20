"""Tests for the background FileLoader."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from astroai.ui.main.loader import FileLoader


class TestFileLoader:
    @pytest.fixture()
    def loader(self, qtbot) -> FileLoader:  # type: ignore[no-untyped-def]
        return FileLoader()

    def test_initial_state(self, loader: FileLoader) -> None:
        assert not loader.is_loading

    def test_load_emits_image_loaded(self, loader: FileLoader, qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        img_path = tmp_path / "test.png"
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (32, 32), dtype=np.uint8), mode="L")
        img.save(img_path)

        with qtbot.waitSignal(loader.image_loaded, timeout=5000) as blocker:
            loader.load(img_path)

        data, name = blocker.args
        assert isinstance(data, np.ndarray)
        assert data.shape == (32, 32)
        assert name == "test.png"

    def test_load_emits_error_for_missing_file(self, loader: FileLoader, qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        bad_path = tmp_path / "nonexistent.png"

        with qtbot.waitSignal(loader.load_error, timeout=5000):
            loader.load(bad_path)

    def test_load_emits_status(self, loader: FileLoader, qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        img_path = tmp_path / "status_test.png"
        from PIL import Image

        img = Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L")
        img.save(img_path)

        with qtbot.waitSignal(loader.load_status, timeout=5000) as blocker:
            loader.load(img_path)

        assert "status_test.png" in blocker.args[0]

    def test_prevents_concurrent_loads(self, loader: FileLoader, qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        img_path = tmp_path / "concurrent.png"
        from PIL import Image

        img = Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L")
        img.save(img_path)

        loader.load(img_path)
        assert loader.is_loading
        loader.load(img_path)  # should be silently ignored

        with qtbot.waitSignal(loader.image_loaded, timeout=5000):
            pass
