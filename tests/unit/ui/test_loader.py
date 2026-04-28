"""Tests for the background FileLoader and _LoadWorker."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroai.ui.main.loader import FileLoader, _LoadWorker


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


# ---------------------------------------------------------------------------
# _LoadWorker – direct unit tests for each branch in run()
# ---------------------------------------------------------------------------

def _make_fits_hdul(data):
    """Return a mock FITS HDUList context manager with hdul[0].data == data."""
    mock_hdu = MagicMock()
    mock_hdu.data = data
    mock_hdul = MagicMock()
    mock_hdul.__enter__ = lambda s: mock_hdul
    mock_hdul.__exit__ = MagicMock(return_value=False)
    mock_hdul.__getitem__ = lambda s, i: mock_hdu
    return mock_hdul


class TestLoadWorkerFitsBranch:
    """Cover the FITS loading path (lines 24-32) of _LoadWorker.run()."""

    def test_fits_success_emits_finished(self, qtbot) -> None:
        """FITS path: valid data array → finished signal with float32 ndarray."""
        fake_data = np.ones((8, 8), dtype=np.int16)
        worker = _LoadWorker(Path("/fake/image.fits"))

        with patch("astropy.io.fits.open", return_value=_make_fits_hdul(fake_data)):
            with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
                worker.run()

        img, name = blocker.args
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.float32
        assert name == "image.fits"

    def test_fits_none_data_emits_error(self, qtbot) -> None:
        """Lines 29-31: hdul[0].data is None → error signal emitted."""
        worker = _LoadWorker(Path("/fake/empty.fits"))

        with patch("astropy.io.fits.open", return_value=_make_fits_hdul(None)):
            with qtbot.waitSignal(worker.error, timeout=3000) as blocker:
                worker.run()

        assert "empty.fits" in blocker.args[0]

    def test_fit_suffix_uses_fits_branch(self, qtbot) -> None:
        """Suffix .fit also triggers the FITS branch."""
        fake_data = np.zeros((4, 4), dtype=np.float32)
        worker = _LoadWorker(Path("/data/frame.fit"))

        with patch("astropy.io.fits.open", return_value=_make_fits_hdul(fake_data)):
            with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
                worker.run()

        _, name = blocker.args
        assert name == "frame.fit"


class TestLoadWorkerTiffBranch:
    """Cover the TIFF loading path (lines 33-39) of _LoadWorker.run()."""

    def test_tiff_grayscale_emits_finished(self, qtbot) -> None:
        """2-D TIFF array passes through unchanged as float32."""
        import PIL.Image as PILImage

        grey = np.full((10, 10), 128, dtype=np.uint8)
        pil_img = PILImage.fromarray(grey, mode="L")
        worker = _LoadWorker(Path("/data/mono.tiff"))

        with patch("PIL.Image.open", return_value=pil_img):
            with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
                worker.run()

        img, name = blocker.args
        assert img.dtype == np.float32
        assert img.ndim == 2
        assert name == "mono.tiff"

    def test_tiff_rgb_collapses_to_2d(self, qtbot) -> None:
        """3-D TIFF is mean-collapsed to a 2-D float32 array (lines 38-39)."""
        import PIL.Image as PILImage

        rgb = np.ones((6, 6, 3), dtype=np.uint8) * 100
        pil_img = PILImage.fromarray(rgb, mode="RGB")
        worker = _LoadWorker(Path("/data/colour.tif"))

        with patch("PIL.Image.open", return_value=pil_img):
            with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
                worker.run()

        img, _ = blocker.args
        assert img.ndim == 2


class TestLoadWorkerOtherBranch:
    """Cover the 'other image' loading path (lines 40-44) of _LoadWorker.run()."""

    def test_png_uses_convert_L(self, qtbot) -> None:
        """Non-FITS/TIFF path converts to 'L' via PIL."""
        import PIL.Image as PILImage

        grey = np.full((8, 8), 200, dtype=np.uint8)
        pil_grey = PILImage.fromarray(grey, mode="L")

        mock_open_result = MagicMock()
        mock_open_result.convert.return_value = pil_grey
        worker = _LoadWorker(Path("/data/snap.png"))

        with patch("PIL.Image.open", return_value=mock_open_result):
            with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
                worker.run()

        img, name = blocker.args
        assert img.dtype == np.float32
        assert img.ndim == 2
        assert name == "snap.png"
        mock_open_result.convert.assert_called_once_with("L")

    def test_jpg_emits_error_on_io_failure(self, qtbot) -> None:
        """Line 47-48: PIL.Image.open raising IOError → error signal."""
        worker = _LoadWorker(Path("/missing/photo.jpg"))

        with patch("PIL.Image.open", side_effect=OSError("file not found")):
            with qtbot.waitSignal(worker.error, timeout=3000) as blocker:
                worker.run()

        assert "file not found" in blocker.args[0]


class TestLoadWorkerStatusSignal:
    """status signal is emitted before loading starts (line 22)."""

    def test_status_signal_emitted_with_filename(self, qtbot) -> None:
        """status.emit fires synchronously before any I/O."""
        import PIL.Image as PILImage

        grey = PILImage.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L")
        mock_open = MagicMock()
        mock_open.convert.return_value = grey
        worker = _LoadWorker(Path("/data/test_status.png"))

        received: list[str] = []
        worker.status.connect(received.append)

        with patch("PIL.Image.open", return_value=mock_open):
            with qtbot.waitSignal(worker.finished, timeout=3000):
                worker.run()

        assert received
        assert "test_status.png" in received[0]
