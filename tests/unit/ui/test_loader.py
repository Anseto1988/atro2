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

def _make_fits_hdul(data, header_dict: dict | None = None):
    """Return a mock FITS HDUList context manager with hdul[0].data == data."""
    mock_hdu = MagicMock()
    mock_hdu.data = data
    mock_header = MagicMock()
    mock_header.get = lambda key, default=None: (header_dict or {}).get(key, default)
    mock_hdu.header = mock_header
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

        img, name, _hdr = blocker.args
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

        _, name, _hdr = blocker.args
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

        img, name, _hdr = blocker.args
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

        img, _, _hdr = blocker.args
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

        img, name, _hdr = blocker.args
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


class TestFITSHeaderExtraction:
    """Tests for FITS header extraction in _LoadWorker."""

    def test_fits_header_dict_returned(self, qtbot) -> None:
        fake_data = np.ones((4, 4), dtype=np.float32)
        worker = _LoadWorker(Path("/data/m31.fits"))
        hdul = _make_fits_hdul(fake_data, {"OBJECT": "M31", "EXPTIME": "300.0"})

        with patch("astropy.io.fits.open", return_value=hdul):
            with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
                worker.run()

        _, _name, header = blocker.args
        assert isinstance(header, dict)
        assert header.get("OBJECT") == "M31"
        assert header.get("EXPTIME") == "300.0"

    def test_non_fits_header_is_none(self, qtbot) -> None:
        import PIL.Image as PILImage

        grey = PILImage.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L")
        worker = _LoadWorker(Path("/data/snap.png"))

        with patch("PIL.Image.open", return_value=MagicMock(convert=lambda m: grey)):
            with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
                worker.run()

        _, _name, header = blocker.args
        assert header is None

    def test_file_loader_emits_header_loaded(self, qtbot, tmp_path: Path) -> None:
        loader = FileLoader()
        img_path = tmp_path / "test.png"
        from PIL import Image
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(img_path)

        with qtbot.waitSignal(loader.header_loaded, timeout=5000) as blocker:
            loader.load(img_path)

        # PNG → header is None
        assert blocker.args[0] is None


class TestFITSMetadataPanel:
    @pytest.fixture()
    def panel(self, qtbot):  # type: ignore[no-untyped-def]
        from astroai.ui.widgets.fits_metadata_panel import FITSMetadataPanel
        w = FITSMetadataPanel()
        qtbot.addWidget(w)
        w.resize(300, 400)
        w.show()
        return w

    def test_initial_shows_placeholder(self, panel) -> None:
        assert panel._placeholder.isVisible()
        assert not panel._table.isVisible()

    def test_set_header_with_data_shows_table(self, panel) -> None:
        panel.set_header({"OBJECT": "M31", "EXPTIME": "120"})
        assert panel._table.isVisible()
        assert not panel._placeholder.isVisible()

    def test_set_header_none_shows_placeholder(self, panel) -> None:
        panel.set_header({"OBJECT": "M31"})
        panel.set_header(None)
        assert panel._placeholder.isVisible()

    def test_set_header_empty_dict_shows_placeholder(self, panel) -> None:
        panel.set_header({})
        assert panel._placeholder.isVisible()

    def test_set_header_known_key_uses_label(self, panel) -> None:
        panel.set_header({"OBJECT": "NGC7000"})
        items = [
            panel._table.item(r, 0).text()
            for r in range(panel._table.rowCount())
        ]
        assert "Ziel" in items

    def test_set_header_exptime_key_shown(self, panel) -> None:
        panel.set_header({"EXPTIME": "300.0"})
        values = [
            panel._table.item(r, 1).text()
            for r in range(panel._table.rowCount())
        ]
        assert "300.0" in values

    def test_clear_hides_table(self, panel) -> None:
        panel.set_header({"OBJECT": "M42"})
        panel.clear()
        assert not panel._table.isVisible()
        assert panel._placeholder.isVisible()

    def test_clear_empties_rows(self, panel) -> None:
        panel.set_header({"OBJECT": "M42"})
        panel.clear()
        assert panel._table.rowCount() == 0

    def test_unknown_key_not_shown(self, panel) -> None:
        panel.set_header({"WEIRD_KEY": "some_value"})
        # WEIRD_KEY is not in _LABEL_MAP → placeholder shown
        assert panel._placeholder.isVisible()

    def test_multiple_known_keys_all_shown(self, panel) -> None:
        panel.set_header({
            "OBJECT": "M101",
            "EXPTIME": "600",
            "FILTER": "Ha",
            "TELESCOP": "Celestron C8",
        })
        assert panel._table.rowCount() == 4


class TestLoadWorkerRawBranch:
    """Cover the RAW DSLR loading path (lines 71-84) of _LoadWorker.run()."""

    def _make_meta(self, exposure=None, gain_iso=None, date_obs=None, width=64, height=64):
        m = MagicMock()
        m.exposure = exposure
        m.gain_iso = gain_iso
        m.date_obs = date_obs
        m.width = width
        m.height = height
        return m

    def test_raw_cr2_emits_finished(self, qtbot) -> None:
        rgb = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        meta = self._make_meta(exposure=120.0, gain_iso=800, date_obs="2026-01-01T00:00:00", width=64, height=64)
        worker = _LoadWorker(Path("/data/frame.cr2"))

        with patch("astroai.core.io.raw_io.read_raw", return_value=(rgb, meta)):
            with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
                worker.run()

        img, name, header = blocker.args
        assert img.dtype == np.float32
        assert img.ndim == 2
        assert name == "frame.cr2"
        assert header["Format"] == "CR2"
        assert header["EXPTIME"] == "120.0"
        assert header["GAIN"] == "800"
        assert header["DATE-OBS"] == "2026-01-01T00:00:00"
        assert header["NAXIS1"] == "64"
        assert header["NAXIS2"] == "64"

    def test_raw_nef_no_optional_fields(self, qtbot) -> None:
        """Lines 77-82: None optional meta fields are skipped."""
        rgb = np.ones((32, 32, 3), dtype=np.float32)
        meta = self._make_meta(exposure=None, gain_iso=None, date_obs=None, width=32, height=32)
        worker = _LoadWorker(Path("/data/shot.nef"))

        with patch("astroai.core.io.raw_io.read_raw", return_value=(rgb, meta)):
            with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
                worker.run()

        img, name, header = blocker.args
        assert name == "shot.nef"
        assert "EXPTIME" not in header
        assert "GAIN" not in header
        assert "DATE-OBS" not in header
        assert header["NAXIS1"] == "32"

    def test_raw_arw_collapses_rgb_to_2d(self, qtbot) -> None:
        """Line 75: mean over axis=2 yields 2-D grayscale."""
        rgb = np.arange(64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)
        meta = self._make_meta(width=64, height=64)
        worker = _LoadWorker(Path("/sensor/raw.arw"))

        with patch("astroai.core.io.raw_io.read_raw", return_value=(rgb, meta)):
            with qtbot.waitSignal(worker.finished, timeout=3000) as blocker:
                worker.run()

        img, _, _hdr = blocker.args
        assert img.ndim == 2
        assert img.shape == (64, 64)
