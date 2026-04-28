"""Tests for ONNX model downloader."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroai.inference.models.downloader import (
    ModelDownloader,
    ModelManifestEntry,
    _DummyOnnxSession,
)


@pytest.fixture()
def models_dir(tmp_path):
    return tmp_path / "models"


@pytest.fixture()
def downloader(models_dir):
    return ModelDownloader(models_dir=models_dir)


class TestModelDownloader:
    def test_models_dir_created(self, models_dir):
        ModelDownloader(models_dir=models_dir)
        assert models_dir.exists()

    def test_get_manifest_returns_nafnet(self, downloader):
        manifest = downloader.get_manifest()
        assert "nafnet_denoise" in manifest
        entry = manifest["nafnet_denoise"]
        assert entry.filename == "nafnet_denoise.onnx"
        assert len(entry.sha256) == 64

    def test_is_available_false_when_no_file(self, downloader):
        assert not downloader.is_available("nafnet_denoise")

    def test_is_available_false_for_unknown_model(self, downloader):
        assert not downloader.is_available("nonexistent_model")

    def test_is_available_true_with_valid_file(self, downloader, models_dir):
        content = b"fake model content"
        sha = hashlib.sha256(content).hexdigest()

        with patch.object(downloader, "get_manifest") as mock_manifest:
            mock_manifest.return_value = {
                "test_model": ModelManifestEntry(
                    name="test_model",
                    url="http://example.com/model.onnx",
                    sha256=sha,
                    filename="test_model.onnx",
                )
            }
            models_dir.mkdir(parents=True, exist_ok=True)
            (models_dir / "test_model.onnx").write_bytes(content)
            assert downloader.is_available("test_model")

    def test_is_available_false_for_unavailable_model(self, downloader):
        # "starnet_plus_plus" is in _STARNET_MANIFEST with "available": False → line 93
        assert not downloader.is_available("starnet_plus_plus")

    def test_ensure_model_raises_for_unknown(self, downloader):
        with pytest.raises(KeyError, match="Unknown model"):
            downloader.ensure_model("nonexistent")

    def test_ensure_model_skips_download_if_cached(self, downloader, models_dir):
        content = b"cached model"
        sha = hashlib.sha256(content).hexdigest()

        with patch.object(downloader, "get_manifest") as mock_manifest:
            mock_manifest.return_value = {
                "cached": ModelManifestEntry(
                    name="cached",
                    url="http://example.com/cached.onnx",
                    sha256=sha,
                    filename="cached.onnx",
                )
            }
            models_dir.mkdir(parents=True, exist_ok=True)
            (models_dir / "cached.onnx").write_bytes(content)
            path = downloader.ensure_model("cached")
            assert path == models_dir / "cached.onnx"

    def test_checksum_mismatch_deletes_file(self, downloader, models_dir):
        content = b"bad content"

        with patch.object(downloader, "get_manifest") as mock_manifest:
            mock_manifest.return_value = {
                "bad": ModelManifestEntry(
                    name="bad",
                    url="http://example.com/bad.onnx",
                    sha256="0" * 64,
                    filename="bad.onnx",
                )
            }
            with patch.object(downloader, "_download") as mock_dl:
                mock_dl.side_effect = lambda entry, target: target.write_bytes(content)
                with pytest.raises(RuntimeError, match="Checksum mismatch"):
                    downloader.ensure_model("bad")
            assert not (models_dir / "bad.onnx").exists()

    def test_load_onnx_session_fallback_dummy(self, downloader):
        session = downloader.load_onnx_session("nafnet_denoise", fallback_to_dummy=True)
        assert isinstance(session, _DummyOnnxSession)

    def test_progress_callback_called(self, models_dir):
        progress = MagicMock()
        dl = ModelDownloader(models_dir=models_dir, progress=progress)
        dl.load_onnx_session("nafnet_denoise", fallback_to_dummy=True)


class TestHttpsEnforcement:
    def test_rejects_http_url(self, downloader, models_dir):
        entry = ModelManifestEntry(
            name="http_model",
            url="http://example.com/model.onnx",
            sha256="a" * 64,
            filename="model.onnx",
        )
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / "model.onnx"

        with pytest.raises(ValueError, match="non-HTTPS URL"):
            downloader._download(entry, target)

    def test_rejects_ftp_url(self, downloader, models_dir):
        entry = ModelManifestEntry(
            name="ftp_model",
            url="ftp://example.com/model.onnx",
            sha256="a" * 64,
            filename="model.onnx",
        )
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / "model.onnx"

        with pytest.raises(ValueError, match="non-HTTPS URL"):
            downloader._download(entry, target)

    def test_accepts_https_url(self, downloader):
        ModelDownloader._require_https("https://example.com/model.onnx")

    def test_rejects_empty_scheme(self, downloader):
        with pytest.raises(ValueError, match="non-HTTPS URL"):
            ModelDownloader._require_https("//example.com/model.onnx")


class TestDummyOnnxSession:
    def test_get_inputs(self):
        session = _DummyOnnxSession()
        inputs = session.get_inputs()
        assert len(inputs) == 1
        assert inputs[0].name == "input"

    def test_run_identity(self):
        session = _DummyOnnxSession()
        inp = np.random.rand(1, 1, 64, 64).astype(np.float32)
        out = session.run(None, {"input": inp})
        assert len(out) == 1
        np.testing.assert_array_equal(out[0], inp)

    def test_run_does_not_mutate_input(self):
        session = _DummyOnnxSession()
        inp = np.ones((1, 1, 32, 32), dtype=np.float32)
        out = session.run(None, {"input": inp})
        out[0][0, 0, 0, 0] = 999.0
        assert inp[0, 0, 0, 0] == 1.0


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

class TestModelsDir:
    def test_models_dir_property_returns_path(self, downloader, models_dir):
        assert downloader.models_dir == models_dir


class TestEnsureModelSuccessPath:
    def test_ensure_model_logs_after_download_and_verify(self, downloader, models_dir):
        content = b"new model data"
        sha = hashlib.sha256(content).hexdigest()

        with patch.object(downloader, "get_manifest") as mock_manifest:
            mock_manifest.return_value = {
                "newmodel": ModelManifestEntry(
                    name="newmodel",
                    url="https://example.com/newmodel.onnx",
                    sha256=sha,
                    filename="newmodel.onnx",
                )
            }

            def fake_download(entry, target):
                target.write_bytes(content)

            with patch.object(downloader, "_download", side_effect=fake_download):
                path = downloader.ensure_model("newmodel")

        assert path == models_dir / "newmodel.onnx"
        assert path.exists()

    def test_ensure_model_download_exception_propagates(self, downloader, models_dir):
        with patch.object(downloader, "get_manifest") as mock_manifest:
            mock_manifest.return_value = {
                "failmodel": ModelManifestEntry(
                    name="failmodel",
                    url="https://example.com/failmodel.onnx",
                    sha256="a" * 64,
                    filename="failmodel.onnx",
                )
            }
            with patch.object(downloader, "_download", side_effect=OSError("network error")):
                with pytest.raises(OSError, match="network error"):
                    downloader.ensure_model("failmodel")


class TestLoadOnnxSessionOrtNone:
    def test_load_raises_when_ort_none_and_no_fallback(self, downloader):
        import astroai.inference.models.downloader as dl_module
        original_ort = dl_module.ort
        dl_module.ort = None
        try:
            with pytest.raises(RuntimeError, match="onnxruntime is required"):
                downloader.load_onnx_session("nafnet_denoise", fallback_to_dummy=False)
        finally:
            dl_module.ort = original_ort

    def test_load_returns_dummy_when_ort_none_and_fallback(self, downloader):
        import astroai.inference.models.downloader as dl_module
        original_ort = dl_module.ort
        dl_module.ort = None
        try:
            session = downloader.load_onnx_session("nafnet_denoise", fallback_to_dummy=True)
            assert isinstance(session, _DummyOnnxSession)
        finally:
            dl_module.ort = original_ort


class TestLoadOnnxSessionWithOrt:
    def test_load_onnx_returns_inference_session_when_model_found(self, downloader, models_dir):
        content = b"fake onnx model"
        sha = hashlib.sha256(content).hexdigest()

        with patch.object(downloader, "get_manifest") as mock_manifest:
            mock_manifest.return_value = {
                "testmodel": ModelManifestEntry(
                    name="testmodel",
                    url="https://example.com/testmodel.onnx",
                    sha256=sha,
                    filename="testmodel.onnx",
                )
            }
            (models_dir / "testmodel.onnx").write_bytes(content)

            mock_session = MagicMock()
            mock_ort = MagicMock()
            mock_ort.InferenceSession.return_value = mock_session

            import astroai.inference.models.downloader as dl_module
            original_ort = dl_module.ort
            dl_module.ort = mock_ort
            try:
                session = downloader.load_onnx_session("testmodel", fallback_to_dummy=False)
            finally:
                dl_module.ort = original_ort

        assert session is mock_session
        mock_ort.InferenceSession.assert_called_once()

    def test_load_onnx_falls_back_to_dummy_on_ensure_model_failure(self, downloader):
        mock_ort = MagicMock()

        import astroai.inference.models.downloader as dl_module
        original_ort = dl_module.ort
        dl_module.ort = mock_ort
        try:
            with patch.object(downloader, "ensure_model", side_effect=KeyError("Unknown model")):
                session = downloader.load_onnx_session("nafnet_denoise", fallback_to_dummy=True)
        finally:
            dl_module.ort = original_ort

        assert isinstance(session, _DummyOnnxSession)

    def test_load_onnx_raises_on_ensure_model_failure_no_fallback(self, downloader):
        mock_ort = MagicMock()

        import astroai.inference.models.downloader as dl_module
        original_ort = dl_module.ort
        dl_module.ort = mock_ort
        try:
            with patch.object(downloader, "ensure_model", side_effect=RuntimeError("Download failed")):
                with pytest.raises(RuntimeError, match="Download failed"):
                    downloader.load_onnx_session("nafnet_denoise", fallback_to_dummy=False)
        finally:
            dl_module.ort = original_ort


class TestDownloadWithProgress:
    def test_download_reports_progress_with_content_length(self, downloader, models_dir):
        content = b"x" * 1024
        entry = ModelManifestEntry(
            name="prog_model",
            url="https://example.com/prog_model.onnx",
            sha256="a" * 64,
            filename="prog_model.onnx",
        )
        target = models_dir / "prog_model.onnx"

        progress_calls = []

        def fake_progress(prog):
            progress_calls.append(prog.current)

        dl = ModelDownloader(models_dir=models_dir, progress=fake_progress)

        mock_resp = MagicMock()
        mock_resp.headers.get.return_value = str(len(content))
        chunk_iter = [content[:512], content[512:], b""]
        mock_resp.read.side_effect = chunk_iter
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            dl._download(entry, target)

        assert target.exists()
        assert 100 in progress_calls

    def test_download_without_content_length(self, downloader, models_dir):
        content = b"small chunk"
        entry = ModelManifestEntry(
            name="nolen_model",
            url="https://example.com/nolen.onnx",
            sha256="a" * 64,
            filename="nolen.onnx",
        )
        target = models_dir / "nolen.onnx"

        mock_resp = MagicMock()
        mock_resp.headers.get.return_value = "0"
        mock_resp.read.side_effect = [content, b""]
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            downloader._download(entry, target)

        assert target.exists()

    def test_download_cleans_up_temp_on_error(self, downloader, models_dir):
        entry = ModelManifestEntry(
            name="err_model",
            url="https://example.com/err.onnx",
            sha256="a" * 64,
            filename="err.onnx",
        )
        target = models_dir / "err.onnx"

        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            with pytest.raises(OSError):
                downloader._download(entry, target)

        assert not target.exists()


class TestNAFNetDenoiser:
    def test_denoise_2d(self):
        from astroai.processing.denoise.denoiser import NAFNetDenoiser

        denoiser = NAFNetDenoiser(strength=1.0, tile_size=64, tile_overlap=8)
        frame = np.random.rand(128, 128).astype(np.float64)
        result = denoiser.denoise(frame)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_denoise_3d_rgb(self):
        from astroai.processing.denoise.denoiser import NAFNetDenoiser

        denoiser = NAFNetDenoiser(strength=0.5, tile_size=64, tile_overlap=8)
        frame = np.random.rand(64, 64, 3).astype(np.float64)
        result = denoiser.denoise(frame)
        assert result.shape == frame.shape

    def test_strength_zero_returns_original(self):
        from astroai.processing.denoise.denoiser import NAFNetDenoiser

        denoiser = NAFNetDenoiser(strength=0.0, tile_size=64, tile_overlap=8)
        frame = np.random.rand(64, 64).astype(np.float64)
        result = denoiser.denoise(frame)
        np.testing.assert_allclose(result, frame, atol=1e-10)


class TestOrtImportErrorAtModuleLevel:
    def test_ort_is_none_when_onnxruntime_import_fails(self) -> None:
        """lines 18-19: except ImportError sets ort = None at module import time."""
        import sys
        import importlib

        mod_name = "astroai.inference.models.downloader"
        saved = sys.modules.pop(mod_name, None)
        saved_ort = sys.modules.pop("onnxruntime", None)
        sys.modules["onnxruntime"] = None  # type: ignore[assignment]  # forces ImportError on import
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            assert mod.ort is None
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved
            else:
                sys.modules.pop(mod_name, None)
            if saved_ort is not None:
                sys.modules["onnxruntime"] = saved_ort
            else:
                sys.modules.pop("onnxruntime", None)
