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
