"""Tests for OnnxModelRegistry (VER-363)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from astroai.core.onnx_registry import (
    DEFAULT_CACHE_DIR,
    ModelSpec,
    OnnxModelRegistry,
    _DummyOnnxSession,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    OnnxModelRegistry.reset()
    yield
    OnnxModelRegistry.reset()


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture()
def registry(cache_dir: Path) -> OnnxModelRegistry:
    return OnnxModelRegistry(cache_dir=cache_dir)


class TestSingleton:
    def test_returns_same_instance(self, cache_dir: Path) -> None:
        a = OnnxModelRegistry(cache_dir=cache_dir)
        b = OnnxModelRegistry()
        assert a is b

    def test_reset_creates_new_instance(self, cache_dir: Path) -> None:
        a = OnnxModelRegistry(cache_dir=cache_dir)
        OnnxModelRegistry.reset()
        b = OnnxModelRegistry(cache_dir=cache_dir)
        assert a is not b


class TestBuiltinModels:
    def test_list_models_contains_builtins(self, registry: OnnxModelRegistry) -> None:
        models = registry.list_models()
        assert "nafnet_denoise" in models
        assert "starnet_plus_plus" in models
        assert "blind_deconv" in models

    def test_register_custom_model(self, registry: OnnxModelRegistry) -> None:
        spec = ModelSpec(
            name="custom_model",
            url="https://example.com/model.onnx",
            sha256="a" * 64,
            filename="custom.onnx",
            version="1.0.0",
        )
        registry.register(spec)
        assert "custom_model" in registry.list_models()


class TestIsAvailable:
    def test_returns_false_for_unknown_model(self, registry: OnnxModelRegistry) -> None:
        assert not registry.is_available("nonexistent")

    def test_returns_false_for_unavailable_model(self, registry: OnnxModelRegistry) -> None:
        assert not registry.is_available("starnet_plus_plus")

    def test_returns_false_when_file_missing(self, registry: OnnxModelRegistry) -> None:
        assert not registry.is_available("nafnet_denoise")

    def test_returns_true_with_valid_cached_file(
        self, registry: OnnxModelRegistry, cache_dir: Path
    ) -> None:
        content = b"test model data"
        sha = hashlib.sha256(content).hexdigest()
        spec = ModelSpec(
            name="cached_test",
            url="https://example.com/model.onnx",
            sha256=sha,
            filename="cached.onnx",
            version="1.0.0",
        )
        registry.register(spec)
        model_dir = cache_dir / "cached_test" / "1.0.0"
        model_dir.mkdir(parents=True)
        (model_dir / "cached.onnx").write_bytes(content)
        assert registry.is_available("cached_test")


class TestVersionedCacheDir:
    def test_model_path_includes_version(
        self, registry: OnnxModelRegistry, cache_dir: Path
    ) -> None:
        spec = registry._specs["nafnet_denoise"]
        path = registry._model_path(spec)
        assert path == cache_dir / "nafnet_denoise" / "0.1.0" / "nafnet_denoise.onnx"


class TestGetSession:
    def test_returns_dummy_for_unknown_model_with_fallback(
        self, registry: OnnxModelRegistry
    ) -> None:
        session = registry.get_session("unknown_model", fallback_to_dummy=True)
        assert isinstance(session, _DummyOnnxSession)

    def test_raises_for_unknown_model_without_fallback(
        self, registry: OnnxModelRegistry
    ) -> None:
        with pytest.raises(KeyError, match="Unknown model"):
            registry.get_session("unknown_model", fallback_to_dummy=False)

    def test_returns_dummy_when_ort_unavailable(
        self, registry: OnnxModelRegistry
    ) -> None:
        import astroai.core.onnx_registry as mod
        original = mod.ort
        mod.ort = None
        try:
            OnnxModelRegistry.reset()
            reg = OnnxModelRegistry(cache_dir=registry._cache_dir)
            session = reg.get_session("nafnet_denoise", fallback_to_dummy=True)
            assert isinstance(session, _DummyOnnxSession)
        finally:
            mod.ort = original

    def test_raises_when_ort_unavailable_no_fallback(
        self, registry: OnnxModelRegistry
    ) -> None:
        import astroai.core.onnx_registry as mod
        original = mod.ort
        mod.ort = None
        try:
            OnnxModelRegistry.reset()
            reg = OnnxModelRegistry(cache_dir=registry._cache_dir)
            with pytest.raises(RuntimeError, match="onnxruntime is required"):
                reg.get_session("nafnet_denoise", fallback_to_dummy=False)
        finally:
            mod.ort = original

    def test_caches_session_across_calls(
        self, registry: OnnxModelRegistry
    ) -> None:
        s1 = registry.get_session("nafnet_denoise", fallback_to_dummy=True)
        s2 = registry.get_session("nafnet_denoise", fallback_to_dummy=True)
        assert s1 is s2

    def test_evict_clears_cached_session(
        self, registry: OnnxModelRegistry
    ) -> None:
        s1 = registry.get_session("nafnet_denoise", fallback_to_dummy=True)
        registry.evict("nafnet_denoise")
        s2 = registry.get_session("nafnet_denoise", fallback_to_dummy=True)
        assert s1 is not s2


class TestLoadFromPath:
    def test_returns_none_for_missing_path(self, registry: OnnxModelRegistry) -> None:
        result = registry.load_from_path("/nonexistent/model.onnx")
        assert result is None

    def test_returns_none_when_ort_unavailable(
        self, registry: OnnxModelRegistry, tmp_path: Path
    ) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"data")
        import astroai.core.onnx_registry as mod
        original = mod.ort
        mod.ort = None
        try:
            OnnxModelRegistry.reset()
            reg = OnnxModelRegistry(cache_dir=tmp_path / "cache")
            assert reg.load_from_path(str(model_file)) is None
        finally:
            mod.ort = original


class TestBackendLabel:
    def test_cpu_label_without_gpu(self, registry: OnnxModelRegistry) -> None:
        label = registry.backend_label
        assert label in ("[CPU]", "[GPU]", "[ONNX]")

    def test_cpu_label_when_ort_unavailable(self) -> None:
        import astroai.core.onnx_registry as mod
        original = mod.ort
        mod.ort = None
        try:
            OnnxModelRegistry.reset()
            reg = OnnxModelRegistry()
            assert reg.backend_label == "[CPU]"
        finally:
            mod.ort = original
            OnnxModelRegistry.reset()


class TestProviderDetection:
    def test_providers_returns_list(self, registry: OnnxModelRegistry) -> None:
        providers = registry.providers
        assert isinstance(providers, list)
        assert "CPUExecutionProvider" in providers

    def test_detect_providers_without_ort(self) -> None:
        import astroai.core.onnx_registry as mod
        original = mod.ort
        mod.ort = None
        try:
            result = OnnxModelRegistry._detect_providers()
            assert result == ["CPUExecutionProvider"]
        finally:
            mod.ort = original


class TestSHA256Verification:
    def test_verify_valid_hash(self, tmp_path: Path) -> None:
        content = b"verify me"
        sha = hashlib.sha256(content).hexdigest()
        path = tmp_path / "file.bin"
        path.write_bytes(content)
        assert OnnxModelRegistry._verify_sha256(path, sha)

    def test_verify_invalid_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "file.bin"
        path.write_bytes(b"some data")
        assert not OnnxModelRegistry._verify_sha256(path, "0" * 64)


class TestDownloadHttpsEnforcement:
    def test_rejects_http_url(self, registry: OnnxModelRegistry, cache_dir: Path) -> None:
        spec = ModelSpec(
            name="http_model",
            url="http://example.com/model.onnx",
            sha256="a" * 64,
            filename="model.onnx",
        )
        target = cache_dir / "http_model" / "0.1.0" / "model.onnx"
        target.parent.mkdir(parents=True)
        with pytest.raises(ValueError, match="non-HTTPS"):
            registry._download(spec, target, None)


class TestDummySession:
    def test_identity_pass_through(self) -> None:
        session = _DummyOnnxSession()
        inp = np.random.rand(1, 1, 32, 32).astype(np.float32)
        out = session.run(None, {"input": inp})
        np.testing.assert_array_equal(out[0], inp)

    def test_does_not_mutate_input(self) -> None:
        session = _DummyOnnxSession()
        inp = np.ones((1, 1, 16, 16), dtype=np.float32)
        out = session.run(None, {"input": inp})
        out[0][0, 0, 0, 0] = 999.0
        assert inp[0, 0, 0, 0] == 1.0


class TestProgressCallback:
    def test_set_progress_is_used(self, registry: OnnxModelRegistry) -> None:
        calls: list[str] = []
        registry.set_progress(lambda p: calls.append(p.message))
        session = registry.get_session("nafnet_denoise", fallback_to_dummy=True)
        assert isinstance(session, _DummyOnnxSession)


# ---------------------------------------------------------------------------
# _DummyOnnxSession.get_inputs (line 65)
# ---------------------------------------------------------------------------

class TestDummySessionGetInputs:
    def test_get_inputs_returns_named_input(self) -> None:
        session = _DummyOnnxSession()
        inputs = session.get_inputs()
        assert len(inputs) == 1
        assert inputs[0].name == "input"


# ---------------------------------------------------------------------------
# backend_label — GPU path (line 185)
# ---------------------------------------------------------------------------

class TestBackendLabelGPU:
    def test_gpu_label_with_cuda_provider(self, registry: OnnxModelRegistry) -> None:
        registry._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert registry.backend_label == "[GPU]"

    def test_gpu_label_with_dml_provider(self, registry: OnnxModelRegistry) -> None:
        registry._providers = ["DmlExecutionProvider"]
        assert registry.backend_label == "[GPU]"

    def test_onnx_label_with_cpu_only_provider(self, registry: OnnxModelRegistry) -> None:
        registry._providers = ["CPUExecutionProvider"]
        assert registry.backend_label == "[ONNX]"


# ---------------------------------------------------------------------------
# load_from_path — session creation failure (lines 164-168)
# ---------------------------------------------------------------------------

class TestLoadFromPathFailure:
    def test_returns_none_when_session_creation_fails(
        self, registry: OnnxModelRegistry, tmp_path: Path
    ) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"not a real onnx file")
        import astroai.core.onnx_registry as mod

        mock_ort = MagicMock()
        mock_ort.InferenceSession.side_effect = RuntimeError("invalid model")
        original = mod.ort
        mod.ort = mock_ort
        try:
            result = registry.load_from_path(str(model_file))
        finally:
            mod.ort = original
        assert result is None


# ---------------------------------------------------------------------------
# _load_session — InferenceSession success (line 211) + raise path (line 216)
# ---------------------------------------------------------------------------

class TestLoadSessionBranches:
    def test_returns_real_session_when_model_cached(
        self, registry: OnnxModelRegistry, cache_dir: Path
    ) -> None:
        content = b"fake onnx content"
        sha = hashlib.sha256(content).hexdigest()
        spec = ModelSpec(
            name="cached_real",
            url="https://example.com/m.onnx",
            sha256=sha,
            filename="m.onnx",
        )
        registry.register(spec)
        model_dir = cache_dir / "cached_real" / "0.1.0"
        model_dir.mkdir(parents=True)
        (model_dir / "m.onnx").write_bytes(content)

        mock_session = MagicMock()
        import astroai.core.onnx_registry as mod

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        original = mod.ort
        mod.ort = mock_ort
        try:
            session = registry.get_session("cached_real", fallback_to_dummy=False)
        finally:
            mod.ort = original
        assert session is mock_session

    def test_raises_when_ensure_model_fails_no_fallback(
        self, registry: OnnxModelRegistry
    ) -> None:
        with patch.object(registry, "_ensure_model", side_effect=RuntimeError("dl fail")):
            with pytest.raises(RuntimeError, match="dl fail"):
                registry.get_session("nafnet_denoise", fallback_to_dummy=False)

    def test_returns_dummy_when_ensure_model_fails_with_fallback(
        self, registry: OnnxModelRegistry
    ) -> None:
        with patch.object(registry, "_ensure_model", side_effect=RuntimeError("dl fail")):
            session = registry.get_session("nafnet_denoise", fallback_to_dummy=True)
        assert isinstance(session, _DummyOnnxSession)


# ---------------------------------------------------------------------------
# _ensure_model — cached path (lines 221-222) + sha mismatch (lines 228-234)
# ---------------------------------------------------------------------------

class TestEnsureModel:
    def test_returns_cached_path_when_file_valid(
        self, registry: OnnxModelRegistry, cache_dir: Path
    ) -> None:
        content = b"valid model"
        sha = hashlib.sha256(content).hexdigest()
        spec = ModelSpec(
            name="valid_cached",
            url="https://example.com/v.onnx",
            sha256=sha,
            filename="v.onnx",
        )
        registry.register(spec)
        model_dir = cache_dir / "valid_cached" / "0.1.0"
        model_dir.mkdir(parents=True)
        (model_dir / "v.onnx").write_bytes(content)
        path = registry._ensure_model(spec, None)
        assert path.exists()
        assert path == model_dir / "v.onnx"

    def test_raises_on_sha_mismatch_after_download(
        self, registry: OnnxModelRegistry, cache_dir: Path
    ) -> None:
        spec = ModelSpec(
            name="sha_mismatch",
            url="https://example.com/x.onnx",
            sha256="0" * 64,
            filename="x.onnx",
        )
        registry.register(spec)

        def fake_download(s, target, cb):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"wrong content")

        with patch.object(registry, "_download", side_effect=fake_download):
            with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
                registry._ensure_model(spec, None)

    def test_returns_path_after_successful_download(
        self, registry: OnnxModelRegistry, cache_dir: Path
    ) -> None:
        content = b"good model data"
        sha = hashlib.sha256(content).hexdigest()
        spec = ModelSpec(
            name="dl_good",
            url="https://example.com/g.onnx",
            sha256=sha,
            filename="g.onnx",
        )
        registry.register(spec)

        def fake_download(s, target, cb):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        with patch.object(registry, "_download", side_effect=fake_download):
            path = registry._ensure_model(spec, None)
        assert path.exists()
        assert path.read_bytes() == content


# ---------------------------------------------------------------------------
# _download — success path (lines 255-268)
# ---------------------------------------------------------------------------

class TestDownloadSuccess:
    def test_download_writes_file(
        self, registry: OnnxModelRegistry, cache_dir: Path
    ) -> None:
        content = b"onnx model bytes"
        spec = ModelSpec(
            name="dl_test",
            url="https://example.com/dl.onnx",
            sha256="placeholder",
            filename="dl.onnx",
        )
        target = cache_dir / "dl_test" / "0.1.0" / "dl.onnx"
        target.parent.mkdir(parents=True)

        class _FakeResp:
            headers = {"Content-Length": str(len(content))}
            _done = False

            def read(self, size: int) -> bytes:
                if not self._done:
                    self._done = True
                    return content
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        with patch("urllib.request.urlopen", return_value=_FakeResp()):
            registry._download(spec, target, None)

        assert target.exists()
        assert target.read_bytes() == content

    def test_download_with_progress_callback(
        self, registry: OnnxModelRegistry, cache_dir: Path
    ) -> None:
        content = b"x" * 128
        spec = ModelSpec(
            name="dl_progress",
            url="https://example.com/p.onnx",
            sha256="placeholder",
            filename="p.onnx",
        )
        target = cache_dir / "dl_progress" / "0.1.0" / "p.onnx"
        target.parent.mkdir(parents=True)

        calls: list[str] = []

        class _FakeResp:
            headers = {"Content-Length": str(len(content))}
            _done = False

            def read(self, size: int) -> bytes:
                if not self._done:
                    self._done = True
                    return content
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        with patch("urllib.request.urlopen", return_value=_FakeResp()):
            registry._download(spec, target, lambda p: calls.append(p.message))

        assert any("Downloading" in c for c in calls)
