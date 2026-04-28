"""Unit tests for ModelRegistry."""
from __future__ import annotations
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from astroai.inference.models.registry import ModelRegistry


class TestModelRegistryBasic:
    def test_list_models_empty(self) -> None:
        reg = ModelRegistry()
        assert reg.list_models() == []

    def test_get_unknown_raises_key_error(self) -> None:
        reg = ModelRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("unknown")

    def test_reload_unknown_raises_key_error(self) -> None:
        reg = ModelRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.reload("unknown")

    def test_unregister_unknown_is_noop(self) -> None:
        reg = ModelRegistry()
        reg.unregister("nonexistent")  # should not raise

    def test_register_stores_model(self, tmp_path: Path) -> None:
        reg = ModelRegistry()
        fake_model = MagicMock()
        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"fake")
        with patch.object(reg, "_load", return_value=fake_model):
            reg.register("mymodel", model_path)
        assert "mymodel" in reg.list_models()

    def test_get_returns_registered_model(self, tmp_path: Path) -> None:
        reg = ModelRegistry()
        fake_model = MagicMock()
        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"fake")
        with patch.object(reg, "_load", return_value=fake_model):
            reg.register("mymodel", model_path)
        assert reg.get("mymodel") is fake_model

    def test_unregister_removes_model(self, tmp_path: Path) -> None:
        reg = ModelRegistry()
        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"fake")
        with patch.object(reg, "_load", return_value=MagicMock()):
            reg.register("mymodel", model_path)
        reg.unregister("mymodel")
        assert "mymodel" not in reg.list_models()

    def test_reload_calls_load_again(self, tmp_path: Path) -> None:
        reg = ModelRegistry()
        model_path = tmp_path / "model.pth"
        model_path.write_bytes(b"fake")
        call_count = 0
        def _fake_load(path: Path) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return MagicMock()
        with patch.object(reg, "_load", side_effect=_fake_load):
            reg.register("mymodel", model_path)
            reg.reload("mymodel")
        assert call_count == 2

    def test_list_models_returns_all_names(self, tmp_path: Path) -> None:
        reg = ModelRegistry()
        for name in ("a", "b", "c"):
            p = tmp_path / f"{name}.pth"
            p.write_bytes(b"x")
            with patch.object(reg, "_load", return_value=MagicMock()):
                reg.register(name, p)
        assert sorted(reg.list_models()) == ["a", "b", "c"]


class TestModelRegistryLoad:
    def test_load_unsupported_format_raises(self, tmp_path: Path) -> None:
        reg = ModelRegistry()
        p = tmp_path / "model.txt"
        p.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported model format"):
            reg._load(p)

    def test_load_onnx_raises_when_ort_missing(self, tmp_path: Path) -> None:
        reg = ModelRegistry()
        p = tmp_path / "model.onnx"
        p.write_bytes(b"fake")
        import astroai.inference.models.registry as _mod
        orig_ort = _mod.ort
        try:
            _mod.ort = None
            with pytest.raises(RuntimeError, match="onnxruntime"):
                reg._load(p)
        finally:
            _mod.ort = orig_ort


class TestModelRegistryThreadSafety:
    def test_concurrent_register_and_list(self, tmp_path: Path) -> None:
        reg = ModelRegistry()
        errors: list[Exception] = []

        def _worker(name: str) -> None:
            try:
                p = tmp_path / f"{name}.pth"
                p.write_bytes(b"x")
                with patch.object(reg, "_load", return_value=MagicMock()):
                    reg.register(name, p)
                _ = reg.list_models()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(f"m{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(reg.list_models()) == 8
