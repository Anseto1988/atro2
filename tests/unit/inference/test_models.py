import torch
import pytest
from pathlib import Path
from astroai.inference.models import ModelRegistry


@pytest.fixture()
def registry():
    return ModelRegistry()


@pytest.fixture()
def pth_file(tmp_path):
    p = tmp_path / "test_model.pth"
    torch.save({"key": torch.tensor([1.0])}, p)
    return p


def test_register_and_get(registry, pth_file):
    registry.register("m1", pth_file)
    data = registry.get("m1")
    assert "key" in data
    assert torch.equal(data["key"], torch.tensor([1.0]))


def test_get_raises_keyerror_for_unknown(registry):
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_list_models(registry, pth_file):
    registry.register("alpha", pth_file)
    registry.register("beta", pth_file)
    names = registry.list_models()
    assert "alpha" in names
    assert "beta" in names


def test_unregister(registry, pth_file):
    registry.register("m1", pth_file)
    registry.unregister("m1")
    assert "m1" not in registry.list_models()


def test_reload(registry, tmp_path):
    p = tmp_path / "model.pth"
    torch.save({"v": torch.tensor([1.0])}, p)
    registry.register("m", p)
    assert torch.equal(registry.get("m")["v"], torch.tensor([1.0]))
    torch.save({"v": torch.tensor([99.0])}, p)
    registry.reload("m")
    assert torch.equal(registry.get("m")["v"], torch.tensor([99.0]))


def test_unsupported_format_raises_valueerror(registry, tmp_path):
    p = tmp_path / "model.xyz"
    p.write_bytes(b"fake")
    with pytest.raises(ValueError, match="Unsupported"):
        registry.register("bad", p)
