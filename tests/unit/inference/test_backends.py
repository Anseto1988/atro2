from unittest.mock import patch

import pytest
import torch

from astroai.inference.backends import DeviceManager


@pytest.fixture(autouse=True)
def reset_singleton():
    DeviceManager._instance = None
    yield
    DeviceManager._instance = None


def test_singleton_pattern():
    a = DeviceManager()
    b = DeviceManager()
    assert a is b


def test_get_device_returns_torch_device():
    dm = DeviceManager()
    assert isinstance(dm.get_device(), torch.device)


def test_device_info_keys():
    dm = DeviceManager()
    info = dm.device_info()
    expected = {"type", "name", "cuda_available", "mps_available"}
    assert expected.issubset(info.keys())


def test_to_device_moves_tensor():
    dm = DeviceManager()
    t = torch.tensor([1.0, 2.0, 3.0])
    result = dm.to_device(t)
    assert result.device.type == dm.get_device().type


# ---------------------------------------------------------------------------
# _resolve_device: MPS path (line 26-27) and CPU path (line 28)
# ---------------------------------------------------------------------------

def test_resolve_device_returns_mps_when_available():
    """Line 26-27: when MPS is available and CUDA is not, return mps device."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True):
        device = DeviceManager._resolve_device()
    assert device == torch.device("mps")


def test_resolve_device_returns_cpu_when_no_accelerator():
    """Line 28: when neither CUDA nor MPS is available, return cpu device."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        device = DeviceManager._resolve_device()
    assert device == torch.device("cpu")


def test_singleton_uses_mps_when_available():
    """DeviceManager singleton resolves to mps on Apple Silicon (mocked)."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True):
        dm = DeviceManager()
    assert dm.get_device() == torch.device("mps")
