import torch
import pytest
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
