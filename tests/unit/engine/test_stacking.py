import numpy as np
import pytest

from astroai.engine.stacking import FrameStacker


@pytest.fixture()
def stacker():
    return FrameStacker()


@pytest.fixture()
def grayscale_frames():
    np.random.seed(42)
    base = 100.0
    return [
        np.full((50, 50), base) + np.random.normal(0, 5, (50, 50))
        for _ in range(10)
    ]


@pytest.fixture()
def color_frames():
    np.random.seed(42)
    base = 100.0
    return [
        np.full((50, 50, 3), base)
        + np.random.normal(0, 5, (50, 50, 3))
        for _ in range(10)
    ]


def test_stack_mean(stacker, grayscale_frames):
    result = stacker.stack_mean(grayscale_frames)
    assert result.shape == (50, 50)
    assert abs(result.mean() - 100.0) < 1.0


def test_stack_median(stacker, grayscale_frames):
    result = stacker.stack_median(grayscale_frames)
    assert result.shape == (50, 50)
    assert abs(result.mean() - 100.0) < 1.0


def test_stack_sigma_clip_rejects_outlier(stacker):
    np.random.seed(42)
    base = 100.0
    frames = [
        np.full((50, 50), base) + np.random.normal(0, 5, (50, 50))
        for _ in range(10)
    ]
    outlier = np.full((50, 50), 10000.0)
    frames.append(outlier)
    result = stacker.stack_sigma_clip(frames)
    assert result.shape == (50, 50)
    assert abs(result.mean() - 100.0) < 5.0


def test_stack_dispatcher_mean(stacker, grayscale_frames):
    result = stacker.stack(grayscale_frames, method="mean")
    expected = stacker.stack_mean(grayscale_frames)
    np.testing.assert_array_equal(result, expected)


def test_stack_dispatcher_median(stacker, grayscale_frames):
    result = stacker.stack(grayscale_frames, method="median")
    expected = stacker.stack_median(grayscale_frames)
    np.testing.assert_array_equal(result, expected)


def test_stack_dispatcher_sigma_clip(stacker, grayscale_frames):
    result = stacker.stack(grayscale_frames, method="sigma_clip")
    expected = stacker.stack_sigma_clip(grayscale_frames)
    np.testing.assert_array_equal(result, expected)


def test_stack_unknown_method(stacker, grayscale_frames):
    with pytest.raises(ValueError, match="Unknown method"):
        stacker.stack(grayscale_frames, method="unknown")


def test_validate_empty_list(stacker):
    with pytest.raises(ValueError, match="No frames provided"):
        stacker.stack_mean([])


def test_validate_mismatched_shapes(stacker):
    frames = [
        np.zeros((50, 50)),
        np.zeros((60, 60)),
    ]
    with pytest.raises(ValueError, match="does not match"):
        stacker.stack_mean(frames)


def test_mean_color_frames(stacker, color_frames):
    result = stacker.stack_mean(color_frames)
    assert result.shape == (50, 50, 3)
    assert abs(result.mean() - 100.0) < 1.0


def test_median_color_frames(stacker, color_frames):
    result = stacker.stack_median(color_frames)
    assert result.shape == (50, 50, 3)
    assert abs(result.mean() - 100.0) < 1.0
