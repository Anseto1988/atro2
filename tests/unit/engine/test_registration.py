import numpy as np
import pytest

from astroai.engine.registration import FrameAligner


@pytest.fixture()
def aligner():
    return FrameAligner(upsample_factor=10)


def _shift_no_wrap(img, dy, dx):
    out = np.zeros_like(img)
    sy = slice(max(dy, 0), img.shape[0] + min(dy, 0))
    sx = slice(max(dx, 0), img.shape[1] + min(dx, 0))
    oy = slice(max(-dy, 0), img.shape[0] + min(-dy, 0))
    ox = slice(max(-dx, 0), img.shape[1] + min(-dx, 0))
    out[sy, sx] = img[oy, ox]
    return out


@pytest.fixture()
def reference_frame():
    np.random.seed(42)
    y, x = np.mgrid[-50:50, -50:50]
    frame = np.exp(-(x**2 + y**2) / (2 * 10**2))
    frame += np.random.normal(0, 0.01, frame.shape)
    return frame


@pytest.fixture()
def shifted_frame(reference_frame):
    return _shift_no_wrap(reference_frame, 3, 5)


def test_align_returns_tuple(aligner, reference_frame, shifted_frame):
    result = aligner.align(reference_frame, shifted_frame)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_align_returns_image_and_matrix(
    aligner, reference_frame, shifted_frame
):
    aligned, transform = aligner.align(reference_frame, shifted_frame)
    assert isinstance(aligned, np.ndarray)
    assert isinstance(transform, np.ndarray)


def test_transform_matrix_is_3x3(
    aligner, reference_frame, shifted_frame
):
    _, transform = aligner.align(reference_frame, shifted_frame)
    assert transform.shape == (3, 3)


def test_detected_shift_approximately_correct(
    aligner, reference_frame, shifted_frame
):
    _, transform = aligner.align(reference_frame, shifted_frame)
    detected_dx = transform[0, 2]
    detected_dy = transform[1, 2]
    assert abs(abs(detected_dx) - 5.0) < 1.5
    assert abs(abs(detected_dy) - 3.0) < 1.5


def test_align_batch_returns_correct_length(
    aligner, reference_frame, shifted_frame
):
    targets = [shifted_frame, shifted_frame, shifted_frame]
    result = aligner.align_batch(reference_frame, targets)
    assert isinstance(result, list)
    assert len(result) == 3


def test_align_batch_elements_are_arrays(
    aligner, reference_frame, shifted_frame
):
    targets = [shifted_frame, shifted_frame]
    result = aligner.align_batch(reference_frame, targets)
    for img in result:
        assert isinstance(img, np.ndarray)
        assert img.shape == reference_frame.shape


def test_grayscale_frame(aligner):
    np.random.seed(42)
    ref = np.random.rand(100, 100)
    tgt = np.roll(ref, 3, axis=1)
    aligned, transform = aligner.align(ref, tgt)
    assert aligned.shape == ref.shape
    assert transform.shape == (3, 3)


def test_rgb_frame(aligner):
    np.random.seed(42)
    ref = np.random.rand(100, 100, 3)
    tgt = np.roll(ref, 3, axis=1)
    aligned, transform = aligner.align(ref, tgt)
    assert aligned.shape == ref.shape
    assert transform.shape == (3, 3)


def test_no_subpixel_align_upsample_factor_1() -> None:
    """upsample_factor=1 hits the else branch in _phase_correlate (lines 68-77)."""
    from astroai.engine.registration.aligner import FrameAligner
    rng = np.random.default_rng(55)
    ref = rng.random((64, 64)).astype(np.float64)
    tgt = np.roll(ref, 5, axis=1)
    aligner = FrameAligner(upsample_factor=1)
    aligned, transform = aligner.align(ref, tgt)
    assert aligned.shape == ref.shape
    assert transform.shape == (3, 3)


def test_no_subpixel_large_vertical_shift_unwraps() -> None:
    """shift_y > h//2 triggers shift_y -= h in the else branch (line 75)."""
    from astroai.engine.registration.aligner import FrameAligner
    rng = np.random.default_rng(77)
    ref = rng.random((64, 64)).astype(np.float64)
    # Roll by >32 rows so phase correlation peak is in upper half → needs unwrap
    tgt = np.roll(ref, 40, axis=0)
    aligner = FrameAligner(upsample_factor=1)
    aligned, transform = aligner.align(ref, tgt)
    assert aligned.shape == ref.shape
