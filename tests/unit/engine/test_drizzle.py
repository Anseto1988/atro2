"""Unit tests for DrizzleEngine."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from astroai.astrometry.catalog import WcsSolution
from astroai.engine.drizzle.engine import DrizzleEngine


@pytest.fixture()
def identity_wcs() -> WcsSolution:
    return WcsSolution(
        ra_center=0.0,
        dec_center=0.0,
        pixel_scale_arcsec=1.0,
        rotation_deg=0.0,
        fov_width_deg=256 / 3600.0,
        fov_height_deg=256 / 3600.0,
        cd_matrix=(1.0 / 3600.0, 0.0, 0.0, 1.0 / 3600.0),
        crpix1=128.0,
        crpix2=128.0,
    )


@pytest.fixture()
def synthetic_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(100.0, 500.0, (256, 256)).astype(np.float32)


class TestDrizzleEngine:
    def test_drizzle_identity(self, synthetic_frame, identity_wcs):
        """Single frame without WCS offset preserves image statistics."""
        engine = DrizzleEngine(drop_size=1.0, pixfrac=1.0, scale=1.0)
        result = engine.drizzle(
            [synthetic_frame], [identity_wcs], output_shape=(256, 256)
        )

        assert result.shape == (256, 256)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        covered = result > 0
        assert covered.sum() > 0
        assert abs(result[covered].mean() - synthetic_frame.mean()) / synthetic_frame.mean() < 0.5

    @pytest.mark.parametrize("drop_size", [0.5, 0.7, 1.0])
    def test_drop_sizes(self, synthetic_frame, identity_wcs, drop_size):
        """All valid drop sizes produce finite output."""
        engine = DrizzleEngine(drop_size=drop_size, pixfrac=1.0, scale=1.0)
        result = engine.drizzle(
            [synthetic_frame], [identity_wcs], output_shape=(256, 256)
        )

        assert result.shape == (256, 256)
        assert np.isfinite(result).all()

    @pytest.mark.parametrize("pixfrac", [0.1, 0.3, 0.5, 0.7, 1.0])
    def test_pixfrac_range(self, synthetic_frame, identity_wcs, pixfrac):
        """Pixfrac values in (0, 1] produce valid output arrays."""
        engine = DrizzleEngine(drop_size=1.0, pixfrac=pixfrac, scale=1.0)
        result = engine.drizzle(
            [synthetic_frame], [identity_wcs], output_shape=(256, 256)
        )

        assert result.shape == (256, 256)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_scale_output_shape(self, synthetic_frame, identity_wcs):
        """Scale=2.0 doubles the output dimension."""
        engine = DrizzleEngine(drop_size=1.0, pixfrac=1.0, scale=2.0)
        result = engine.drizzle(
            [synthetic_frame], [identity_wcs], output_shape=(512, 512)
        )

        assert result.shape == (512, 512)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()

    def test_wcs_alignment(self):
        """Two frames with known WCS offset align correctly."""
        frame = np.zeros((64, 64), dtype=np.float32)
        frame[30:34, 30:34] = 1000.0

        wcs_ref = WcsSolution(
            ra_center=0.0, dec_center=0.0,
            pixel_scale_arcsec=1.0, rotation_deg=0.0,
            fov_width_deg=64 / 3600.0, fov_height_deg=64 / 3600.0,
            cd_matrix=(1.0 / 3600.0, 0.0, 0.0, 1.0 / 3600.0),
            crpix1=32.0, crpix2=32.0,
        )

        wcs_shifted = WcsSolution(
            ra_center=0.0, dec_center=0.0,
            pixel_scale_arcsec=1.0, rotation_deg=0.0,
            fov_width_deg=64 / 3600.0, fov_height_deg=64 / 3600.0,
            cd_matrix=(1.0 / 3600.0, 0.0, 0.0, 1.0 / 3600.0),
            crpix1=34.0, crpix2=34.0,
        )

        engine = DrizzleEngine(drop_size=1.0, pixfrac=1.0, scale=1.0)
        result = engine.drizzle(
            [frame, frame], [wcs_ref, wcs_shifted], output_shape=(64, 64)
        )

        assert result.shape == (64, 64)
        assert result.max() > 0.0
        peak_y, peak_x = np.unravel_index(result.argmax(), result.shape)
        assert 28 <= peak_y <= 35
        assert 28 <= peak_x <= 35

    def test_invalid_drop_size_raises(self):
        """Invalid drop_size raises ValueError."""
        with pytest.raises(ValueError, match="drop_size must be one of"):
            DrizzleEngine(drop_size=0.3)

    def test_invalid_pixfrac_raises(self):
        """Invalid pixfrac raises ValueError."""
        with pytest.raises(ValueError, match="pixfrac must be in"):
            DrizzleEngine(drop_size=1.0, pixfrac=0.0)

    def test_empty_frames_raises(self, identity_wcs):
        """Empty frame list raises ValueError."""
        engine = DrizzleEngine(drop_size=1.0, pixfrac=1.0, scale=1.0)
        with pytest.raises(ValueError, match="No frames provided"):
            engine.drizzle([], [], output_shape=(64, 64))

    def test_invalid_scale_raises(self):
        """Non-positive scale raises ValueError."""
        with pytest.raises(ValueError, match="scale must be positive"):
            DrizzleEngine(drop_size=1.0, pixfrac=1.0, scale=0.0)
        with pytest.raises(ValueError, match="scale must be positive"):
            DrizzleEngine(drop_size=1.0, pixfrac=1.0, scale=-1.0)

    def test_mismatched_frame_wcs_count_raises(self, identity_wcs):
        """Mismatched frame/WCS count raises ValueError."""
        engine = DrizzleEngine(drop_size=1.0, pixfrac=1.0, scale=1.0)
        frame = np.zeros((64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="Frame count"):
            engine.drizzle([frame, frame], [identity_wcs], output_shape=(64, 64))

    def test_color_frames(self, identity_wcs):
        """Drizzle handles 3-channel color frames."""
        rng = np.random.default_rng(7)
        frame = rng.uniform(100.0, 500.0, (64, 64, 3)).astype(np.float32)
        wcs = WcsSolution(
            ra_center=0.0, dec_center=0.0,
            pixel_scale_arcsec=1.0, rotation_deg=0.0,
            fov_width_deg=64 / 3600.0, fov_height_deg=64 / 3600.0,
            cd_matrix=(1.0 / 3600.0, 0.0, 0.0, 1.0 / 3600.0),
            crpix1=32.0, crpix2=32.0,
        )
        engine = DrizzleEngine(drop_size=1.0, pixfrac=1.0, scale=1.0)
        result = engine.drizzle([frame], [wcs], output_shape=(64, 64))

        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        covered = result.sum(axis=2) > 0
        assert covered.sum() > 0

    def test_two_panel_mosaic_overlap(self):
        """Two-panel mosaic with >10% overlap is assembled correctly.

        Frame A maps to output columns [0, w).
        Frame B is shifted right by (w - overlap_px), mapping to [w-overlap, 2w-overlap).
        Overlap region: columns [w-overlap, w).
        """
        h, w = 64, 64
        rng = np.random.default_rng(100)
        frame_a = rng.uniform(200.0, 400.0, (h, w)).astype(np.float32)
        frame_b = rng.uniform(200.0, 400.0, (h, w)).astype(np.float32)

        overlap_px = 10
        shift = w - overlap_px

        wcs_a = WcsSolution(
            ra_center=0.0, dec_center=0.0,
            pixel_scale_arcsec=1.0, rotation_deg=0.0,
            fov_width_deg=w / 3600.0, fov_height_deg=h / 3600.0,
            cd_matrix=(1.0 / 3600.0, 0.0, 0.0, 1.0 / 3600.0),
            crpix1=w / 2.0, crpix2=h / 2.0,
        )
        wcs_b = WcsSolution(
            ra_center=0.0, dec_center=0.0,
            pixel_scale_arcsec=1.0, rotation_deg=0.0,
            fov_width_deg=w / 3600.0, fov_height_deg=h / 3600.0,
            cd_matrix=(1.0 / 3600.0, 0.0, 0.0, 1.0 / 3600.0),
            crpix1=w / 2.0 - shift, crpix2=h / 2.0,
        )

        out_w = 2 * w - overlap_px
        engine = DrizzleEngine(drop_size=1.0, pixfrac=1.0, scale=1.0)
        result = engine.drizzle(
            [frame_a, frame_b], [wcs_a, wcs_b], output_shape=(h, out_w)
        )

        assert result.shape == (h, out_w)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()

        left_only = result[:, :w - overlap_px]
        right_only = result[:, w:]
        overlap_region = result[:, w - overlap_px:w]

        assert left_only.max() > 0, "Left panel should have signal"
        assert right_only.max() > 0, "Right panel should have signal"
        assert overlap_region.max() > 0, "Overlap region should have signal from both"

    def test_properties_accessible(self, identity_wcs):
        """Engine properties return configured values."""
        engine = DrizzleEngine(drop_size=0.7, pixfrac=0.5, scale=2.0)
        assert engine.drop_size == 0.7
        assert engine.pixfrac == 0.5
        assert engine.scale == 2.0

    def test_pixel_overlap_correctness(self):
        """Static _pixel_overlap computes expected overlap areas."""
        assert DrizzleEngine._pixel_overlap(0.5, 0.5, 0.5, 0.0, 0.0) == pytest.approx(1.0)
        assert DrizzleEngine._pixel_overlap(1.5, 0.5, 0.5, 0.0, 0.0) == pytest.approx(0.0)
        assert DrizzleEngine._pixel_overlap(0.75, 0.5, 0.5, 0.0, 0.0) == pytest.approx(0.75)
