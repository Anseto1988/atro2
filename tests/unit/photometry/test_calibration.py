from __future__ import annotations

import numpy as np
import pytest

from astroai.engine.photometry.calibration import MagnitudeCalibrator


class TestMagnitudeCalibrator:
    def test_fit_and_predict_linear(self) -> None:
        instr = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        catalog = np.array([8.0, 9.0, 10.0, 11.0, 12.0])

        cal = MagnitudeCalibrator(alpha=0.0001)
        cal.fit(instr, catalog)
        predicted = cal.predict(instr)

        np.testing.assert_allclose(predicted, catalog, atol=0.1)

    def test_r_squared_high_on_clean_data(self) -> None:
        instr = np.linspace(8.0, 15.0, 20)
        catalog = instr - 2.0

        cal = MagnitudeCalibrator(alpha=0.0001)
        cal.fit(instr, catalog)

        assert cal.r_squared >= 0.95

    def test_r_squared_is_zero_before_fit(self) -> None:
        cal = MagnitudeCalibrator()
        assert cal.r_squared == 0.0

    def test_predict_without_fit_raises(self) -> None:
        cal = MagnitudeCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.predict(np.array([10.0, 11.0]))

    def test_fit_returns_self(self) -> None:
        cal = MagnitudeCalibrator()
        instr = np.array([10.0, 11.0, 12.0])
        catalog = np.array([8.0, 9.0, 10.0])
        result = cal.fit(instr, catalog)
        assert result is cal

    def test_fit_with_noise(self) -> None:
        rng = np.random.default_rng(42)
        instr = np.linspace(8.0, 15.0, 50)
        catalog = instr - 2.0 + rng.normal(0, 0.1, 50)

        cal = MagnitudeCalibrator(alpha=0.01)
        cal.fit(instr, catalog)

        assert cal.r_squared > 0.9
        predicted = cal.predict(instr)
        residuals = np.abs(predicted - catalog)
        assert np.mean(residuals) < 0.2
