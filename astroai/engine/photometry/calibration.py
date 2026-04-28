from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


class MagnitudeCalibrator:
    def __init__(self, alpha: float = 1.0) -> None:
        self._model = Ridge(alpha=alpha)
        self._r_squared: float = 0.0
        self._fitted = False

    def fit(
        self,
        instr_mags: NDArray[np.floating],
        catalog_mags: NDArray[np.floating],
    ) -> MagnitudeCalibrator:
        X = np.asarray(instr_mags, dtype=float).reshape(-1, 1)
        y = np.asarray(catalog_mags, dtype=float)
        self._model.fit(X, y)
        self._r_squared = float(self._model.score(X, y))
        self._fitted = True
        return self

    def predict(self, instr_mags: NDArray[np.floating]) -> NDArray[np.floating]:
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted yet")
        X = np.asarray(instr_mags, dtype=float).reshape(-1, 1)
        return self._model.predict(X)

    @property
    def r_squared(self) -> float:
        return self._r_squared
