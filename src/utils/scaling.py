"""Leak-free scaling utilities.

Fixes Issue #4 (scaler fitted on all data) and Issue #9 (inconsistent
inverse transform) by:
  - Using separate scalers for features (X) and target (y).
  - Fitting only on training data.
  - Providing clean inverse_y() for predictions.

Uses StandardScaler instead of MinMaxScaler because anomaly data is
naturally centered near zero; StandardScaler is also less sensitive
to outliers in small datasets.
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class FeatureTargetScaler:
    """Manages separate scalers for X (features) and y (target)."""

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._is_fitted = False

    # ── fit / transform ─────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "FeatureTargetScaler":
        """Fit scalers on training data only.

        Parameters
        ----------
        X_train : array of shape (n_samples, n_features)
            Training features (2-D; if 3-D, will be reshaped internally).
        y_train : array of shape (n_samples,) or (n_samples, 1)
            Training targets.
        """
        X_2d = self._to_2d(X_train)
        self.scaler_X.fit(X_2d)
        self.scaler_y.fit(y_train.reshape(-1, 1))
        self._is_fitted = True
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform features.  Preserves original ndim (2-D or 3-D)."""
        self._check_fitted()
        orig_shape = X.shape
        X_2d = self._to_2d(X)
        X_scaled = self.scaler_X.transform(X_2d)
        return X_scaled.reshape(orig_shape)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Transform target."""
        self._check_fitted()
        return self.scaler_y.transform(y.reshape(-1, 1)).ravel()

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform predictions back to anomaly scale."""
        self._check_fitted()
        return self.scaler_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).ravel()

    # ── persistence ─────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist both scalers to a single file."""
        path = Path(path)
        joblib.dump(
            {"scaler_X": self.scaler_X, "scaler_y": self.scaler_y},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureTargetScaler":
        """Load persisted scalers."""
        data = joblib.load(path)
        obj = cls()
        obj.scaler_X = data["scaler_X"]
        obj.scaler_y = data["scaler_y"]
        obj._is_fitted = True
        return obj

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _to_2d(X: np.ndarray) -> np.ndarray:
        """Reshape (n, timesteps, features) -> (n, timesteps*features)."""
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call .fit() first.")
