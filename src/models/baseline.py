"""Baseline models to benchmark GRU against.

The current project has no baselines, making it impossible to know
if the GRU adds value.  These three baselines establish a hierarchy:

  climatology (trivial)
    < persistence (temporal autocorrelation)
      < Ridge regression (linear feature relationships)
        < GRU (non-linear temporal)

Each step up must show significant improvement or it is not justified.
"""

import logging

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from config import RIDGE_ALPHA

logger = logging.getLogger(__name__)


class ClimatologyBaseline:
    """Predict the training-period mean anomaly (i.e. zero).

    Since anomalies are centered around zero by construction,
    RMSE = std(test anomalies).  If the GRU can't beat this,
    it is learning nothing useful.
    """

    def __init__(self):
        self.mean_anomaly = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ClimatologyBaseline":
        self.mean_anomaly = float(np.mean(y_train))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.mean_anomaly)


class PersistenceBaseline:
    """Predict this month's anomaly = previous same-month anomaly.

    For a 6-month lag setup this simplifies to: predict the anomaly
    of the most recent input timestep (the last value in the sequence).

    This is a temporal baseline -- the GRU should beat this to
    justify learning recurrent patterns.
    """

    def __init__(self):
        self._target_idx = -1  # last value in last timestep

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PersistenceBaseline":
        # No parameters to learn
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the mean of the last timestep features as a naive prediction."""
        # X shape: (n_samples, n_timesteps, n_features)
        # Use the target-correlated feature at the most recent timestep
        # Fallback: mean of last timestep
        if X.ndim == 3:
            return X[:, -1, :].mean(axis=1)
        return X[:, -1]


class LinearRegressionBaseline:
    """Ridge regression on the flattened feature set.

    If the GRU can't significantly beat this, the non-linearity
    isn't needed for this problem.

    Uses Ridge (L2) because n_features may be comparable to n_samples.
    """

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LinearRegressionBaseline":
        X_flat = self._flatten(X_train)
        self.model.fit(X_flat, y_train)
        logger.info("Ridge baseline fitted. n_features=%d", X_flat.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X)
        return self.model.predict(X_flat)

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """(n, timesteps, features) -> (n, timesteps * features)."""
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X
