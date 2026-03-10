"""Monthly anomaly computation.

Fixes Issue #10 (seasonal cycle domination) -- THE ROOT CAUSE:

  OLD: Model trained on raw temperatures across all 12 months.
       R^2 = 0.98 overall but R^2 = -0.61 for January.
       The model just learned "summer is hot, winter is cold"
       (the seasonal cycle) instead of inter-annual variability.

  NEW: Subtract monthly climatology (computed from training data only)
       before modeling.  The model now predicts *deviations from normal*
       (anomalies), which is what we actually care about: "will this
       January be warmer or colder than a typical January?"

Critical: climatology is fitted on the training period only.
Applying a climatology derived from all data (including test) would
leak future information.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AnomalyTransformer:
    """Compute and apply monthly climatology for anomaly conversion.

    Usage
    -----
    >>> at = AnomalyTransformer()
    >>> at.fit(train_series, train_series.index)
    >>> anomalies = at.transform(full_series, full_series.index)
    >>> raw_back  = at.inverse_transform(anomalies, full_series.index)
    """

    def __init__(self):
        self.climatology: pd.Series | None = None  # indexed 1..12
        self._is_fitted = False

    def fit(
        self, values: pd.Series | np.ndarray, time_index: pd.DatetimeIndex
    ) -> "AnomalyTransformer":
        """Compute monthly climatology from training data.

        Parameters
        ----------
        values : array-like
            Raw values for the training period.
        time_index : pd.DatetimeIndex
            Corresponding timestamps (must have ``.month``).
        """
        s = pd.Series(np.asarray(values), index=time_index)
        self.climatology = s.groupby(s.index.month).mean()
        self.climatology.index.name = "month"
        self._is_fitted = True
        logger.debug("Fitted climatology: %s", self.climatology.to_dict())
        return self

    def transform(
        self, values: pd.Series | np.ndarray, time_index: pd.DatetimeIndex
    ) -> np.ndarray:
        """Subtract climatology to obtain anomalies.

        Can be applied to train, val, or test data -- but the
        climatology was fitted on training data only.
        """
        self._check_fitted()
        arr = np.asarray(values, dtype=np.float64)
        months = time_index.month
        clim_vals = self.climatology.loc[months].values
        return arr - clim_vals

    def inverse_transform(
        self, anomalies: np.ndarray, time_index: pd.DatetimeIndex
    ) -> np.ndarray:
        """Add climatology back to convert anomalies to raw values."""
        self._check_fitted()
        arr = np.asarray(anomalies, dtype=np.float64)
        months = time_index.month
        clim_vals = self.climatology.loc[months].values
        return arr + clim_vals

    # ── persistence ─────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist climatology."""
        joblib.dump({"climatology": self.climatology}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyTransformer":
        """Load persisted climatology."""
        data = joblib.load(path)
        obj = cls()
        obj.climatology = data["climatology"]
        obj._is_fitted = True
        return obj

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "AnomalyTransformer has not been fitted. Call .fit() first."
            )


def compute_anomalies_dataframe(
    df: pd.DataFrame,
    train_mask: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, dict[str, AnomalyTransformer]]:
    """Convert every column of *df* to anomalies.

    Parameters
    ----------
    df : pd.DataFrame
        Raw values with DatetimeIndex.
    train_mask : boolean array
        True for rows belonging to the training period.

    Returns
    -------
    df_anom : pd.DataFrame
        Anomaly values (same shape as input).
    transformers : dict
        Column name -> fitted AnomalyTransformer (for later inverse).
    """
    transformers: dict[str, AnomalyTransformer] = {}
    anom_cols: dict[str, np.ndarray] = {}

    for col in df.columns:
        at = AnomalyTransformer()
        at.fit(df.loc[train_mask, col].values, df.index[train_mask])
        anom_cols[col] = at.transform(df[col].values, df.index)
        transformers[col] = at

    df_anom = pd.DataFrame(anom_cols, index=df.index)
    logger.info("Converted %d columns to anomalies", len(df.columns))
    return df_anom, transformers
