from __future__ import annotations

"""Feature engineering: lag creation and temporal windowing for GRU.

Fixes three issues simultaneously:

Issue #3 (double-lagging):
  OLD: Lags applied via xr.shift() in preprocessing, then
       series_to_supervised() created another 6-month window.
       Features were effectively 7-12 months back, not 1-6.
  NEW: Lags applied exactly ONCE here.

Issue #5 (GRU input shape):
  OLD: (n_samples, 1, 30) -- 30 features crammed into 1 timestep.
       GRU had no temporal sequence to learn from.
  NEW: (n_samples, 6, n_features) -- 6 real timesteps, each with
       all features.  GRU can learn temporal evolution.

Integrates with Issue #2 fix:
  Feature selection is done on anomalies, and this module consumes
  only the selected features.
"""

import logging

import numpy as np
import pandas as pd

from config import N_TIMESTEPS

logger = logging.getLogger(__name__)


def create_gru_sequences(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    n_timesteps: int = N_TIMESTEPS,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build 3-D input arrays for GRU from flat feature DataFrame.

    For each target at time *t*, the input is the feature matrix from
    times ``[t - n_timesteps, ..., t - 1]``.  This applies lags
    exactly **once** (fixing Issue #3).

    Parameters
    ----------
    features_df : pd.DataFrame
        Anomaly features with DatetimeIndex.  Should contain only
        the selected features (post feature-selection).
    target_series : pd.Series
        Anomaly target with DatetimeIndex.
    n_timesteps : int
        Number of past months to include per sample (sequence length).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_timesteps, n_features)
    y : np.ndarray, shape (n_samples,)
    time_index : pd.DatetimeIndex
        Timestamps corresponding to each sample (the target time).
    """
    # Align indices
    common_idx = features_df.index.intersection(target_series.index)
    features_df = features_df.loc[common_idx]
    target_series = target_series.loc[common_idx]

    feat_values = features_df.values   # (T, n_features)
    tgt_values = target_series.values  # (T,)

    n_total = len(common_idx)
    n_features = feat_values.shape[1]

    X_list = []
    y_list = []
    idx_list = []

    for t in range(n_timesteps, n_total):
        # Input: features from [t - n_timesteps, ..., t - 1]
        X_list.append(feat_values[t - n_timesteps : t, :])
        # Target: value at time t
        y_list.append(tgt_values[t])
        idx_list.append(common_idx[t])

    X = np.array(X_list, dtype=np.float32)   # (n_samples, n_timesteps, n_features)
    y = np.array(y_list, dtype=np.float32)   # (n_samples,)
    time_index = pd.DatetimeIndex(idx_list)

    logger.info(
        "GRU sequences: X=%s, y=%s, period=%s to %s",
        X.shape, y.shape, time_index[0].strftime("%Y-%m"), time_index[-1].strftime("%Y-%m"),
    )
    return X, y, time_index


def split_by_date(
    X: np.ndarray,
    y: np.ndarray,
    time_index: pd.DatetimeIndex,
    train_end: str,
    val_end: str,
) -> dict:
    """Split arrays into train / val / test by date boundaries.

    Parameters
    ----------
    X, y : arrays from :func:`create_gru_sequences`.
    time_index : corresponding timestamps.
    train_end : str
        Last month of training period, e.g. "2014-12".
    val_end : str
        Last month of validation period, e.g. "2019-12".

    Returns
    -------
    dict with keys:
        X_train, y_train, idx_train,
        X_val, y_val, idx_val,
        X_test, y_test, idx_test
    """
    train_mask = time_index <= pd.Timestamp(train_end)
    val_mask = (time_index > pd.Timestamp(train_end)) & (time_index <= pd.Timestamp(val_end))
    test_mask = time_index > pd.Timestamp(val_end)

    splits = {
        "X_train": X[train_mask], "y_train": y[train_mask], "idx_train": time_index[train_mask],
        "X_val":   X[val_mask],   "y_val":   y[val_mask],   "idx_val":   time_index[val_mask],
        "X_test":  X[test_mask],  "y_test":  y[test_mask],  "idx_test":  time_index[test_mask],
    }

    for name in ["train", "val", "test"]:
        n = splits[f"X_{name}"].shape[0]
        logger.info("  %s: %d samples", name, n)

    return splits
