from __future__ import annotations

"""Feature selection: lag-correlation analysis, stepwise regression, VIF.

Fixes Issue #2 (feature selection on wrong month):
  OLD: Stepwise regression run on June data only (month==6),
       but model trained on all 12 months.
  NEW: With anomaly-based approach, the seasonal cycle is removed.
       Feature selection is done on anomalies across ALL months,
       which is statistically valid because monthly means are gone.

Refactored from notebook 03 cells 9-11, 23-24.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


# ─── Lag Correlation ────────────────────────────────────────────────

def compute_lag_correlations(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    max_lag: int = 6,
) -> pd.DataFrame:
    """Compute Pearson correlation between each feature and target at lags 1..max_lag.

    Both ``features_df`` and ``target_series`` should already be in
    *anomaly* space so the seasonal cycle does not dominate the
    correlation.

    Parameters
    ----------
    features_df : pd.DataFrame
        Anomaly features with DatetimeIndex.
    target_series : pd.Series
        Anomaly target with DatetimeIndex.
    max_lag : int
        Maximum lag in months to test.

    Returns
    -------
    pd.DataFrame
        Shape (n_features, max_lag).  Index = feature names,
        columns = lag values 1..max_lag.  Values = Pearson r.
    """
    results = {}

    for lag in range(1, max_lag + 1):
        shifted_target = target_series.shift(-lag)
        valid = shifted_target.dropna().index
        valid = valid.intersection(features_df.index)

        corrs = {}
        for col in features_df.columns:
            x = features_df.loc[valid, col]
            y = shifted_target.loc[valid]
            mask = x.notna() & y.notna()
            if mask.sum() > 10:
                corrs[col] = np.corrcoef(x[mask], y[mask])[0, 1]
            else:
                corrs[col] = np.nan

        results[lag] = corrs

    df = pd.DataFrame(results)
    df.columns.name = "lag"
    df.index.name = "feature"
    return df


def select_optimal_lag_per_feature(corr_df: pd.DataFrame) -> dict[str, int]:
    """For each feature, pick the lag with the highest absolute correlation.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Output of :func:`compute_lag_correlations`.

    Returns
    -------
    dict
        {feature_name: optimal_lag}
    """
    return corr_df.abs().idxmax(axis=1).to_dict()


# ─── Stepwise Regression ───────────────────────────────────────────

def stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    sl_enter: float = 0.05,
    sl_remove: float = 0.05,
    verbose: bool = True,
) -> list[str]:
    """Forward-backward stepwise regression for feature selection.

    Refactored from notebook 03 cell 9.

    Parameters
    ----------
    X : pd.DataFrame
        Candidate features.
    y : pd.Series
        Target variable.
    sl_enter : float
        P-value threshold for a variable to enter the model.
    sl_remove : float
        P-value threshold for a variable to be removed.
    verbose : bool
        Whether to log each selection step.

    Returns
    -------
    list of str
        Names of selected features.
    """
    initial_features: list[str] = []
    remaining = list(X.columns)
    best_features = list(initial_features)

    while remaining:
        # --- Forward step ---
        pvals = {}
        for feat in remaining:
            candidate = best_features + [feat]
            X_cand = sm.add_constant(X[candidate])
            try:
                model = sm.OLS(y, X_cand).fit()
                pvals[feat] = model.pvalues.iloc[-1]
            except Exception:
                pvals[feat] = 1.0

        if not pvals:
            break

        best_candidate = min(pvals, key=pvals.get)
        if pvals[best_candidate] < sl_enter:
            best_features.append(best_candidate)
            remaining.remove(best_candidate)
            if verbose:
                logger.info(
                    "  + Added '%s' (p=%.4f), %d features total",
                    best_candidate, pvals[best_candidate], len(best_features),
                )
        else:
            break  # no variable meets entry threshold

        # --- Backward step ---
        while len(best_features) > 1:
            X_sel = sm.add_constant(X[best_features])
            model = sm.OLS(y, X_sel).fit()
            worst_pval = model.pvalues.iloc[1:].max()
            if worst_pval >= sl_remove:
                worst_feat = model.pvalues.iloc[1:].idxmax()
                best_features.remove(worst_feat)
                remaining.append(worst_feat)
                if verbose:
                    logger.info(
                        "  - Removed '%s' (p=%.4f), %d features remain",
                        worst_feat, worst_pval, len(best_features),
                    )
            else:
                break

    logger.info("Stepwise selected %d features: %s", len(best_features), best_features)
    return best_features


# ─── VIF ────────────────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each column.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix (no constant column).

    Returns
    -------
    pd.DataFrame
        Columns: ['feature', 'VIF'].
    """
    X = df.values.astype(float)
    vif_data = []
    for i in range(X.shape[1]):
        vif_val = variance_inflation_factor(X, i)
        vif_data.append({"feature": df.columns[i], "VIF": vif_val})

    vif_df = pd.DataFrame(vif_data)
    logger.info("VIF:\n%s", vif_df.to_string(index=False))
    return vif_df


def remove_high_vif(
    df: pd.DataFrame,
    threshold: float = 10.0,
) -> list[str]:
    """Iteratively remove features with VIF > threshold.

    Returns the list of surviving feature names.
    """
    cols = list(df.columns)

    while True:
        if len(cols) <= 1:
            break
        vif_df = compute_vif(df[cols])
        max_vif = vif_df["VIF"].max()
        if max_vif > threshold:
            drop_feat = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
            logger.info("Dropping '%s' (VIF=%.2f > %.2f)", drop_feat, max_vif, threshold)
            cols.remove(drop_feat)
        else:
            break

    logger.info("Features after VIF removal: %s", cols)
    return cols
