from __future__ import annotations

"""Evaluation metrics and visualisation.

Provides RMSE, MAE, R^2, anomaly correlation coefficient (ACC),
and comparison plots for baselines vs. GRU.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


# ─── Scalar Metrics ─────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics.

    Returns dict with keys: rmse, mae, r2, acc.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    acc = anomaly_correlation(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2, "acc": acc}


def anomaly_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Anomaly Correlation Coefficient.

    ACC = corr(predicted_anomaly, true_anomaly).
    Standard metric in seasonal climate forecasting.
    """
    if len(y_true) < 3:
        return float("nan")
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


# ─── Formatted Output ──────────────────────────────────────────────

def print_summary_table(results: dict[str, dict]) -> None:
    """Print a formatted model comparison table.

    Parameters
    ----------
    results : dict
        {model_name: {rmse, mae, r2, acc}} for each model.
    """
    header = f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'ACC':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<25} "
            f"{m.get('rmse', float('nan')):>8.4f} "
            f"{m.get('mae', float('nan')):>8.4f} "
            f"{m.get('r2', float('nan')):>8.4f} "
            f"{m.get('acc', float('nan')):>8.4f}"
        )
    print("=" * len(header) + "\n")


# ─── Plots ──────────────────────────────────────────────────────────

def plot_prediction_timeseries(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_index: pd.DatetimeIndex,
    title: str = "Predictions vs Actual",
    save_path: str | Path | None = None,
) -> None:
    """Line plot of true vs predicted values over time."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_index, y_true, label="Actual", color="black", linewidth=1.5)
    ax.plot(time_index, y_pred, label="Predicted", color="tab:red", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature Anomaly (K)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    plt.close(fig)


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    save_path: str | Path | None = None,
) -> None:
    """Scatter plot with 1:1 line and R^2 annotation."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)

    # 1:1 reference line
    lims = [
        min(y_true.min(), y_pred.min()) - 0.5,
        max(y_true.max(), y_pred.max()) + 0.5,
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    r2 = r2_score(y_true, y_pred)
    ax.annotate(f"R$^2$ = {r2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)

    ax.set_xlabel("Actual Anomaly (K)")
    ax.set_ylabel("Predicted Anomaly (K)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    plt.close(fig)


def plot_january_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    years: np.ndarray,
    title: str = "January Predictions",
    save_path: str | Path | None = None,
) -> None:
    """Bar chart comparing true vs predicted January anomalies by year."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(years))
    width = 0.35

    ax.bar(x - width / 2, y_true, width, label="Actual", color="steelblue")
    ax.bar(x + width / 2, y_pred, width, label="Predicted", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Temperature Anomaly (K)")
    ax.set_title(title)
    ax.legend()
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    plt.close(fig)


def plot_baseline_comparison(
    results: dict[str, dict],
    metric: str = "rmse",
    title: str = "Model Comparison",
    save_path: str | Path | None = None,
) -> None:
    """Bar chart comparing a metric across models."""
    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#888888"] * len(names)
    # Highlight the GRU bar
    for i, name in enumerate(names):
        if "gru" in name.lower():
            colors[i] = "tab:red"

    ax.barh(names, values, color=colors)
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    plt.close(fig)
