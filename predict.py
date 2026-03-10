#!/usr/bin/env python3
from __future__ import annotations

"""Load a trained model and make predictions.

Usage
-----
    python predict.py                          # predict on saved test set
    python predict.py --model models/gru_final.keras
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils.scaling import FeatureTargetScaler
from src.preprocessing.anomaly import AnomalyTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("predict")


def load_artifacts(model_path: str | Path | None = None):
    """Load model, scaler, and anomaly transformer.

    Returns
    -------
    tuple of (model, scaler, anomaly_transformer)
    """
    import tensorflow as tf

    model_path = Path(model_path or config.MODELS_DIR / "gru_final.keras")
    model = tf.keras.models.load_model(model_path)
    logger.info("Loaded model from %s", model_path)

    scaler = FeatureTargetScaler.load(config.PROCESSED_DIR / "scaler.pkl")
    anom_tf = AnomalyTransformer.load(config.PROCESSED_DIR / "target_climatology.pkl")

    return model, scaler, anom_tf


def predict_on_test_set(model_path: str | Path | None = None):
    """Run predictions on the saved test set and print results.

    This is the default mode: loads the pre-split test arrays,
    runs inference, and prints predicted temperatures.
    """
    model, scaler, anom_tf = load_artifacts(model_path)

    X_test = np.load(config.PROCESSED_DIR / "X_test.npy")
    y_test = np.load(config.PROCESSED_DIR / "y_test.npy")
    idx_test = pd.read_csv(
        config.PROCESSED_DIR / "idx_test.csv", index_col=0, parse_dates=True
    ).index

    # Predict (scaled anomaly space)
    pred_scaled = model.predict(X_test, verbose=0).ravel()

    # Inverse scale -> anomaly space
    pred_anom = scaler.inverse_y(pred_scaled)
    true_anom = scaler.inverse_y(y_test)

    # Inverse anomaly -> raw temperature
    pred_raw = anom_tf.inverse_transform(pred_anom, idx_test)
    true_raw = anom_tf.inverse_transform(true_anom, idx_test)

    # Print results
    print(f"\n{'Date':<12} {'Actual (K)':>12} {'Predicted (K)':>14} {'Error (K)':>10}")
    print("-" * 50)
    for dt, tr, pr in zip(idx_test, true_raw, pred_raw):
        err = pr - tr
        marker = " *" if dt.month == config.TARGET_MONTH else ""
        print(f"{dt.strftime('%Y-%m'):<12} {tr:>12.2f} {pr:>14.2f} {err:>+10.2f}{marker}")

    # January summary
    jan_mask = idx_test.month == config.TARGET_MONTH
    if jan_mask.sum() > 0:
        jan_rmse = float(np.sqrt(np.mean((true_anom[jan_mask] - pred_anom[jan_mask]) ** 2)))
        print(f"\nJanuary RMSE (anomaly): {jan_rmse:.4f} K")
        print(f"January count: {jan_mask.sum()}")

    overall_rmse = float(np.sqrt(np.mean((true_anom - pred_anom) ** 2)))
    print(f"Overall RMSE (anomaly): {overall_rmse:.4f} K")


def main():
    parser = argparse.ArgumentParser(description="WCDL Prediction")
    parser.add_argument("--model", type=str, default=None, help="Path to saved .keras model")
    args = parser.parse_args()

    predict_on_test_set(args.model)


if __name__ == "__main__":
    main()
