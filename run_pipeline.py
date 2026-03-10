#!/usr/bin/env python3
"""WCDL main pipeline orchestrator.

Usage
-----
    python run_pipeline.py                     # full pipeline (skip download)
    python run_pipeline.py --step download     # only data download
    python run_pipeline.py --step preprocess   # only preprocessing
    python run_pipeline.py --step select       # only feature selection
    python run_pipeline.py --step engineer     # only feature engineering
    python run_pipeline.py --step train        # only training
    python run_pipeline.py --step evaluate     # only evaluation
    python run_pipeline.py --download          # include download step

Each step saves intermediate outputs to data/processed/ so subsequent
steps can be run independently.
"""

import argparse
import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils.paths import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ════════════════════════════════════════════════════════════════════
# Step 1: Download
# ════════════════════════════════════════════════════════════════════

def step_download():
    """Download raw NOAA indices and ERA5 data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Download")
    logger.info("=" * 60)

    from src.data.download_noaa import download_noaa_indices
    from src.data.download_era5 import download_era5_single_levels, download_era5_land

    download_noaa_indices()
    download_era5_single_levels()
    download_era5_land()

    logger.info("Download complete.")


# ════════════════════════════════════════════════════════════════════
# Step 2: Preprocess
# ════════════════════════════════════════════════════════════════════

def step_preprocess():
    """Load raw data, compute region averages, compute anomalies."""
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 60)

    from src.data.load import load_era5_merged, load_noaa_indices, compute_korea_mean_temp
    from src.preprocessing.era5_preprocess import regrid_era5, compute_all_region_averages
    from src.preprocessing.indices_preprocess import get_clean_indices
    from src.preprocessing.anomaly import compute_anomalies_dataframe

    # 2a. Load ERA5 and compute region-averaged features
    logger.info("Loading ERA5 data...")
    era5_ds = load_era5_merged()
    era5_ds = regrid_era5(era5_ds)
    era5_features = compute_all_region_averages(era5_ds)

    # 2b. Load NOAA indices
    logger.info("Loading NOAA indices...")
    noaa_features = get_clean_indices()

    # 2c. Compute Korea mean temperature (target)
    logger.info("Computing Korea mean temperature...")
    korea_temp = compute_korea_mean_temp(era5_ds)

    # 2d. Merge all features
    logger.info("Merging features...")
    all_features = pd.concat([noaa_features, era5_features], axis=1)
    # Align to common time range
    common_idx = all_features.index.intersection(korea_temp.index).sort_values()
    all_features = all_features.loc[common_idx]
    korea_temp = korea_temp.loc[common_idx]

    # Drop features with any remaining NaN
    all_features = all_features.dropna(axis=1, how="any")
    logger.info("Combined features: %d variables, %d months", len(all_features.columns), len(all_features))

    # 2e. Compute anomalies (fit climatology on TRAINING period only)
    train_mask = all_features.index <= pd.Timestamp(config.TRAIN_END)
    logger.info("Computing anomalies (climatology from training period: %s to %s)...",
                config.TRAIN_START, config.TRAIN_END)

    features_anom, feat_transformers = compute_anomalies_dataframe(all_features, train_mask)

    from src.preprocessing.anomaly import AnomalyTransformer
    target_anom_tf = AnomalyTransformer()
    target_anom_tf.fit(korea_temp[train_mask].values, korea_temp.index[train_mask])
    target_anom = pd.Series(
        target_anom_tf.transform(korea_temp.values, korea_temp.index),
        index=korea_temp.index,
        name="kmt_anom",
    )

    # 2f. Save intermediate outputs
    features_anom.to_csv(config.PROCESSED_DIR / "features_anomaly.csv")
    target_anom.to_csv(config.PROCESSED_DIR / "target_anomaly.csv")
    target_anom_tf.save(config.PROCESSED_DIR / "target_climatology.pkl")
    joblib.dump(feat_transformers, config.PROCESSED_DIR / "feature_climatologies.pkl")
    # Save raw target for later inverse transform
    korea_temp.to_csv(config.PROCESSED_DIR / "korea_mean_temp_raw.csv")

    logger.info("Preprocessing complete. Outputs saved to %s", config.PROCESSED_DIR)


# ════════════════════════════════════════════════════════════════════
# Step 3: Feature Selection
# ════════════════════════════════════════════════════════════════════

def step_select():
    """Select features via lag-correlation, stepwise regression, VIF."""
    logger.info("=" * 60)
    logger.info("STEP 3: Feature Selection")
    logger.info("=" * 60)

    from src.features.feature_selection import (
        compute_lag_correlations,
        select_optimal_lag_per_feature,
        stepwise_selection,
        remove_high_vif,
    )

    features_anom = pd.read_csv(config.PROCESSED_DIR / "features_anomaly.csv", index_col=0, parse_dates=True)
    target_anom = pd.read_csv(config.PROCESSED_DIR / "target_anomaly.csv", index_col=0, parse_dates=True).squeeze()

    # Use only training data for feature selection
    train_mask = features_anom.index <= pd.Timestamp(config.TRAIN_END)
    feat_train = features_anom[train_mask]
    tgt_train = target_anom[train_mask]

    # 3a. Lag correlations on anomaly data (all months — Fix #2)
    logger.info("Computing lag correlations on anomalies...")
    corr_df = compute_lag_correlations(feat_train, tgt_train, max_lag=config.MAX_LAG_MONTHS)
    logger.info("Top correlations:\n%s", corr_df.abs().max(axis=1).sort_values(ascending=False).head(10))

    # 3b. Optimal lag per feature
    optimal_lags = select_optimal_lag_per_feature(corr_df)

    # 3c. Build lagged feature matrix at optimal lags (for stepwise)
    lagged_cols = {}
    for feat, lag in optimal_lags.items():
        lagged_cols[feat] = feat_train[feat].shift(lag)
    lagged_df = pd.DataFrame(lagged_cols).dropna()
    tgt_aligned = tgt_train.loc[lagged_df.index]

    # 3d. Stepwise regression (p < 0.05, tighter than old 0.1)
    logger.info("Running stepwise selection...")
    selected = stepwise_selection(lagged_df, tgt_aligned, sl_enter=0.05, sl_remove=0.05)

    if len(selected) < 2:
        logger.warning("Stepwise selected < 2 features. Falling back to top-5 by correlation.")
        top_corr = corr_df.abs().max(axis=1).sort_values(ascending=False)
        selected = top_corr.head(5).index.tolist()

    # 3e. VIF check
    logger.info("Checking VIF...")
    selected = remove_high_vif(lagged_df[selected], threshold=10.0)

    # 3f. Save selection results
    selection_result = {
        "selected_features": selected,
        "optimal_lags": {f: int(optimal_lags[f]) for f in selected},
    }
    output_path = config.PROCESSED_DIR / "selected_features.json"
    with open(output_path, "w") as f:
        json.dump(selection_result, f, indent=2)

    logger.info("Selected %d features: %s", len(selected), selected)
    logger.info("Saved to %s", output_path)


# ════════════════════════════════════════════════════════════════════
# Step 4: Feature Engineering
# ════════════════════════════════════════════════════════════════════

def step_engineer():
    """Create GRU-ready sequences and split into train/val/test."""
    logger.info("=" * 60)
    logger.info("STEP 4: Feature Engineering")
    logger.info("=" * 60)

    from src.features.feature_engineering import create_gru_sequences, split_by_date
    from src.utils.scaling import FeatureTargetScaler

    features_anom = pd.read_csv(config.PROCESSED_DIR / "features_anomaly.csv", index_col=0, parse_dates=True)
    target_anom = pd.read_csv(config.PROCESSED_DIR / "target_anomaly.csv", index_col=0, parse_dates=True).squeeze()

    with open(config.PROCESSED_DIR / "selected_features.json") as f:
        sel = json.load(f)
    selected_features = sel["selected_features"]

    # Use only selected features
    features_sel = features_anom[selected_features]

    # 4a. Create GRU sequences — lags applied ONCE here (Fix #3)
    # Shape: (n_samples, N_TIMESTEPS, n_features) (Fix #5)
    logger.info("Creating GRU sequences (timesteps=%d)...", config.N_TIMESTEPS)
    X, y, time_idx = create_gru_sequences(features_sel, target_anom, n_timesteps=config.N_TIMESTEPS)

    logger.info("X shape: %s  (should be (n, %d, %d))", X.shape, config.N_TIMESTEPS, len(selected_features))

    # 4b. Split by date
    splits = split_by_date(X, y, time_idx, config.TRAIN_END, config.VAL_END)

    # 4c. Fit scaler on TRAINING data only (Fix #4)
    scaler = FeatureTargetScaler()
    scaler.fit(splits["X_train"], splits["y_train"])

    # 4d. Scale all splits
    for name in ["train", "val", "test"]:
        splits[f"X_{name}"] = scaler.transform_X(splits[f"X_{name}"])
        splits[f"y_{name}"] = scaler.transform_y(splits[f"y_{name}"])

    # 4e. Save
    for key, arr in splits.items():
        if isinstance(arr, np.ndarray):
            np.save(config.PROCESSED_DIR / f"{key}.npy", arr)
        else:
            arr.to_frame().to_csv(config.PROCESSED_DIR / f"{key}.csv")

    scaler.save(config.PROCESSED_DIR / "scaler.pkl")

    logger.info("Feature engineering complete.")
    logger.info("  X_train: %s, X_val: %s, X_test: %s",
                splits["X_train"].shape, splits["X_val"].shape, splits["X_test"].shape)


# ════════════════════════════════════════════════════════════════════
# Step 5: Train
# ════════════════════════════════════════════════════════════════════

def step_train():
    """Train GRU model + baselines."""
    logger.info("=" * 60)
    logger.info("STEP 5: Training")
    logger.info("=" * 60)

    from src.models.gru_model import build_gru_model
    from src.models.baseline import ClimatologyBaseline, PersistenceBaseline, LinearRegressionBaseline
    from src.models.training import set_seeds, train_with_timeseries_cv, train_final_model

    # Load data
    X_train = np.load(config.PROCESSED_DIR / "X_train.npy")
    y_train = np.load(config.PROCESSED_DIR / "y_train.npy")
    X_val = np.load(config.PROCESSED_DIR / "X_val.npy")
    y_val = np.load(config.PROCESSED_DIR / "y_val.npy")

    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    set_seeds()

    # 5a. TimeSeriesSplit CV on training data
    logger.info("Running TimeSeriesSplit CV...")
    build_fn = lambda: build_gru_model(n_timesteps, n_features)
    cv_results = train_with_timeseries_cv(build_fn, X_train, y_train)

    # 5b. Train final GRU model (validation = val set, NOT test)
    logger.info("Training final GRU model...")
    set_seeds()
    gru_model, gru_history = train_final_model(build_fn, X_train, y_train, X_val, y_val)

    # 5c. Train baselines
    logger.info("Training baselines...")
    clim = ClimatologyBaseline().fit(X_train, y_train)
    pers = PersistenceBaseline().fit(X_train, y_train)
    ridge = LinearRegressionBaseline().fit(X_train, y_train)

    joblib.dump(clim, config.MODELS_DIR / "baseline_climatology.pkl")
    joblib.dump(pers, config.MODELS_DIR / "baseline_persistence.pkl")
    joblib.dump(ridge, config.MODELS_DIR / "baseline_ridge.pkl")
    joblib.dump(cv_results, config.MODELS_DIR / "cv_results.pkl")
    joblib.dump(gru_history, config.MODELS_DIR / "gru_history.pkl")

    logger.info("Training complete. CV RMSE: %.4f +/- %.4f",
                cv_results["mean_rmse"], cv_results["std_rmse"])


# ════════════════════════════════════════════════════════════════════
# Step 6: Evaluate
# ════════════════════════════════════════════════════════════════════

def step_evaluate():
    """Evaluate all models on test data, produce plots and metrics."""
    logger.info("=" * 60)
    logger.info("STEP 6: Evaluation")
    logger.info("=" * 60)

    import tensorflow as tf
    from src.evaluation.metrics import (
        compute_metrics,
        print_summary_table,
        plot_prediction_timeseries,
        plot_scatter,
        plot_january_predictions,
        plot_baseline_comparison,
    )
    from src.utils.scaling import FeatureTargetScaler
    from src.preprocessing.anomaly import AnomalyTransformer

    # Load test data + models
    X_test = np.load(config.PROCESSED_DIR / "X_test.npy")
    y_test = np.load(config.PROCESSED_DIR / "y_test.npy")
    idx_test = pd.read_csv(config.PROCESSED_DIR / "idx_test.csv", index_col=0, parse_dates=True).index

    scaler = FeatureTargetScaler.load(config.PROCESSED_DIR / "scaler.pkl")
    target_anom_tf = AnomalyTransformer.load(config.PROCESSED_DIR / "target_climatology.pkl")

    gru_model = tf.keras.models.load_model(config.MODELS_DIR / "gru_final.keras")
    clim = joblib.load(config.MODELS_DIR / "baseline_climatology.pkl")
    pers = joblib.load(config.MODELS_DIR / "baseline_persistence.pkl")
    ridge = joblib.load(config.MODELS_DIR / "baseline_ridge.pkl")

    # Predict (in scaled anomaly space)
    models = {
        "Climatology": clim,
        "Persistence": pers,
        "Ridge Regression": ridge,
        "GRU": gru_model,
    }

    results_all = {}
    results_jan = {}
    predictions = {}

    for name, model in models.items():
        if name == "GRU":
            pred_scaled = model.predict(X_test, verbose=0).ravel()
        else:
            pred_scaled = model.predict(X_test)

        # Inverse scale back to anomaly space
        pred_anom = scaler.inverse_y(pred_scaled)
        true_anom = scaler.inverse_y(y_test)

        predictions[name] = pred_anom

        # All-month metrics
        results_all[name] = compute_metrics(true_anom, pred_anom)

        # January-only metrics
        jan_mask = idx_test.month == config.TARGET_MONTH
        if jan_mask.sum() > 0:
            results_jan[name] = compute_metrics(true_anom[jan_mask], pred_anom[jan_mask])

    # Print tables
    print("\n>>> ALL MONTHS (Test Set)")
    print_summary_table(results_all)

    print(">>> JANUARY ONLY (Test Set)")
    print_summary_table(results_jan)

    # Plots
    true_anom = scaler.inverse_y(y_test)
    gru_pred_anom = predictions["GRU"]

    plot_prediction_timeseries(
        true_anom, gru_pred_anom, idx_test,
        title="GRU: All Months (Test Set Anomalies)",
        save_path=config.OUTPUTS_DIR / "prediction_all_months_anomaly.png",
    )

    plot_scatter(
        true_anom, gru_pred_anom,
        title="GRU: Predicted vs Actual Anomaly (Test Set)",
        save_path=config.OUTPUTS_DIR / "scatter_all_months.png",
    )

    # January plots
    jan_mask = idx_test.month == config.TARGET_MONTH
    if jan_mask.sum() > 0:
        jan_years = idx_test[jan_mask].year.values

        plot_january_predictions(
            true_anom[jan_mask], gru_pred_anom[jan_mask], jan_years,
            title="GRU: January Predictions (Test Set)",
            save_path=config.OUTPUTS_DIR / "prediction_january_anomaly.png",
        )

        plot_scatter(
            true_anom[jan_mask], gru_pred_anom[jan_mask],
            title="GRU: January Predicted vs Actual",
            save_path=config.OUTPUTS_DIR / "scatter_january.png",
        )

    # Baseline comparison
    plot_baseline_comparison(
        results_all, metric="rmse",
        title="Model Comparison: RMSE (All Months)",
        save_path=config.OUTPUTS_DIR / "comparison_rmse_all.png",
    )
    if results_jan:
        plot_baseline_comparison(
            results_jan, metric="rmse",
            title="Model Comparison: RMSE (January Only)",
            save_path=config.OUTPUTS_DIR / "comparison_rmse_january.png",
        )

    logger.info("Evaluation complete. Plots saved to %s", config.OUTPUTS_DIR)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

STEPS = {
    "download": step_download,
    "preprocess": step_preprocess,
    "select": step_select,
    "engineer": step_engineer,
    "train": step_train,
    "evaluate": step_evaluate,
}


def main():
    parser = argparse.ArgumentParser(description="WCDL Pipeline")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        default=None,
        help="Run a single pipeline step. Omit to run all steps.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Include the download step in the full pipeline run.",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.step:
        STEPS[args.step]()
    else:
        # Full pipeline
        if args.download:
            step_download()
        else:
            logger.info("Skipping download step (use --download to include).")
        step_preprocess()
        step_select()
        step_engineer()
        step_train()
        step_evaluate()

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
