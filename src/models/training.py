from __future__ import annotations

"""Training loop with proper temporal cross-validation.

Fixes Issue #7 (test set used as validation):
  OLD: model.fit(validation_data=(test_X, test_y)) with EarlyStopping.
       The test set directly influenced model selection.
  NEW: 3-way split.  Validation set (2015-2019) is used for early
       stopping.  Test set (2020-2023) is NEVER seen during training.

Fixes Issue #8 (KFold instead of TimeSeriesSplit):
  OLD: sklearn KFold(shuffle=False).  Future data can still appear
       in training folds.
  NEW: sklearn TimeSeriesSplit guarantees training always precedes
       validation chronologically.
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

from config import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    LOGS_DIR,
    MAX_EPOCHS,
    MODELS_DIR,
    N_CV_SPLITS,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SEED,
)

logger = logging.getLogger(__name__)


# ─── Reproducibility ────────────────────────────────────────────────

def set_seeds(seed: int = SEED) -> None:
    """Set random seeds for numpy, TensorFlow, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seeds set to %d", seed)


# ─── TimeSeriesSplit Cross-Validation ───────────────────────────────

def train_with_timeseries_cv(
    build_model_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_CV_SPLITS,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
) -> dict:
    """Evaluate a model architecture using TimeSeriesSplit CV.

    Unlike KFold, TimeSeriesSplit ensures the training set always
    precedes the validation set in time, preventing look-ahead bias.

    Parameters
    ----------
    build_model_fn : callable
        Zero-argument function that returns a fresh compiled model.
    X : np.ndarray, shape (n_samples, n_timesteps, n_features)
    y : np.ndarray, shape (n_samples,)
    n_splits : int
    batch_size, max_epochs, patience : training params

    Returns
    -------
    dict with keys:
        'mean_rmse', 'std_rmse', 'fold_rmses', 'fold_histories'
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses = []
    fold_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info("  CV Fold %d/%d (train=%d, val=%d)",
                     fold_idx + 1, n_splits, len(train_idx), len(val_idx))

        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        model = build_model_fn()

        es = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        )

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0,
            shuffle=False,
        )

        y_pred = model.predict(X_vl, verbose=0).ravel()
        rmse = float(np.sqrt(np.mean((y_vl - y_pred) ** 2)))
        fold_rmses.append(rmse)
        fold_histories.append(history.history)
        logger.info("    Fold %d RMSE: %.4f", fold_idx + 1, rmse)

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse = float(np.std(fold_rmses))
    logger.info("  CV Result: RMSE = %.4f +/- %.4f", mean_rmse, std_rmse)

    return {
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "fold_rmses": fold_rmses,
        "fold_histories": fold_histories,
    }


# ─── Final Model Training ──────────────────────────────────────────

def train_final_model(
    build_model_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    experiment_name: str = "gru_final",
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
) -> tuple[tf.keras.Model, dict]:
    """Train the final model with a proper validation set.

    Uses validation set for early stopping.  The test set is used
    only for final evaluation and is never passed to this function.

    Parameters
    ----------
    build_model_fn : callable
        Returns a fresh compiled model.
    X_train, y_train : training data.
    X_val, y_val : validation data (NOT test data).
    experiment_name : str
        Used for TensorBoard log directory and saved model filename.

    Returns
    -------
    (model, history_dict)
    """
    log_dir = LOGS_DIR / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{experiment_name}.keras"

    model = build_model_fn()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=1,
        ),
        TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0,
        ),
    ]

    logger.info(
        "Training final model '%s': train=%d, val=%d",
        experiment_name, len(y_train), len(y_val),
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,
    )

    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    return model, history.history
