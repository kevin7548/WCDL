from __future__ import annotations

"""GRU model architecture builder.

Fixes Issue #6 (overparameterization):
  OLD: 6 GRU layers x 256 units = 1,778,945 parameters.
       Training set: ~420 samples.  Ratio = 4,235:1 (extreme overfitting).
  NEW: 2 GRU layers (64 + 32) ~ 25,000 parameters.
       Ratio ~ 60:1 which is aggressive but within bounds for
       a regularised model with early stopping and dropout.
"""

import logging

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input

from config import GRU_UNITS, DROPOUT_RATE, LEARNING_RATE

logger = logging.getLogger(__name__)


def build_gru_model(
    n_timesteps: int,
    n_features: int,
    gru_units: list[int] | None = None,
    dropout: float | None = None,
    learning_rate: float | None = None,
) -> tf.keras.Model:
    """Build a compact GRU model for anomaly prediction.

    Architecture (default)::

        Input(n_timesteps, n_features)
         -> GRU(64, return_sequences=True)
         -> Dropout(0.2)
         -> GRU(32, return_sequences=False)
         -> Dropout(0.2)
         -> Dense(1, linear)

    Parameters
    ----------
    n_timesteps : int
        Sequence length (number of past months).
    n_features : int
        Number of input features per timestep.
    gru_units : list of int, optional
        Number of units per GRU layer.  Defaults to ``config.GRU_UNITS``.
    dropout : float, optional
        Dropout rate after each GRU layer.
    learning_rate : float, optional
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled model ready for training.
    """
    gru_units = gru_units or GRU_UNITS
    dropout = dropout if dropout is not None else DROPOUT_RATE
    learning_rate = learning_rate or LEARNING_RATE

    model = Sequential(name="gru_anomaly")
    model.add(Input(shape=(n_timesteps, n_features)))

    for i, units in enumerate(gru_units):
        return_seq = i < len(gru_units) - 1
        model.add(
            GRU(
                units=units,
                return_sequences=return_seq,
                name=f"gru_{i+1}",
            )
        )
        model.add(Dropout(dropout, name=f"dropout_{i+1}"))

    model.add(Dense(1, activation="linear", name="output"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    n_params = model.count_params()
    logger.info(
        "GRU model built: %d layers, units=%s, %d params",
        len(gru_units), gru_units, n_params,
    )
    return model
