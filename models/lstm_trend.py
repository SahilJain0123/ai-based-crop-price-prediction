# models/lstm_trend.py
# ─────────────────────────────────────────────────────────────────────────────
# Stacked LSTM — trained on the TREND mode from VMD.
#
# Architecture:
#   Input (seq_len, 1)
#   → LSTM(64, return_sequences=True) → Dropout
#   → LSTM(32) → Dropout
#   → Dense(horizon)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SEQUENCE_LEN, FORECAST_DAYS, MODEL_DIR,
    LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH, LSTM_LR,
    EARLY_STOP_PAT, REDUCE_LR_PAT, REDUCE_LR_FACTOR, RANDOM_SEED
)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_PATH = MODEL_DIR / "lstm_trend.keras"


def build_lstm(seq_len: int = SEQUENCE_LEN,
               horizon: int = FORECAST_DAYS,
               units: list  = LSTM_UNITS,
               dropout: float = LSTM_DROPOUT,
               lr: float      = LSTM_LR) -> "keras.Model":
    """Build and compile the stacked LSTM model."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    tf.random.set_seed(RANDOM_SEED)

    inp = keras.Input(shape=(seq_len, 1), name="trend_input")
    x   = inp

    for i, u in enumerate(units):
        ret_seq = (i < len(units) - 1)   # all but last return sequences
        x = layers.LSTM(u, return_sequences=ret_seq, name=f"lstm_{i}")(x)
        x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    out = layers.Dense(horizon, name="trend_output")(x)
    model = keras.Model(inp, out, name="LSTM_Trend")

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()
    return model


def train_lstm(X_train: np.ndarray,
               y_train: np.ndarray,
               X_val:   np.ndarray,
               y_val:   np.ndarray,
               epochs:  int = LSTM_EPOCHS,
               batch:   int = LSTM_BATCH) -> tuple:
    """
    Train LSTM trend model with early stopping & LR reduction.

    Returns (model, history)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed.")

    model = build_lstm(seq_len=X_train.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PAT,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PAT, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), save_best_only=True,
            monitor="val_loss", verbose=0
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"[LSTM] ✅ Best val_loss: {min(history.history['val_loss']):.6f}")
    return model, history


def load_lstm() -> "keras.Model":
    """Load saved LSTM model from disk."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"[LSTM] Model not found: {MODEL_PATH}. Train first.")
    model = keras.models.load_model(str(MODEL_PATH))
    print(f"[LSTM] Model loaded ← {MODEL_PATH}")
    return model