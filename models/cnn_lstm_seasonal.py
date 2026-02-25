# models/cnn_lstm_seasonal.py
# ─────────────────────────────────────────────────────────────────────────────
# CNN-LSTM — trained on SEASONAL mode from VMD.
#
# Architecture:
#   Input (seq_len, 1)
#   → Conv1D(64, kernel=3, relu)     ← detects local seasonal patterns
#   → MaxPool1D(2)
#   → Conv1D(32, kernel=3, relu)
#   → LSTM(32)                       ← captures temporal order of patterns
#   → Dropout
#   → Dense(horizon)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SEQUENCE_LEN, FORECAST_DAYS, MODEL_DIR,
    CNN_FILTERS, CNN_KERNEL, CNNLSTM_UNITS,
    CNNLSTM_EPOCHS, CNNLSTM_BATCH, CNNLSTM_LR,
    LSTM_DROPOUT, EARLY_STOP_PAT, REDUCE_LR_PAT,
    REDUCE_LR_FACTOR, RANDOM_SEED
)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_PATH = MODEL_DIR / "cnn_lstm_seasonal.keras"


def build_cnn_lstm(seq_len: int   = SEQUENCE_LEN,
                   horizon: int   = FORECAST_DAYS,
                   filters: int   = CNN_FILTERS,
                   kernel: int    = CNN_KERNEL,
                   lstm_u: int    = CNNLSTM_UNITS,
                   dropout: float = LSTM_DROPOUT,
                   lr: float      = CNNLSTM_LR) -> "keras.Model":
    """Build CNN-LSTM for seasonal pattern learning."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed.")

    tf.random.set_seed(RANDOM_SEED)

    inp = keras.Input(shape=(seq_len, 1), name="seasonal_input")

    # CNN block — learns local repeating shapes (weekly/monthly patterns)
    x = layers.Conv1D(filters, kernel, activation="relu",
                      padding="same", name="conv1")(inp)
    x = layers.MaxPooling1D(2, name="pool1")(x)
    x = layers.Conv1D(filters // 2, kernel, activation="relu",
                      padding="same", name="conv2")(x)

    # LSTM block — captures order of patterns
    x   = layers.LSTM(lstm_u, name="lstm_seasonal")(x)
    x   = layers.Dropout(dropout, name="drop_seasonal")(x)
    out = layers.Dense(horizon, name="seasonal_output")(x)

    model = keras.Model(inp, out, name="CNN_LSTM_Seasonal")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()
    return model


def train_cnn_lstm(X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val:   np.ndarray,
                   y_val:   np.ndarray,
                   epochs:  int = CNNLSTM_EPOCHS,
                   batch:   int = CNNLSTM_BATCH) -> tuple:
    """Train CNN-LSTM seasonal model."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed.")

    model = build_cnn_lstm(seq_len=X_train.shape[1])

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

    print(f"[CNN-LSTM] ✅ Best val_loss: {min(history.history['val_loss']):.6f}")
    return model, history


def load_cnn_lstm() -> "keras.Model":
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"[CNN-LSTM] Model not found: {MODEL_PATH}.")
    model = keras.models.load_model(str(MODEL_PATH))
    print(f"[CNN-LSTM] Model loaded ← {MODEL_PATH}")
    return model