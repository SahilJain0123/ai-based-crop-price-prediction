# models/transformer_shock.py
# ─────────────────────────────────────────────────────────────────────────────
# Transformer — trained on SHOCK mode from VMD.
#
# Why Transformer for shocks?
#   Multi-head attention looks at ALL past time steps simultaneously and
#   learns WHICH past shock events are most relevant right now.
#   Attention(Q,K,V) = softmax(QKᵀ / √d) · V
#
# Architecture:
#   Input (seq_len, 1)
#   → Linear projection to d_model
#   → N × TransformerBlock [MultiHeadAttention + FFN + LayerNorm]
#   → GlobalAvgPool
#   → Dense(horizon)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SEQUENCE_LEN, FORECAST_DAYS, MODEL_DIR,
    TRANS_D_MODEL, TRANS_N_HEADS, TRANS_FF_DIM,
    TRANS_N_LAYERS, TRANS_DROPOUT, TRANS_EPOCHS,
    TRANS_BATCH, TRANS_LR,
    EARLY_STOP_PAT, REDUCE_LR_PAT, REDUCE_LR_FACTOR, RANDOM_SEED
)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_PATH = MODEL_DIR / "transformer_shock.keras"


# ─────────────────────────────────────────────────────────────────────────────
#  Custom Transformer Block
# ─────────────────────────────────────────────────────────────────────────────
def transformer_block(x, d_model: int, n_heads: int,
                      ff_dim: int, dropout: float, name: str = ""):
    """
    One Transformer encoder block:
        x → MultiHeadAttention → Add & Norm
          → FFN(Dense-ReLU-Dense) → Add & Norm
    """
    # Multi-head self-attention
    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads,
        dropout=dropout, name=f"mha_{name}"
    )(x, x)
    x = layers.LayerNormalization(name=f"ln1_{name}")(x + attn_out)

    # Feed-forward network
    ffn = layers.Dense(ff_dim, activation="relu", name=f"ffn1_{name}")(x)
    ffn = layers.Dense(d_model, name=f"ffn2_{name}")(ffn)
    ffn = layers.Dropout(dropout, name=f"drop_{name}")(ffn)
    x   = layers.LayerNormalization(name=f"ln2_{name}")(x + ffn)

    return x


def build_transformer(seq_len: int   = SEQUENCE_LEN,
                      horizon: int   = FORECAST_DAYS,
                      d_model: int   = TRANS_D_MODEL,
                      n_heads: int   = TRANS_N_HEADS,
                      ff_dim: int    = TRANS_FF_DIM,
                      n_layers: int  = TRANS_N_LAYERS,
                      dropout: float = TRANS_DROPOUT,
                      lr: float      = TRANS_LR) -> "keras.Model":
    """Build Transformer model for shock component."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed.")

    tf.random.set_seed(RANDOM_SEED)

    inp = keras.Input(shape=(seq_len, 1), name="shock_input")

    # Project input to d_model dimensions
    x = layers.Dense(d_model, name="input_proj")(inp)

    # Stack Transformer blocks
    for i in range(n_layers):
        x = transformer_block(x, d_model, n_heads, ff_dim, dropout, name=str(i))

    # Aggregate across time dimension
    x   = layers.GlobalAveragePooling1D(name="gap")(x)
    x   = layers.Dropout(dropout, name="final_drop")(x)
    out = layers.Dense(horizon, name="shock_output")(x)

    model = keras.Model(inp, out, name="Transformer_Shock")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()
    return model


def train_transformer(X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val:   np.ndarray,
                      y_val:   np.ndarray,
                      epochs:  int = TRANS_EPOCHS,
                      batch:   int = TRANS_BATCH) -> tuple:
    """Train Transformer shock model."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed.")

    model = build_transformer(seq_len=X_train.shape[1])

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

    print(f"[Transformer] ✅ Best val_loss: {min(history.history['val_loss']):.6f}")
    return model, history


def load_transformer() -> "keras.Model":
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"[Transformer] Model not found: {MODEL_PATH}.")
    model = keras.models.load_model(str(MODEL_PATH))
    print(f"[Transformer] Model loaded ← {MODEL_PATH}")
    return model