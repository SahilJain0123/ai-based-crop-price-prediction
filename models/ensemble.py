# models/ensemble.py
# ─────────────────────────────────────────────────────────────────────────────
# Gating Network — combines 4 model predictions.
#
# Math:
#   P̂ = w_t·P_trend + w_s·P_seasonal + w_c·P_cycle + w_k·P_shock
#   Weights [w_t, w_s, w_c, w_k] learned by neural net conditioned on regime.
#   Softmax output ensures Σ w_i = 1.
#
# Gating input  : [P_t(7) | P_s(7) | P_c(7) | P_k(7) | regime_onehot(3)] = 31
# Gating output : [w_t, w_s, w_c, w_k]  (softmax, sums to 1)
#
# Two modes:
#   1. StaticGating   — config.ENSEMBLE_WEIGHTS (fast fallback)
#   2. LearnedGating  — neural net, minimises final MSE (trained by trainer.py)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ENSEMBLE_WEIGHTS, MODEL_DIR, RANDOM_SEED, FORECAST_DAYS

GATE_PATH = MODEL_DIR / "gating_weights.pkl"

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Static Gating (no training needed, uses config weights)
# ─────────────────────────────────────────────────────────────────────────────
class StaticGating:
    """
    Regime-conditional fixed weights from config.ENSEMBLE_WEIGHTS.

    Usage
    -----
    sg  = StaticGating()
    out = sg.combine(p_trend, p_seasonal, p_cycle, p_shock, regime="Shock")
    """

    def combine(self,
                pred_trend:    np.ndarray,
                pred_seasonal: np.ndarray,
                pred_cycle:    np.ndarray,
                pred_shock:    np.ndarray,
                regime:        str = "Normal") -> np.ndarray:
        """
        Weighted combination of 4 model predictions.

        All inputs: shape (horizon,) or (N, horizon)
        Returns   : same shape as inputs
        """
        w  = ENSEMBLE_WEIGHTS.get(regime, ENSEMBLE_WEIGHTS["Normal"])
        wt = w["trend"]
        ws = w["seasonal"]
        wc = w["cycle"]
        wk = w["shock"]

        final = wt * pred_trend + ws * pred_seasonal + wc * pred_cycle + wk * pred_shock

        print(f"[Ensemble/Static] Regime={regime} → "
              f"trend={wt:.2f}, seasonal={ws:.2f}, "
              f"cycle={wc:.2f}, shock={wk:.2f}")
        return final

    def get_weights(self, regime: str = "Normal") -> dict:
        return ENSEMBLE_WEIGHTS.get(regime, ENSEMBLE_WEIGHTS["Normal"])


# ─────────────────────────────────────────────────────────────────────────────
#  Learned Gating (neural net, trained end-to-end)
# ─────────────────────────────────────────────────────────────────────────────
class LearnedGating:
    """
    Neural gating network: learns optimal regime-conditional weights.

    Architecture
    ────────────
    Input  (31,) = [P_t | P_s | P_c | P_k | regime_onehot]
                    7  +  7  +  7  +  7  +      3
    → Dense(64, relu)
    → Dense(32, relu)
    → Dense(4, softmax)  ← weights summing to 1
    Final: P̂ = w[0]·P_t + w[1]·P_s + w[2]·P_c + w[3]·P_k

    Trained with custom GradientTape loop minimising final MSE.
    """

    def __init__(self):
        self.gate_net = None
        self._fitted  = False
        self._horizon = FORECAST_DAYS

    def _regime_onehot(self, regime: str) -> np.ndarray:
        mapping = {"Calm": [1, 0, 0], "Normal": [0, 1, 0], "Shock": [0, 0, 1]}
        return np.array(mapping.get(regime, [0, 1, 0]), dtype=np.float32)

    def _build(self, horizon: int = FORECAST_DAYS) -> "keras.Model":
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed.")

        tf.random.set_seed(RANDOM_SEED)

        inp_size = 4 * horizon + 3      # 4 model predictions + regime one-hot
        inp = keras.Input(shape=(inp_size,), name="gate_input")
        x   = layers.Dense(64, activation="relu")(inp)
        x   = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(4, activation="softmax", name="gate_weights")(x)

        net = keras.Model(inp, out, name="GatingNetwork_4model")
        net.compile(optimizer="adam", loss="mse")
        self.gate_net = net
        self._horizon = horizon
        return net

    def fit(self,
            P_trend:    np.ndarray,
            P_seasonal: np.ndarray,
            P_cycle:    np.ndarray,
            P_shock:    np.ndarray,
            y_true:     np.ndarray,
            regimes:    list,
            epochs:     int = 50,
            batch:      int = 32):
        """
        Train the gating network.

        Parameters
        ----------
        P_trend, P_seasonal, P_cycle, P_shock : (N, horizon) — model outputs
        y_true    : (N, horizon) — ground truth combined prices
        regimes   : list of N regime strings ("Calm"|"Normal"|"Shock")
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed.")

        N, horizon = P_trend.shape
        self._build(horizon)

        # Build gating input: [P_t | P_s | P_c | P_k | regime_onehot]
        regime_arr = np.array(
            [self._regime_onehot(r) for r in regimes], dtype=np.float32
        )                                               # (N, 3)
        X_gate = np.concatenate(
            [P_trend.astype(np.float32),
             P_seasonal.astype(np.float32),
             P_cycle.astype(np.float32),
             P_shock.astype(np.float32),
             regime_arr], axis=1
        )                                               # (N, 31)

        optimizer  = tf.keras.optimizers.Adam(0.001)
        best_loss  = np.inf
        best_w     = None

        for ep in range(epochs):
            # Mini-batch loop
            idx      = np.random.permutation(N)
            ep_losses = []

            for start in range(0, N, batch):
                b_idx = idx[start: start + batch]
                Xb    = X_gate[b_idx]
                Ptb   = P_trend[b_idx].astype(np.float32)
                Psb   = P_seasonal[b_idx].astype(np.float32)
                Pcb   = P_cycle[b_idx].astype(np.float32)
                Pkb   = P_shock[b_idx].astype(np.float32)
                yb    = y_true[b_idx].astype(np.float32)

                with tf.GradientTape() as tape:
                    w    = self.gate_net(Xb, training=True)   # (B, 4)
                    pred = (w[:, 0:1] * Ptb + w[:, 1:2] * Psb
                            + w[:, 2:3] * Pcb + w[:, 3:4] * Pkb)
                    loss = tf.reduce_mean((pred - yb) ** 2)

                grads = tape.gradient(loss, self.gate_net.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, self.gate_net.trainable_variables)
                )
                ep_losses.append(float(loss))

            ep_loss = np.mean(ep_losses)
            if ep_loss < best_loss:
                best_loss = ep_loss
                best_w    = self.gate_net.get_weights()

            if ep % 10 == 0:
                print(f"  [Gating] Epoch {ep:3d}/{epochs}  loss={ep_loss:.6f}")

        self.gate_net.set_weights(best_w)
        self._fitted = True
        print(f"[Gating] ✅ Best loss: {best_loss:.6f}")

    def combine(self,
                pred_trend:    np.ndarray,
                pred_seasonal: np.ndarray,
                pred_cycle:    np.ndarray,
                pred_shock:    np.ndarray,
                regime:        str = "Normal") -> np.ndarray:
        """Combine 4 predictions using learned weights."""
        if not self._fitted:
            print("[Gating] Not fitted — falling back to StaticGating")
            return StaticGating().combine(
                pred_trend, pred_seasonal, pred_cycle, pred_shock, regime)

        pt = np.atleast_2d(pred_trend).astype(np.float32)
        ps = np.atleast_2d(pred_seasonal).astype(np.float32)
        pc = np.atleast_2d(pred_cycle).astype(np.float32)
        pk = np.atleast_2d(pred_shock).astype(np.float32)
        ro = np.tile(
            self._regime_onehot(regime), (len(pt), 1)
        ).astype(np.float32)

        X = np.concatenate([pt, ps, pc, pk, ro], axis=1)  # (N, 31)

        # Validate shape
        expected = 4 * self._horizon + 3
        if X.shape[1] != expected:
            print(f"[Gating] Shape mismatch ({X.shape[1]}≠{expected}) — "
                  "using StaticGating")
            return StaticGating().combine(
                pred_trend, pred_seasonal, pred_cycle, pred_shock, regime)

        w    = self.gate_net.predict(X, verbose=0)          # (N, 4)
        pred = (w[:, 0:1] * pt + w[:, 1:2] * ps
                + w[:, 2:3] * pc + w[:, 3:4] * pk)

        wm = w.mean(axis=0)
        print(f"[Gating/Learned] trend={wm[0]:.3f}  seasonal={wm[1]:.3f}  "
              f"cycle={wm[2]:.3f}  shock={wm[3]:.3f}")
        return pred.squeeze()

    def save(self, path: Path = GATE_PATH):
        if not self._fitted:
            return
        keras_path = str(path).replace(".pkl", ".keras")
        self.gate_net.save(keras_path)
        print(f"[Gating] Saved → {keras_path}")

    def load(self, path: Path = GATE_PATH):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed.")
        keras_path = str(path).replace(".pkl", ".keras")
        if not Path(keras_path).exists():
            raise FileNotFoundError(f"[Gating] Not found: {keras_path}")
        self.gate_net = keras.models.load_model(keras_path)
        # Infer horizon from model input shape: 4*h+3 → h = (inp-3)//4
        inp_size      = self.gate_net.input_shape[1]
        self._horizon = (inp_size - 3) // 4
        self._fitted  = True
        print(f"[Gating] Loaded ← {keras_path}  (horizon={self._horizon})")
        return self


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    N, H = 150, 7
    pt = np.random.rand(N, H).astype(np.float32) * 100 + 800
    ps = np.random.rand(N, H).astype(np.float32) * 30
    pc = np.random.rand(N, H).astype(np.float32) * 20 - 10
    pk = np.random.rand(N, H).astype(np.float32) * 15 - 7
    y  = pt + ps + pc + pk + np.random.randn(N, H).astype(np.float32) * 5

    # Static test
    sg  = StaticGating()
    out = sg.combine(pt[0], ps[0], pc[0], pk[0], regime="Shock")
    print(f"Static output shape: {out.shape}, sample: {out[:3]}")

    # Learned test (requires TF)
    if TF_AVAILABLE:
        regs = np.random.choice(["Calm", "Normal", "Shock"], N)
        lg   = LearnedGating()
        lg.fit(pt, ps, pc, pk, y, list(regs), epochs=20)
        out2 = lg.combine(pt[0], ps[0], pc[0], pk[0], "Normal")
        print(f"Learned output: {out2[:3]}")