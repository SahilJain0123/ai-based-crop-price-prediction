# predictor.py
# ─────────────────────────────────────────────────────────────────────────────
# Inference pipeline — 4 models (LSTM, CNN-LSTM, Bi-LSTM, Transformer).
#
# Implements full methodology:
#   ✅ decomp dict always has 'model', 'label', 'value', 'contribution_pct'
#   ✅ Mode scalers loaded from trainer (SAME scalers, no scale mismatch)
#   ✅ K=4 VMD: Trend → LSTM, Seasonal → CNN-LSTM, Cycle → Bi-LSTM, Shock → Transformer
#   ✅ MC Dropout uncertainty → P̂ ± σ  (methodology §5)
#   ✅ Decision output: Hold / Sell / Wait  (methodology §7.4)
#   ✅ Gradient-based day importance (methodology §explainability)
#   ✅ Sanity-clipped predictions — no wild outliers
#   ✅ Graceful fallback if any model file missing
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from datetime import timedelta

from config import (
    PRICE_COL, DATE_COL, SEQUENCE_LEN, FORECAST_DAYS,
    MODEL_DIR, RESULT_DIR, ENSEMBLE_WEIGHTS
)
from preprocessor import Preprocessor
from vmd_decomposer import VMDDecomposer
from shock_detector import ShockDetector
from models.ensemble import StaticGating

try:
    from models.lstm_trend        import load_lstm
    from models.cnn_lstm_seasonal import load_cnn_lstm
    from models.bilstm_cycle      import load_bilstm
    from models.transformer_shock import load_transformer
    from models.ensemble          import LearnedGating
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

MODE_SCALERS_PATH = MODEL_DIR / "mode_scalers.pkl"
LABELS            = ["Trend", "Seasonal", "Cycle", "Shock"]

# Fallback display metadata (used when old predictor.py decomp lacks these keys)
_MODEL_NAMES  = {"trend": "LSTM", "seasonal": "CNN-LSTM",
                 "cycle": "Bi-LSTM", "shock": "Transformer"}
_MODE_LABELS  = {
    "trend":    "Long-term supply/demand direction",
    "seasonal": "Harvest cycles & weekly market patterns",
    "cycle":    "Medium-term economic cycle (forward+backward context)",
    "shock":    "Supply disruption / sudden price event",
}


class PricePredictor:

    def __init__(self):
        self.lstm_model   = None   # Trend
        self.cnn_model    = None   # Seasonal
        self.bili_model   = None   # Cycle (Bi-LSTM)
        self.trans_model  = None   # Shock
        self.gate         = None
        self.pp           = None
        self.mode_scalers = None   # dict: label → fitted MinMaxScaler
        self.sd           = ShockDetector()
        self._loaded      = False

    # ── Load all models ───────────────────────────────────────────────────
    def load(self):
        """Load all trained models and scalers from disk."""
        if not TF_AVAILABLE:
            print("[Predictor] TensorFlow unavailable — using statistical fallback")
            return self

        try:
            self.lstm_model  = load_lstm()
            self.cnn_model   = load_cnn_lstm()
            self.trans_model = load_transformer()
        except FileNotFoundError as e:
            print(f"[Predictor] ❌ {e}")
            print("  → Run: python trainer.py  to train all models.")
            return self

        # Bi-LSTM: new model, may not exist if user hasn't retrained yet
        try:
            self.bili_model = load_bilstm()
        except FileNotFoundError:
            print("[Predictor] ⚠  Bi-LSTM not found — Cycle mode uses linear fallback.")
            print("  → Retrain (python trainer.py) to enable full 4-model pipeline.")
            self.bili_model = None

        # Global price scaler
        self.pp = Preprocessor().load()

        # Per-mode scalers — MUST be the same scalers used during training
        if MODE_SCALERS_PATH.exists():
            with open(MODE_SCALERS_PATH, "rb") as f:
                self.mode_scalers = pickle.load(f)
            print(f"[Predictor] Mode scalers loaded ← {MODE_SCALERS_PATH}")
        else:
            print("[Predictor] ⚠  mode_scalers.pkl missing. Retrain to fix.")
            self.mode_scalers = None

        # Gating network
        gate_keras = MODEL_DIR / "gating_weights.keras"
        if gate_keras.exists():
            try:
                g = LearnedGating().load()
                expected = 4 * FORECAST_DAYS + 3   # 31 for 4-model gating
                if g.gate_net.input_shape[1] == expected:
                    self.gate = g
                    print("[Predictor] Learned gating loaded")
                else:
                    print(f"[Predictor] Gating shape mismatch "
                          f"({g.gate_net.input_shape[1]}≠{expected}) "
                          "— using StaticGating. Retrain to fix.")
                    self.gate = StaticGating()
            except Exception as exc:
                print(f"[Predictor] Gating load error ({exc}) — StaticGating")
                self.gate = StaticGating()
        else:
            self.gate = StaticGating()

        self._loaded = True
        print("[Predictor] ✅ Ready")
        return self

    # ── Main predict ───────────────────────────────────────────────────────
    def predict(self,
                df: pd.DataFrame,
                target_date=None,
                horizon: int = FORECAST_DAYS) -> dict:
        """
        Full 4-model inference pipeline.

        Parameters
        ----------
        df          : full historical price DataFrame
        target_date : display date (optional)
        horizon     : always forced to FORECAST_DAYS (7) to match gating

        Returns
        -------
        dict containing:
            final_price, forecast_series, forecast_dates,
            regime, shock_score, confidence,
            decomposition  (trend/seasonal/cycle/shock with model/label/value/pct),
            weights, metrics, importance, uncertainty, decision
        """
        prices  = df[PRICE_COL].values.astype(float)
        horizon = FORECAST_DAYS   # always 7 — must match gating network

        if len(prices) < SEQUENCE_LEN + 10:
            raise ValueError(
                f"Need ≥{SEQUENCE_LEN + 10} rows. Got {len(prices)}."
            )

        # ── Regime detection ───────────────────────────────────────────────
        df_reg       = self.sd.detect(df.copy().reset_index(drop=True))
        regime       = self.sd.get_regime(df_reg)
        latest_shock = float(df_reg["shock_score"].iloc[-1])
        recent_vol   = float(df_reg["roll_std"].iloc[-1])

        # ── VMD K=4 decomposition ──────────────────────────────────────────
        vd_local = VMDDecomposer(K=4)
        modes_df = vd_local.decompose(prices)   # Trend | Seasonal | Cycle | Shock

        # ── Scale each mode using TRAINING scalers (no scale mismatch) ─────
        def _get_seq(mode_name: str) -> np.ndarray:
            """Returns (1, seq_len, 1) scaled input for model inference."""
            vals = modes_df[mode_name].values
            if self.mode_scalers and mode_name in self.mode_scalers:
                sc      = self.mode_scalers[mode_name]
                clipped = np.clip(vals,
                                  sc.data_min_[0] * 1.5,
                                  sc.data_max_[0] * 1.5)
                scaled  = sc.transform(clipped.reshape(-1, 1)).flatten()
            else:
                from sklearn.preprocessing import MinMaxScaler
                sc     = MinMaxScaler()
                window = vals[-SEQUENCE_LEN * 3:]
                sc.fit(window.reshape(-1, 1))
                scaled = sc.transform(vals.reshape(-1, 1)).flatten()
            seq = scaled[-SEQUENCE_LEN:]
            return seq[np.newaxis, :, np.newaxis].astype(np.float32)

        trend_seq  = _get_seq("Trend")
        season_seq = _get_seq("Seasonal")
        cycle_seq  = _get_seq("Cycle")
        shock_seq  = _get_seq("Shock")

        # ── Inverse-scale helper ───────────────────────────────────────────
        def _inv(arr: np.ndarray, mode_name: str) -> np.ndarray:
            a = np.array(arr).flatten()[:horizon]
            if self.mode_scalers and mode_name in self.mode_scalers:
                sc = self.mode_scalers[mode_name]
            else:
                from sklearn.preprocessing import MinMaxScaler
                sc = MinMaxScaler()
                sc.fit(modes_df[mode_name].values[-SEQUENCE_LEN * 3:].reshape(-1, 1))
            return sc.inverse_transform(a.reshape(-1, 1)).flatten()

        # ── Model inference ────────────────────────────────────────────────
        if self._loaded and TF_AVAILABLE:
            raw_trend  = self.lstm_model.predict(trend_seq,  verbose=0)[0]
            raw_season = self.cnn_model.predict(season_seq,  verbose=0)[0]
            raw_shock  = self.trans_model.predict(shock_seq, verbose=0)[0]
            if self.bili_model is not None:
                raw_cycle = self.bili_model.predict(cycle_seq, verbose=0)[0]
            else:
                raw_cycle = self._linear_fallback(modes_df["Cycle"].values, horizon)
                if self.mode_scalers and "Cycle" in self.mode_scalers:
                    sc = self.mode_scalers["Cycle"]
                    raw_cycle = sc.transform(
                        raw_cycle.reshape(-1, 1)).flatten()
        else:
            raw_trend, raw_season, raw_cycle, raw_shock = \
                self._statistical_fallback(modes_df, horizon)

        # ── Convert to price-space ─────────────────────────────────────────
        p_trend  = _inv(raw_trend,  "Trend")
        p_season = _inv(raw_season, "Seasonal")
        p_cycle  = _inv(raw_cycle,  "Cycle")
        p_shock  = _inv(raw_shock,  "Shock")

        # ── Ensemble combination ───────────────────────────────────────────
        gate         = self.gate or StaticGating()
        final_series = gate.combine(p_trend, p_season, p_cycle, p_shock, regime)
        final_series = np.array(final_series).flatten()

        weights = (gate.get_weights(regime)
                   if isinstance(gate, StaticGating)
                   else ENSEMBLE_WEIGHTS.get(regime, ENSEMBLE_WEIGHTS["Normal"]))

        # Sanity clip — no prediction beyond 40%–250% of recent range
        p_hist       = prices[-365:] if len(prices) >= 365 else prices
        final_series = np.clip(final_series, p_hist.min() * 0.4, p_hist.max() * 2.5)
        final_price  = float(final_series[0])

        # ── Confidence (methodology §5: uncertainty-aware) ─────────────────
        mean_30    = float(prices[-30:].mean())
        vol_pct    = (recent_vol / (mean_30 + 1e-8)) * 100
        confidence = float(np.clip(100 - vol_pct * 3, 40, 98))

        # ── Forecast dates ─────────────────────────────────────────────────
        last_date_ts   = pd.to_datetime(df[DATE_COL].iloc[-1])
        forecast_dates = [last_date_ts + timedelta(days=i + 1)
                          for i in range(horizon)]

        # ── Decomposition contributions (always includes 'model' key) ──────
        w   = weights
        c_t = abs(float(w.get("trend",    0.35)) * p_trend[0])
        c_s = abs(float(w.get("seasonal", 0.25)) * p_season[0])
        c_c = abs(float(w.get("cycle",    0.15)) * p_cycle[0])
        c_k = abs(float(w.get("shock",    0.25)) * p_shock[0])
        tot = c_t + c_s + c_c + c_k + 1e-8

        decomp = {
            "trend": {
                "value":            round(float(p_trend[0]),  2),
                "contribution_pct": round(c_t / tot * 100,    1),
                "label":            _MODE_LABELS["trend"],
                "model":            _MODEL_NAMES["trend"],
            },
            "seasonal": {
                "value":            round(float(p_season[0]), 2),
                "contribution_pct": round(c_s / tot * 100,    1),
                "label":            _MODE_LABELS["seasonal"],
                "model":            _MODEL_NAMES["seasonal"],
            },
            "cycle": {
                "value":            round(float(p_cycle[0]),  2),
                "contribution_pct": round(c_c / tot * 100,    1),
                "label":            _MODE_LABELS["cycle"],
                "model":            _MODEL_NAMES["cycle"],
            },
            "shock": {
                "value":            round(float(p_shock[0]),  2),
                "contribution_pct": round(c_k / tot * 100,    1),
                "label":            _MODE_LABELS["shock"],
                "model":            _MODEL_NAMES["shock"],
            },
        }

        # ── MC Dropout uncertainty: P̂ ± σ  (methodology §5) ─────────────
        uncertainty = self._mc_dropout_uncertainty(
            trend_seq, season_seq, cycle_seq, shock_seq,
            regime, _inv, gate, horizon, n_samples=20,
        )

        # ── Decision: Hold / Sell / Wait  (methodology §7.4) ──────────────
        decision = self._make_decision(
            final_series, prices, regime, uncertainty, mean_30
        )

        # ── Gradient importance  (methodology §explainability) ─────────────
        importance = self._gradient_importance(
            trend_seq, season_seq, cycle_seq, shock_seq
        )

        # ── Load training metrics ──────────────────────────────────────────
        saved_metrics = {}
        mp = RESULT_DIR / "metrics.json"
        if mp.exists():
            try:
                with open(mp) as f:
                    saved_metrics = json.load(f)
            except Exception:
                pass

        return {
            "final_price":      round(final_price, 2),
            "forecast_series":  [round(float(p), 2) for p in final_series],
            "forecast_dates":   [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "regime":           regime,
            "shock_score":      round(latest_shock, 4),
            "confidence":       round(confidence, 1),
            "decomposition":    decomp,
            "weights":          weights,
            "metrics":          saved_metrics,
            "importance":       importance,
            "uncertainty":      uncertainty,
            "decision":         decision,
            "target_date":      str(target_date or forecast_dates[0].date()),
        }

    # ── MC Dropout uncertainty  (methodology §5) ──────────────────────────
    def _mc_dropout_uncertainty(self,
                                t_seq, s_seq, c_seq, k_seq,
                                regime, inv_fn, gate, horizon,
                                n_samples: int = 20) -> dict:
        """
        Monte Carlo Dropout uncertainty estimation.

        Runs each model N times with dropout ENABLED (training=True).
        Computes per-day mean and std across the stochastic samples.

        Returns
        -------
        {
            "lower"     : [horizon floats],   # mean − 1.96σ (95% CI lower)
            "upper"     : [horizon floats],   # mean + 1.96σ (95% CI upper)
            "std_mean"  : float,              # average std across horizon days
            "n_samples" : int,
        }
        """
        fallback = {"lower": None, "upper": None, "std_mean": None, "n_samples": 0}

        if not (self._loaded and TF_AVAILABLE):
            return fallback

        import tensorflow as tf

        samples = []
        for _ in range(n_samples):
            try:
                rt = self.lstm_model(t_seq, training=True).numpy()[0]
                rs = self.cnn_model(s_seq,  training=True).numpy()[0]
                rk = self.trans_model(k_seq, training=True).numpy()[0]
                if self.bili_model is not None:
                    rc = self.bili_model(c_seq, training=True).numpy()[0]
                else:
                    rc = np.zeros(horizon, dtype=np.float32)

                pt = inv_fn(rt, "Trend");    ps = inv_fn(rs, "Seasonal")
                pc = inv_fn(rc, "Cycle");    pk = inv_fn(rk, "Shock")

                combined = gate.combine(pt, ps, pc, pk, regime)
                arr = np.array(combined).flatten()[:horizon]

                # Sanity clip per sample
                if self.mode_scalers:
                    low = self.mode_scalers["Trend"].data_min_[0] * 0.3
                    high = self.mode_scalers["Trend"].data_max_[0] * 3.0
                    arr = np.clip(arr, low, high)
                samples.append(arr)
            except Exception:
                continue

        if len(samples) < 5:
            return fallback

        S        = np.array(samples)          # (n_samples, horizon)
        mean     = S.mean(axis=0)
        std      = S.std(axis=0)
        lower    = mean - 1.96 * std
        upper    = mean + 1.96 * std

        return {
            "lower":     [round(float(v), 2) for v in lower],
            "upper":     [round(float(v), 2) for v in upper],
            "std_mean":  round(float(std.mean()), 2),
            "n_samples": len(samples),
        }

    # ── Decision: Hold / Sell / Wait  (methodology §7.4) ─────────────────
    @staticmethod
    def _make_decision(final_series: np.ndarray,
                       prices: np.ndarray,
                       regime: str,
                       uncertainty: dict,
                       mean_30: float) -> dict:
        """
        Convert predictions into an actionable recommendation.

        Logic
        -----
        1. Compute expected 4-day price change %
        2. Compute volatility score (MC std or rolling std)
        3. Apply decision rules:
           Shock regime OR high vol  → WAIT  (risky)
           price rising > +5%        → HOLD  (delay sale for better price)
           price falling < -5%       → SELL  (exit before drop)
           else                      → WAIT  (stable, no strong signal)
        """
        last_actual = float(prices[-1])

        # Expected change at day-4 horizon
        mid_idx         = min(3, len(final_series) - 1)
        mid_price       = float(final_series[mid_idx])
        expected_chg_pct = (mid_price - last_actual) / (last_actual + 1e-8) * 100

        # Volatility score
        std_val = uncertainty.get("std_mean")
        vol_score = (float(std_val) / (mean_30 + 1e-8) * 100
                     if std_val is not None else 10.0)

        # Decision rules
        if regime == "Shock" or vol_score > 15:
            action = "WAIT"
            color  = "#f6a623"
            icon   = "🟡"
            reason = (
                "⚠ High market volatility or shock detected. "
                "Avoid large buy/sell commitments. Wait 3–5 days "
                "for prices to stabilise before acting."
            )
        elif expected_chg_pct > 5:
            action = "HOLD"
            color  = "#39d353"
            icon   = "🟢"
            reason = (
                f"📈 Prices expected to rise {expected_chg_pct:+.1f}% "
                f"by day {mid_idx + 1}. Delay selling to capture "
                "higher returns. Review again in 3 days."
            )
        elif expected_chg_pct < -5:
            action = "SELL"
            color  = "#ff6b6b"
            icon   = "🔴"
            reason = (
                f"📉 Prices expected to fall {expected_chg_pct:+.1f}% "
                f"by day {mid_idx + 1}. Consider selling now to "
                "avoid losses. Act within 1–2 days."
            )
        else:
            action = "WAIT"
            color  = "#58a6ff"
            icon   = "🔵"
            reason = (
                "Market is stable with no strong directional signal. "
                "Continue monitoring daily. No immediate action needed."
            )

        return {
            "action":           action,
            "color":            color,
            "icon":             icon,
            "reason":           reason,
            "expected_chg_pct": round(expected_chg_pct, 1),
            "volatility_score": round(vol_score, 1),
        }

    # ── Gradient importance  (methodology §explainability) ────────────────
    def _gradient_importance(self, t_seq, s_seq, c_seq, k_seq) -> dict:
        """
        |∂(mean prediction) / ∂X_t| for each of the 30 input days.

        Shows which historical days most influenced the current prediction.
        """
        n   = SEQUENCE_LEN
        uni = [1.0 / n] * n

        if not (self._loaded and TF_AVAILABLE):
            return {"trend": uni, "seasonal": uni, "cycle": uni,
                    "shock": uni, "combined": uni}

        import tensorflow as tf

        def _grad(model, seq):
            if model is None:
                return uni[:]
            try:
                x = tf.Variable(seq.astype(np.float32))
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(model(x, training=False))
                g   = tape.gradient(loss, x)
                imp = np.abs(g.numpy()).mean(axis=-1).flatten()
                s   = imp.sum() + 1e-8
                return (imp / s).tolist()
            except Exception:
                return uni[:]

        t_imp = _grad(self.lstm_model,  t_seq)
        s_imp = _grad(self.cnn_model,   s_seq)
        c_imp = _grad(self.bili_model,  c_seq)
        k_imp = _grad(self.trans_model, k_seq)

        combined = (np.array(t_imp) + np.array(s_imp)
                    + np.array(c_imp) + np.array(k_imp))
        combined = (combined / (combined.sum() + 1e-8)).tolist()

        return {"trend": t_imp, "seasonal": s_imp,
                "cycle": c_imp, "shock":    k_imp, "combined": combined}

    # ── Static fallbacks ──────────────────────────────────────────────────
    @staticmethod
    def _linear_fallback(series: np.ndarray, horizon: int) -> np.ndarray:
        """Linear extrapolation — used when Bi-LSTM is not trained yet."""
        n    = min(30, len(series))
        x    = np.arange(n)
        y    = series[-n:]
        m, b = np.polyfit(x, y, 1)
        return np.array([m * (n + i) + b for i in range(horizon)])

    @staticmethod
    def _statistical_fallback(modes_df: pd.DataFrame, horizon: int) -> tuple:
        """All-model fallback when TensorFlow is unavailable."""
        from sklearn.preprocessing import MinMaxScaler
        results = []
        for col in ["Trend", "Seasonal", "Cycle", "Shock"]:
            if col not in modes_df.columns:
                results.append(np.zeros(horizon))
                continue
            vals = modes_df[col].values
            n    = min(30, len(vals))
            m, b = np.polyfit(np.arange(n), vals[-n:], 1)
            fut  = np.array([m * (n + i) + b for i in range(horizon)])
            sc   = MinMaxScaler()
            sc.fit(vals.reshape(-1, 1))
            results.append(sc.transform(fut.reshape(-1, 1)).flatten())
        return tuple(results)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not csv_path:
        print("Usage: python predictor.py path/to/onion_data.csv")
        raise SystemExit(1)

    from data_loader import load_data
    df   = load_data(csv_path)
    pred = PricePredictor().load().predict(df)

    print(f"\nFinal price      : ₹{pred['final_price']:,.0f}")
    print(f"Regime           : {pred['regime']}")
    print(f"Confidence       : {pred['confidence']}%")
    print(f"7-day forecast   : {pred['forecast_series']}")
    print(f"Decision         : {pred['decision']['icon']} {pred['decision']['action']}")
    print(f"  {pred['decision']['reason']}")
    print(f"Volatility score : {pred['decision']['volatility_score']}%")
    print(f"Expected change  : {pred['decision']['expected_chg_pct']:+.1f}%")
    print("\nDecomposition:")
    for k, v in pred["decomposition"].items():
        print(f"  {k:10s} [{v['model']:10s}]: ₹{v['value']:+,.0f}  "
              f"({v['contribution_pct']:.0f}%)  — {v['label']}")
    uc = pred.get("uncertainty", {})
    if uc.get("lower"):
        print(f"\nUncertainty (95% CI):")
        print(f"  Lower : {uc['lower']}")
        print(f"  Upper : {uc['upper']}")
        print(f"  Avg σ : ₹{uc['std_mean']:,.0f}  ({uc['n_samples']} MC samples)")