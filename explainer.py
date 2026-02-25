# explainer.py
# ─────────────────────────────────────────────────────────────────────────────
# Explainability: SHAP values + feature importance + attention weights.
#
# Provides human-readable answers to "Why this price?"
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import PRICE_COL, DATE_COL, SEQUENCE_LEN, SHAP_BG_SAMPLES

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[Explainer] SHAP not installed. Run: pip install shap")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class Explainer:
    """
    Provides three types of explanations:

    1. SHAP feature importance — which input days mattered most
    2. Decomposition contribution — how much each VMD mode contributed
    3. Attention proxy — which days the Transformer focused on
    """

    def __init__(self):
        self.shap_values   = None
        self.feature_names = None

    # ── 1. SHAP Explanation ───────────────────────────────────────────────
    def shap_explain(self,
                     model,
                     X_input: np.ndarray,
                     background: np.ndarray,
                     n_bg: int = SHAP_BG_SAMPLES) -> dict:
        """
        Compute SHAP values for a model prediction.

        Parameters
        ----------
        model      : trained Keras model
        X_input    : shape (1, seq_len, 1) — the input sequence to explain
        background : shape (N, seq_len, 1) — background training samples
        n_bg       : number of background samples to use

        Returns
        -------
        dict with:
            shap_values   : (seq_len,) importance for each lag day
            top_days      : list of (day_label, importance) sorted desc
            summary       : human-readable string
        """
        if not (SHAP_AVAILABLE and TF_AVAILABLE):
            return self._shap_fallback(X_input)

        bg = background[np.random.choice(len(background),
                                         min(n_bg, len(background)),
                                         replace=False)]

        # Reshape for SHAP: (N, seq_len)
        bg_2d    = bg.reshape(len(bg), -1)
        X_2d     = X_input.reshape(1, -1)

        # Wrap model to accept 2D input
        def model_fn(x):
            x3d = x.reshape(-1, X_input.shape[1], 1)
            return model.predict(x3d, verbose=0)

        try:
            explainer    = shap.KernelExplainer(model_fn, bg_2d)
            shap_vals    = explainer.shap_values(X_2d, nsamples=100, l1_reg="num_features(20)")
            shap_vals    = np.array(shap_vals)

            # Average over forecast horizon if multi-output
            if shap_vals.ndim == 3:
                shap_vals = shap_vals.mean(axis=0)  # (1, seq_len)
            sv = shap_vals.flatten()                # (seq_len,)

        except Exception as e:
            print(f"[Explainer] SHAP error: {e}. Using gradient fallback.")
            sv = self._gradient_importance(model, X_input)

        self.shap_values = sv
        seq_len  = len(sv)
        days     = [f"t-{seq_len - i}" for i in range(seq_len)]
        top_idx  = np.argsort(np.abs(sv))[::-1][:10]
        top_days = [(days[i], round(float(sv[i]), 5)) for i in top_idx]

        # Build summary
        most_important = top_days[0]
        summary = (f"Day {most_important[0]} had the strongest influence "
                   f"(SHAP={most_important[1]:+.4f}).")

        return {
            "shap_values":   sv.tolist(),
            "feature_names": days,
            "top_days":      top_days,
            "summary":       summary,
        }

    def _gradient_importance(self, model, X_input: np.ndarray) -> np.ndarray:
        """Gradient-based importance as SHAP fallback."""
        if not TF_AVAILABLE:
            return np.random.rand(X_input.shape[1])
        X_tf = tf.Variable(X_input.astype(np.float32))
        with tf.GradientTape() as tape:
            pred = model(X_tf)
            loss = tf.reduce_mean(pred)
        grads = tape.gradient(loss, X_tf)
        importance = np.abs(grads.numpy()).flatten()[:X_input.shape[1]]
        return importance / (importance.max() + 1e-8)

    def _shap_fallback(self, X_input: np.ndarray) -> dict:
        """When SHAP is unavailable, return placeholder importance."""
        seq_len = X_input.shape[1]
        # Recency bias: recent days have more importance
        sv   = np.exp(np.linspace(-2, 0, seq_len))
        sv  /= sv.sum()
        days = [f"t-{seq_len - i}" for i in range(seq_len)]
        return {
            "shap_values":   sv.tolist(),
            "feature_names": days,
            "top_days":      [(days[i], round(float(sv[i]), 5))
                              for i in np.argsort(sv)[::-1][:5]],
            "summary":       "SHAP unavailable — showing recency-based importance.",
        }

    # ── 2. Decomposition Contribution ────────────────────────────────────
    @staticmethod
    def decomposition_report(decomp: dict, regime: str) -> str:
        """
        Convert decomposition dict → human-readable explanation.

        Parameters
        ----------
        decomp : dict from PricePredictor.predict()["decomposition"]
        regime : current market regime
        """
        lines = [f"📊 Market Regime: **{regime}**\n"]

        for component, info in decomp.items():
            pct = info["contribution_pct"]
            val = info["value"]
            lbl = info["label"]
            sign = "+" if val >= 0 else ""
            icon = {"trend": "📈", "seasonal": "🔄", "shock": "⚡"}.get(component, "•")

            lines.append(
                f"{icon} **{component.capitalize()}** ({pct:.0f}% influence)\n"
                f"   Effect: {sign}₹{abs(val):.1f} — {lbl}"
            )

        total = sum(v["value"] for v in decomp.values())
        lines.append(f"\n🎯 **Final Predicted Price: ₹{total:.2f}**")
        return "\n\n".join(lines)

    # ── 3. Attention Proxy ────────────────────────────────────────────────
    @staticmethod
    def attention_proxy(shock_series: np.ndarray,
                        window: int = 30) -> dict:
        """
        Estimate which historical days the Transformer would attend to,
        based on shock magnitude (proxy for real attention weights).

        Returns dict with attention weights (sum to 1) for each lag.
        """
        recent = shock_series[-window:]
        # Attention ∝ |shock| — larger shocks get higher weight
        weights = np.abs(recent) + 1e-6
        weights = weights / weights.sum()

        top_idx    = np.argsort(weights)[::-1][:5]
        top_events = [(f"t-{window - i}", round(float(weights[i]), 4))
                      for i in top_idx]

        return {
            "attention_weights": weights.tolist(),
            "top_attention_days": top_events,
            "summary": (
                f"Highest attention on {top_events[0][0]} "
                f"(weight={top_events[0][1]:.3f}). "
                f"{'Recent shock detected.' if weights[-1] > 0.1 else 'No recent shock.'}"
            )
        }

    # ── 4. Full Explanation Report ────────────────────────────────────────
    def full_report(self,
                    result: dict,
                    shock_series: np.ndarray = None) -> dict:
        """
        Build complete explanation from predictor output dict.

        Parameters
        ----------
        result       : dict from PricePredictor.predict()
        shock_series : raw shock mode values (for attention proxy)

        Returns
        -------
        dict with all explanation components
        """
        decomp_text = self.decomposition_report(
            result["decomposition"], result["regime"])

        attn = None
        if shock_series is not None:
            attn = self.attention_proxy(shock_series)

        regime_advice = {
            "Calm":   "✅ Market is stable. Trend and seasonality dominate. Good time to plan purchases.",
            "Normal": "🟡 Normal market conditions. Monitor for upcoming harvest or weather events.",
            "Shock":  "🔴 Shock detected. Price may be volatile. Wait for stabilisation before bulk buying.",
        }.get(result["regime"], "")

        # Confidence breakdown
        conf = result["confidence"]
        if conf >= 80:
            conf_text = f"High confidence ({conf}%) — stable recent price history."
        elif conf >= 60:
            conf_text = f"Moderate confidence ({conf}%) — some recent volatility."
        else:
            conf_text = f"Low confidence ({conf}%) — high volatility detected."

        return {
            "decomposition_text":  decomp_text,
            "regime_advice":       regime_advice,
            "confidence_text":     conf_text,
            "attention":           attn,
            "weights_used":        result["weights"],
            "shock_score":         result["shock_score"],
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with dummy prediction result
    dummy_result = {
        "final_price": 962.84,
        "regime": "Normal",
        "shock_score": 0.32,
        "confidence": 78.5,
        "decomposition": {
            "trend":    {"value": 940.0,  "contribution_pct": 65.0, "label": "Long-term direction"},
            "seasonal": {"value": 15.0,   "contribution_pct": 20.0, "label": "Harvest cycle"},
            "shock":    {"value": 7.84,   "contribution_pct": 15.0, "label": "Supply disruption"},
        },
        "weights": {"trend": 0.4, "seasonal": 0.3, "shock": 0.3},
    }

    exp    = Explainer()
    report = exp.full_report(dummy_result, shock_series=np.random.randn(100))

    print(report["decomposition_text"])
    print("\n" + report["regime_advice"])
    print("\n" + report["confidence_text"])
    if report["attention"]:
        print(f"\nTop attention days: {report['attention']['top_attention_days']}")