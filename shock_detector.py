# shock_detector.py
# ─────────────────────────────────────────────────────────────────────────────
# Detects market regimes: Calm / Normal / Shock
#
# Mathematical logic:
#   ΔP(t) = |P(t) - P(t-1)|
#   rolling_mean = mean(ΔP, window)
#   rolling_std  = std(ΔP, window)
#   threshold    = rolling_mean + Z * rolling_std
#
#   if ΔP(t) > threshold      → SHOCK
#   elif ΔP(t) > rolling_mean → NORMAL
#   else                       → CALM
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from config import PRICE_COL, SHOCK_Z_THRESH, SHOCK_WINDOW


class ShockDetector:
    """
    Classifies each time point as Calm / Normal / Shock.

    Usage
    ─────
    sd  = ShockDetector()
    df  = sd.detect(df)          # adds columns: delta, regime, shock_score
    reg = sd.get_regime(df)      # latest regime string
    """

    REGIMES = ["Calm", "Normal", "Shock"]

    def __init__(self, z_thresh: float = SHOCK_Z_THRESH,
                 window: int = SHOCK_WINDOW):
        self.z      = z_thresh
        self.window = window

    # ── Main detection ────────────────────────────────────────────────────
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime columns to DataFrame.

        New columns
        ───────────
        delta       : |P(t) - P(t-1)|
        roll_mean   : rolling mean of delta
        roll_std    : rolling std  of delta
        threshold   : roll_mean + Z * roll_std
        shock_score : delta / threshold  (>1 means shock)
        regime      : "Calm" | "Normal" | "Shock"
        """
        df = df.copy()
        price = df[PRICE_COL].values

        delta      = np.abs(np.diff(price, prepend=price[0]))
        roll_mean  = pd.Series(delta).rolling(self.window, min_periods=1).mean().values
        roll_std   = pd.Series(delta).rolling(self.window, min_periods=1).std().fillna(0).values
        threshold  = roll_mean + self.z * roll_std

        shock_score = np.where(threshold > 0, delta / threshold, 0.0)

        regimes = []
        for d, th, rm in zip(delta, threshold, roll_mean):
            if d > th:
                regimes.append("Shock")
            elif d > rm:
                regimes.append("Normal")
            else:
                regimes.append("Calm")

        df["delta"]       = delta.round(2)
        df["roll_mean"]   = roll_mean.round(2)
        df["roll_std"]    = roll_std.round(2)
        df["threshold"]   = threshold.round(2)
        df["shock_score"] = shock_score.round(4)
        df["regime"]      = regimes

        # Summary
        counts = pd.Series(regimes).value_counts()
        total  = len(regimes)
        print("[ShockDetector] Regime distribution:")
        for r in self.REGIMES:
            c = counts.get(r, 0)
            print(f"  {r:8s}: {c:5d} ({c/total*100:.1f}%)")

        n_shocks = counts.get("Shock", 0)
        if n_shocks > 0:
            shock_dates = df.loc[df["regime"] == "Shock", "Arrival_Date"] \
                           if "Arrival_Date" in df.columns else None
            if shock_dates is not None:
                print(f"  ⚡ Shock dates: {list(shock_dates.dt.date)[:10]}")

        return df

    # ── Current regime ────────────────────────────────────────────────────
    @staticmethod
    def get_regime(df: pd.DataFrame) -> str:
        """Return the most recent regime label."""
        return df["regime"].iloc[-1]

    # ── Regime for a new (unseen) price given the last N days ────────────
    def predict_regime(self, history: np.ndarray, new_price: float) -> dict:
        """
        Given the last `window` prices and a new incoming price,
        return its regime + shock score in real time.
        """
        delta = abs(new_price - history[-1])
        recent_deltas = np.abs(np.diff(history[-self.window:]))
        mu  = recent_deltas.mean()
        sig = recent_deltas.std() if len(recent_deltas) > 1 else 0
        th  = mu + self.z * sig
        score = delta / (th + 1e-8)

        if delta > th:
            regime = "Shock"
        elif delta > mu:
            regime = "Normal"
        else:
            regime = "Calm"

        return {
            "regime":      regime,
            "delta":       round(delta, 2),
            "threshold":   round(th, 2),
            "shock_score": round(score, 4),
            "mu":          round(mu, 2),
            "sigma":       round(sig, 2),
        }

    # ── Volatility series (used by Explainer) ────────────────────────────
    @staticmethod
    def rolling_volatility(prices: np.ndarray, window: int = 30) -> np.ndarray:
        """Rolling std of returns — proxy for market volatility."""
        returns = pd.Series(prices).pct_change().fillna(0)
        vol     = returns.rolling(window, min_periods=1).std().values
        return vol


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    df  = generate_synthetic_data(save=False)
    sd  = ShockDetector()
    df2 = sd.detect(df)
    print(df2[["Arrival_Date", "Modal_Price", "delta",
               "threshold", "shock_score", "regime"]].tail(10))

    reg = sd.get_regime(df2)
    print(f"\nLatest regime: {reg}")

    # Live regime test
    info = sd.predict_regime(df["Modal_Price"].values[-60:], new_price=500.0)
    print(f"\nLive regime prediction: {info}")