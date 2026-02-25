# preprocessor.py
# ─────────────────────────────────────────────────────────────────────────────
# MinMax scaling, sequence building, feature engineering, train/test split.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from config import (
    DATE_COL, PRICE_COL, TRAIN_RATIO,
    SEQUENCE_LEN, FORECAST_DAYS, MODEL_DIR
)

SCALER_PATH = MODEL_DIR / "scaler.pkl"


class Preprocessor:
    """
    Fit on training prices → scale → make sequences → inverse-transform predictions.

    Typical flow
    ────────────
    pp = Preprocessor()
    train_df, test_df = pp.split(df)
    train_scaled      = pp.fit_transform(train_df[PRICE_COL].values)
    test_scaled       = pp.transform(test_df[PRICE_COL].values)
    X_tr, y_tr        = pp.make_sequences(train_scaled)
    """

    def __init__(self):
        self.scaler  = MinMaxScaler(feature_range=(0, 1))
        self._fitted = False

    # ── Train / Test Split ────────────────────────────────────────────────────
    def split(self, df: pd.DataFrame):
        n      = len(df)
        cutoff = int(n * TRAIN_RATIO)
        train  = df.iloc[:cutoff].copy().reset_index(drop=True)
        test   = df.iloc[cutoff:].copy().reset_index(drop=True)
        print(f"[Preprocessor] Split → train={len(train)}, test={len(test)}")
        return train, test

    # ── Scaling ───────────────────────────────────────────────────────────────
    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        scaled = self.scaler.fit_transform(values.reshape(-1, 1)).flatten()
        self._fitted = True
        return scaled

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.scaler.transform(values.reshape(-1, 1)).flatten()

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.scaler.inverse_transform(
            np.array(values).reshape(-1, 1)
        ).flatten()

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")

    # ── Sequence Builder ──────────────────────────────────────────────────────
    def make_sequences(
        self,
        series: np.ndarray,
        seq_len: int = SEQUENCE_LEN,
        horizon: int = FORECAST_DAYS,
    ):
        """
        Convert 1-D scaled series → supervised (X, y) sequences.

        X shape: (samples, seq_len, 1)
        y shape: (samples, horizon)

        Example  seq_len=3, horizon=2, series=[a,b,c,d,e,f]:
            X[0]=[a,b,c] → y[0]=[d,e]
            X[1]=[b,c,d] → y[1]=[e,f]
        """
        X, y = [], []
        N = len(series)
        for i in range(N - seq_len - horizon + 1):
            X.append(series[i: i + seq_len])
            y.append(series[i + seq_len: i + seq_len + horizon])
        X = np.array(X)[..., np.newaxis]   # (N, seq_len, 1)
        y = np.array(y)                     # (N, horizon)
        print(f"[Preprocessor] Sequences → X={X.shape}, y={y.shape}")
        return X, y

    # ── Feature Engineering ───────────────────────────────────────────────────
    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar + lag + rolling features."""
        df = df.copy()
        dates = pd.to_datetime(df[DATE_COL])
        df["dow"]   = dates.dt.dayofweek
        df["dom"]   = dates.dt.day
        df["month"] = dates.dt.month
        df["woy"]   = dates.dt.isocalendar().week.astype(int)

        for lag in [1, 3, 7, 14, 30]:
            df[f"lag_{lag}"] = df[PRICE_COL].shift(lag)

        for w in [7, 14, 30]:
            df[f"rmean_{w}"] = df[PRICE_COL].rolling(w).mean()
            df[f"rstd_{w}"]  = df[PRICE_COL].rolling(w).std()

        df["pct1"]  = df[PRICE_COL].pct_change(1)
        df["pct7"]  = df[PRICE_COL].pct_change(7)
        df["pct30"] = df[PRICE_COL].pct_change(30)

        return df.dropna().reset_index(drop=True)

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, path: Path = SCALER_PATH):
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[Preprocessor] Scaler saved → {path}")

    def load(self, path: Path = SCALER_PATH):
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)
        self._fitted = True
        print(f"[Preprocessor] Scaler loaded ← {path}")
        return self