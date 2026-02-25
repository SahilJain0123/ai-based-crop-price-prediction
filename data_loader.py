# data_loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Load & validate your CSV. Handles duplicate dates automatically.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from pathlib import Path
from config import DATE_COL, PRICE_COL, DATA_DIR


def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load onion price CSV → clean → sort → return DataFrame.

    Minimum required CSV columns:
        Arrival_Date   Modal_Price
        2024-01-01     1200
        2024-01-02     1150
    """
    path = Path(csv_path) if csv_path else DATA_DIR / "onion_data.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"\n[DataLoader] CSV not found: {path}\n"
            f"  ➜ Update CSV_PATH in trainer.py and app.py"
        )

    print(f"[DataLoader] Reading: {path}")
    df = pd.read_csv(path)

    # Validate columns
    missing = [c for c in [DATE_COL, PRICE_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"[DataLoader] Missing columns: {missing}\n"
            f"  Found: {list(df.columns)}\n"
            f"  ➜ Update DATE_COL / PRICE_COL in config.py"
        )

    # Parse types
    df[DATE_COL]  = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")

    # Drop rows with bad/missing values
    before = len(df)
    df = df.dropna(subset=[DATE_COL, PRICE_COL])
    dropped = before - len(df)
    if dropped:
        print(f"[DataLoader] Dropped {dropped} invalid rows")

    # Handle duplicate dates — average price per date
    dupes = df[DATE_COL].duplicated().sum()
    if dupes:
        print(f"[DataLoader] Found {dupes} duplicate dates → averaging price per date")
        df = df.groupby(DATE_COL, as_index=False)[PRICE_COL].mean()

    # Sort by date
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, PRICE_COL]].copy()

    print(f"[DataLoader] ✅ {len(df)} records | "
          f"{df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
    print(f"[DataLoader] Price: min=₹{df[PRICE_COL].min():.0f}  "
          f"max=₹{df[PRICE_COL].max():.0f}  mean=₹{df[PRICE_COL].mean():.0f}")
    return df


def generate_synthetic_data(n_days: int = 500, save: bool = True) -> pd.DataFrame:
    """Generate realistic synthetic onion prices for testing."""
    np.random.seed(42)
    dates   = pd.date_range(end="2026-02-21", periods=n_days, freq="D")
    t       = np.arange(n_days)
    trend   = 800 + 0.25 * t
    yearly  = 120 * np.sin(2 * np.pi * t / 365 + 1.0)
    monthly = 50  * np.sin(2 * np.pi * t / 30)
    noise   = np.random.normal(0, 25, n_days)

    shocks = np.zeros(n_days)
    shock_days = np.random.choice(range(60, n_days - 20), size=5, replace=False)
    for sd in shock_days:
        mag = np.random.choice([-1, 1]) * np.random.uniform(250, 500)
        for j in range(7):
            if sd + j < n_days:
                shocks[sd + j] = mag * (0.65 ** j)

    prices = np.clip(trend + yearly + monthly + noise + shocks, 300, 4000).round(2)
    df = pd.DataFrame({DATE_COL: dates, PRICE_COL: prices})

    if save:
        out = DATA_DIR / "onion_data.csv"
        df.to_csv(out, index=False)
        print(f"[DataLoader] Synthetic data saved → {out}")
    return df


if __name__ == "__main__":
    try:
        df = load_data()
    except FileNotFoundError:
        print("[DataLoader] Generating synthetic data for testing...")
        df = generate_synthetic_data(save=True)
        df = load_data()
    print(df.tail(3))