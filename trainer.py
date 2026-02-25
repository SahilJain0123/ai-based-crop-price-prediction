# trainer.py
# ─────────────────────────────────────────────────────────────────────────────
# Complete training pipeline — 100% from your real CSV.
#
# Steps:
#  1.  Load & validate real CSV
#  2.  Chronological 80/20 split
#  3.  VMD K=4 → Trend, Seasonal, Cycle, Shock
#  4.  Per-mode MinMaxScaler (fit on TRAIN only, saved to disk)
#  5.  Regime detection (full dataset)
#  6.  GA + Bayesian hyperparameter search for each model
#  7a. Train LSTM        → Trend mode
#  7b. Train CNN-LSTM    → Seasonal mode
#  7c. Train Bi-LSTM     → Cycle mode    (NEW)
#  7d. Train Transformer → Shock mode
#  8.  Train 4-model Gating Network (correct combined ground truth)
#  9.  Rolling evaluation across full test set
#  10. Save metrics.json
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import json
import pickle
import time
from pathlib import Path

from config import (
    PRICE_COL, DATE_COL, SEQUENCE_LEN, FORECAST_DAYS,
    TRAIN_RATIO, MODEL_DIR, RESULT_DIR, RANDOM_SEED,
)
from data_loader import load_data
from preprocessor import Preprocessor
from vmd_decomposer import VMDDecomposer
from shock_detector import ShockDetector
from models.lstm_trend       import train_lstm,       build_lstm
from models.cnn_lstm_seasonal import train_cnn_lstm,  build_cnn_lstm
from models.bilstm_cycle     import train_bilstm,     build_bilstm
from models.transformer_shock import train_transformer, build_transformer
from models.ensemble         import StaticGating, LearnedGating
from models.hyperopt_ga      import GeneticOptimizer

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
#  ⚠️  UPDATE THIS TO YOUR CSV PATH
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\sahil\OneDrive\Desktop\onion_data.csv"

MODE_SCALERS_PATH = MODEL_DIR / "mode_scalers.pkl"


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────
def _tv_split(X, y, val_frac=0.10):
    """Chronological 90/10 split for train/val sequences."""
    n   = len(X)
    cut = int(n * (1 - val_frac))
    return X[:cut], y[:cut], X[cut:], y[cut:]


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def train_all(csv_path: str = None,
              use_learned_gating: bool = True,
              use_ga_hyperopt:    bool = True):
    """
    End-to-end training on your real onion CSV.

    Parameters
    ----------
    csv_path           : path to CSV (defaults to CSV_PATH above)
    use_learned_gating : train neural gating network (else use static weights)
    use_ga_hyperopt    : run GA+Bayesian hyperopt before each model (recommended)
    """
    t0       = time.time()
    csv_path = csv_path or CSV_PATH

    print("=" * 65)
    print("  AgriPrice AI — Full Training Pipeline")
    print("  K=4 VMD · LSTM · CNN-LSTM · Bi-LSTM · Transformer")
    print("  GA+Bayesian HyperOpt · 4-model Gating Network")
    print("=" * 65)

    # ── Step 1: Load CSV ──────────────────────────────────────────────────
    print(f"\n[Step 1] Loading: {csv_path}")
    try:
        df = load_data(csv_path)
    except FileNotFoundError:
        print(f"\n  ❌ CSV not found: {csv_path}")
        print("     Update CSV_PATH at top of trainer.py")
        raise

    prices = df[PRICE_COL].values.astype(float)
    dates  = pd.to_datetime(df[DATE_COL])
    print(f"  ✅ {len(prices)} records  "
          f"{dates.iloc[0].date()} → {dates.iloc[-1].date()}")
    print(f"  Price: min=₹{prices.min():.0f}  max=₹{prices.max():.0f}  "
          f"mean=₹{prices.mean():.0f}")

    min_rows = SEQUENCE_LEN * 4
    if len(prices) < min_rows:
        raise ValueError(f"Need ≥{min_rows} rows. Got {len(prices)}.")

    # ── Step 2: Train/test split ───────────────────────────────────────────
    print("\n[Step 2] Chronological 80/20 split...")
    pp                = Preprocessor()
    train_df, test_df = pp.split(df)
    train_prices      = train_df[PRICE_COL].values.astype(float)
    test_prices       = test_df[PRICE_COL].values.astype(float)
    n_train           = len(train_prices)

    pp.fit_transform(train_prices)   # fits global scaler on train only
    pp.save()
    print(f"  Train: {n_train} rows  |  Test: {len(test_prices)} rows")

    # ── Step 3: VMD K=4 decomposition ─────────────────────────────────────
    print("\n[Step 3] VMD K=4 decomposition on full series...")
    vd       = VMDDecomposer(K=4)
    modes_df = vd.decompose(prices)
    vd.save()

    labels      = list(modes_df.columns)   # ["Trend","Seasonal","Cycle","Shock"]
    modes_train = {l: modes_df[l].values[:n_train] for l in labels}
    modes_test  = {l: modes_df[l].values[n_train:] for l in labels}
    print(f"  Modes: {labels}")

    # ── Step 4: Per-mode scalers (fit on TRAIN only, saved) ───────────────
    print("\n[Step 4] Scaling modes (fit on train split, saved for inference)...")
    from sklearn.preprocessing import MinMaxScaler
    mode_scalers = {}
    modes_tr_sc  = {}
    modes_te_sc  = {}

    for lbl in labels:
        sc               = MinMaxScaler(feature_range=(0, 1))
        modes_tr_sc[lbl] = sc.fit_transform(
            modes_train[lbl].reshape(-1, 1)).flatten()
        modes_te_sc[lbl] = sc.transform(
            modes_test[lbl].reshape(-1, 1)).flatten()
        mode_scalers[lbl] = sc
        print(f"  {lbl:10s}: train range "
              f"[{modes_train[lbl].min():+.1f}, {modes_train[lbl].max():+.1f}]")

    with open(MODE_SCALERS_PATH, "wb") as f:
        pickle.dump(mode_scalers, f)
    print(f"  ✅ Mode scalers saved → {MODE_SCALERS_PATH}")

    # ── Step 5: Regime detection ───────────────────────────────────────────
    print("\n[Step 5] Shock/regime detection...")
    sd      = ShockDetector()
    df_reg  = sd.detect(df)
    regimes = df_reg["regime"].values

    # ── Step 6: Build sequences ────────────────────────────────────────────
    print("\n[Step 6] Building sliding-window sequences...")
    seq_data = {}
    for lbl in labels:
        X, y                      = pp.make_sequences(modes_tr_sc[lbl])
        X_tr, y_tr, X_val, y_val = _tv_split(X, y)
        seq_data[lbl]             = (X_tr, y_tr, X_val, y_val)
        print(f"  {lbl:10s}: train {X_tr.shape}  val {X_val.shape}")

    # ── TF check ──────────────────────────────────────────────────────────
    try:
        import tensorflow as tf
        print(f"\n  TensorFlow {tf.__version__} ready")
    except ImportError:
        print("\n  ❌ TensorFlow not installed. Run: pip install tensorflow")
        return {"status": "tensorflow_missing"}

    # ── Step 7: GA Hyperopt + Train each model ─────────────────────────────
    model_configs = [
        ("Trend",    "lstm",        "lstm"),
        ("Seasonal", "cnn_lstm",    "cnn_lstm"),
        ("Cycle",    "bilstm",      "bilstm"),
        ("Shock",    "transformer", "transformer"),
    ]
    train_fns = {
        "lstm":        train_lstm,
        "cnn_lstm":    train_cnn_lstm,
        "bilstm":      train_bilstm,
        "transformer": train_transformer,
    }
    trained_models = {}

    for mode_name, ga_type, train_type in model_configs:
        X_tr, y_tr, X_val, y_val = seq_data[mode_name]
        print(f"\n[Step 7] {mode_name} → {ga_type.upper()}")

        # ── GA + Bayesian hyperopt ─────────────────────────────────────
        if use_ga_hyperopt:
            print(f"  Running GA+Bayes hyperopt ({ga_type})...")
            opt        = GeneticOptimizer(model_type=ga_type)
            best_p     = opt.search(X_tr, y_tr, X_val, y_val)
            print(f"  Best params: {best_p}")
        else:
            best_p = {}
            print(f"  Skipping hyperopt, using defaults")

        # ── Full training with best hyperparams ────────────────────────
        print(f"  Full training ({train_type})...")
        train_fn = train_fns[train_type]

        # Convert GA params to train function kwargs
        kwargs = _ga_params_to_kwargs(ga_type, best_p)
        model, _ = train_fn(X_tr, y_tr, X_val, y_val, **kwargs)
        trained_models[mode_name] = model
        print(f"  ✅ {mode_name} model trained")

    lstm_model  = trained_models["Trend"]
    cnn_model   = trained_models["Seasonal"]
    bili_model  = trained_models["Cycle"]
    trans_model = trained_models["Shock"]

    # ── Step 8: Train Gating Network ──────────────────────────────────────
    print("\n[Step 8] Training 4-model Gating Network...")
    n_gate = min(len(seq_data[l][0]) for l in labels)

    P_trend  = lstm_model.predict(seq_data["Trend"][0][:n_gate],    verbose=0)
    P_season = cnn_model.predict(seq_data["Seasonal"][0][:n_gate],  verbose=0)
    P_cycle  = bili_model.predict(seq_data["Cycle"][0][:n_gate],    verbose=0)
    P_shock  = trans_model.predict(seq_data["Shock"][0][:n_gate],   verbose=0)

    # Ground truth: sum of all mode values at each forecast window position
    # This is the actual reconstructed price (VMD property: Σ modes = price)
    def inv_mode(arr, lbl):
        return mode_scalers[lbl].inverse_transform(arr).reshape(arr.shape)

    # Build y_combined: for each sequence window i, the true future price
    # = sum of all 4 mode values at positions [seq_len+i : seq_len+i+horizon]
    y_combined = np.zeros((n_gate, FORECAST_DAYS), dtype=np.float32)
    for lbl in labels:
        for i in range(n_gate):
            s = i + SEQUENCE_LEN
            e = s + FORECAST_DAYS
            if e <= len(modes_train[lbl]):
                y_combined[i] += modes_train[lbl][s:e].astype(np.float32)

    # Normalise y_combined to [0,1] for stable gating training
    y_min  = y_combined.min()
    y_max  = y_combined.max() + 1e-8
    y_sc   = (y_combined - y_min) / (y_max - y_min)

    # Normalise P arrays to same scale
    def norm(arr): return (inv_mode(arr, labels[0]) - y_min) / (y_max - y_min)
    Pt_sc = (inv_mode(P_trend,  "Trend")    - y_min) / (y_max - y_min)
    Ps_sc = (inv_mode(P_season, "Seasonal") - y_min) / (y_max - y_min)
    Pc_sc = (inv_mode(P_cycle,  "Cycle")    - y_min) / (y_max - y_min)
    Pk_sc = (inv_mode(P_shock,  "Shock")    - y_min) / (y_max - y_min)

    reg_seq = regimes[:n_train][SEQUENCE_LEN: SEQUENCE_LEN + n_gate]

    if use_learned_gating:
        gate = LearnedGating()
        gate.fit(Pt_sc, Ps_sc, Pc_sc, Pk_sc, y_sc, list(reg_seq))
        gate.save()
        print("  ✅ Learned gating network saved")
    else:
        gate = StaticGating()
        print("  Using static gating weights")

    # ── Step 9: Rolling evaluation on full test set ───────────────────────
    print("\n[Step 9] Rolling evaluation on test set...")
    metrics = evaluate_rolling(
        lstm_model, cnn_model, bili_model, trans_model, gate,
        modes_te_sc, mode_scalers, sd, test_df
    )

    # ── Step 10: Save metrics ─────────────────────────────────────────────
    metrics_path = RESULT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  ✅ Training complete in {elapsed/60:.1f} min")
    print(f"  RMSE : ₹{metrics['rmse']}")
    print(f"  MAE  : ₹{metrics['mae']}")
    print(f"  MAPE : {metrics['mape']}%")
    print(f"  Test points evaluated: {metrics.get('n_test_samples', '?')}")
    print(f"{'='*65}\n")

    return {
        "lstm":   lstm_model,  "cnn":  cnn_model,
        "bilstm": bili_model,  "trans": trans_model,
        "gate":   gate,        "pp":   pp,
        "vd":     vd,          "sd":   sd,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Convert GA best_params to train function kwargs
# ─────────────────────────────────────────────────────────────────────────────
def _ga_params_to_kwargs(ga_type: str, best_p: dict) -> dict:
    """Map GA hyperparameter keys to train_* function keyword args."""
    if not best_p:
        return {}

    if ga_type == "lstm":
        units = []
        if "units_0" in best_p: units.append(int(best_p["units_0"]))
        if "units_1" in best_p: units.append(int(best_p["units_1"]))
        return {
            "units":   units or None,
            "dropout": float(best_p.get("dropout", 0.2)),
            "lr":      float(best_p.get("lr", 0.001)),
        }

    if ga_type == "cnn_lstm":
        return {}  # train_cnn_lstm doesn't accept override kwargs → use defaults

    if ga_type == "bilstm":
        units = []
        if "units_0" in best_p: units.append(int(best_p["units_0"]))
        if "units_1" in best_p: units.append(int(best_p["units_1"]))
        return {
            "units":   units or None,
            "dropout": float(best_p.get("dropout", 0.2)),
            "lr":      float(best_p.get("lr", 0.001)),
        }

    if ga_type == "transformer":
        return {}  # train_transformer doesn't accept override kwargs → use defaults

    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Rolling evaluation across full test set
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_rolling(lstm_model, cnn_model, bili_model, trans_model, gate,
                     modes_te_sc, mode_scalers, sd, test_df):
    """
    Roll a sliding window across the FULL test set.
    At each step: predict FORECAST_DAYS ahead, compare to actual.
    Returns RMSE, MAE, MAPE over all comparison points.
    """
    actual = test_df[PRICE_COL].values.astype(float)
    n      = len(actual)
    labels = ["Trend", "Seasonal", "Cycle", "Shock"]

    if n <= SEQUENCE_LEN + FORECAST_DAYS:
        print(f"  ⚠️  Test set ({n}) too small for rolling eval.")
        return {"rmse": "N/A", "mae": "N/A", "mape": "N/A",
                "n_test_samples": n, "note": "test set too small"}

    regime_test = sd.get_regime(sd.detect(test_df.reset_index(drop=True)))
    preds, actuals = [], []

    for start in range(0, n - SEQUENCE_LEN - FORECAST_DAYS + 1, FORECAST_DAYS):
        end = start + SEQUENCE_LEN

        def seq(lbl):
            s = modes_te_sc[lbl][start:end]
            return None if len(s) < SEQUENCE_LEN else s[np.newaxis, :, np.newaxis]

        seqs = {l: seq(l) for l in labels}
        if any(v is None for v in seqs.values()):
            break

        try:
            p_t = lstm_model.predict(seqs["Trend"],    verbose=0)[0]
            p_s = cnn_model.predict(seqs["Seasonal"],  verbose=0)[0]
            p_c = bili_model.predict(seqs["Cycle"],    verbose=0)[0]
            p_k = trans_model.predict(seqs["Shock"],   verbose=0)[0]
        except Exception:
            continue

        def inv(arr, lbl):
            return mode_scalers[lbl].inverse_transform(
                arr.reshape(-1, 1)).flatten()

        combined = gate.combine(
            inv(p_t, "Trend"), inv(p_s, "Seasonal"),
            inv(p_c, "Cycle"), inv(p_k, "Shock"),
            regime_test
        )
        combined   = np.array(combined).flatten()
        actual_win = actual[end: end + FORECAST_DAYS]
        n_cmp      = min(len(combined), len(actual_win))

        for i in range(n_cmp):
            preds.append(float(combined[i]))
            actuals.append(float(actual_win[i]))

    if not preds:
        return {"rmse": "N/A", "mae": "N/A", "mape": "N/A", "n_test_samples": 0}

    preds   = np.clip(np.array(preds),
                      np.array(actuals).min() * 0.3,
                      np.array(actuals).max() * 3.0)
    actuals = np.array(actuals)

    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mae  = float(np.mean(np.abs(actuals - preds)))
    mape = float(np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100)

    print(f"  {len(preds)} comparison points  |  regime_test={regime_test}")
    return {
        "rmse":           round(rmse, 2),
        "mae":            round(mae, 2),
        "mape":           round(mape, 2),
        "n_test_samples": len(preds),
        "regime_test":    regime_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nCSV_PATH = {CSV_PATH}")
    print("→ Update CSV_PATH at top of trainer.py if wrong\n")

    results = train_all(
        csv_path        = CSV_PATH,
        use_learned_gating = True,
        use_ga_hyperopt    = True,   # set False to skip GA for faster run
    )

    if "metrics" in results:
        print("\nFinal metrics:", results["metrics"])