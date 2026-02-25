# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration — all other files import from here.
# Full methodology: K=4 VMD · LSTM · CNN-LSTM · Bi-LSTM · Transformer
#                   GA+Bayesian HyperOpt · Learned Gating · Gradient Importance
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
MODEL_DIR  = ROOT_DIR / "outputs" / "models"
RESULT_DIR = ROOT_DIR / "outputs" / "results"

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── CSV column names — must match your file exactly ───────────────────────────
DATE_COL  = "Arrival_Date"
PRICE_COL = "Modal_Price"

# ── Data split (always chronological, never shuffle) ──────────────────────────
TRAIN_RATIO   = 0.80   # 80% train / 20% test
SEQUENCE_LEN  = 30     # input window: 30 days of history
FORECAST_DAYS = 7      # output: predict next 7 days

# ── VMD — K=4 as per methodology ─────────────────────────────────────────────
# Decomposes price into 4 frequency bands (sorted by energy):
#   Mode 0 → TREND    (lowest freq)  → LSTM
#   Mode 1 → SEASONAL (mid-low freq) → CNN-LSTM
#   Mode 2 → CYCLE    (mid-high)     → Bi-LSTM
#   Mode 3 → SHOCK    (highest freq) → Transformer
VMD_K         = 4
VMD_ALPHA     = 2000    # bandwidth penalty (larger = smoother modes)
VMD_TAU       = 0.0     # noise tolerance (0 = strict reconstruction)
VMD_TOL       = 1e-7    # ADMM convergence threshold
VMD_MAX_ITER  = 500

# ── Shock / regime detection ──────────────────────────────────────────────────
SHOCK_Z_THRESH = 2.0    # z-score threshold for shock label
SHOCK_WINDOW   = 30     # rolling window for mean/std of |ΔP|

# ── LSTM — Trend model ────────────────────────────────────────────────────────
LSTM_UNITS    = [64, 32]   # stacked LSTM layer sizes
LSTM_DROPOUT  = 0.2
LSTM_EPOCHS   = 80
LSTM_BATCH    = 32
LSTM_LR       = 0.001

# ── CNN-LSTM — Seasonal model ─────────────────────────────────────────────────
CNN_FILTERS    = 64
CNN_KERNEL     = 3
CNNLSTM_UNITS  = 32
CNNLSTM_EPOCHS = 80
CNNLSTM_BATCH  = 32
CNNLSTM_LR     = 0.001

# ── Bi-LSTM — Cycle model (NEW) ───────────────────────────────────────────────
# Bidirectional LSTM: reads sequence forward AND backward.
# Cycles depend on both past and future context — Bi-LSTM captures this.
BILSTM_UNITS   = [64, 32]  # each direction; actual output = 2×units per layer
BILSTM_DROPOUT = 0.2
BILSTM_EPOCHS  = 80
BILSTM_BATCH   = 32
BILSTM_LR      = 0.001

# ── Transformer — Shock model ─────────────────────────────────────────────────
TRANS_D_MODEL  = 32
TRANS_N_HEADS  = 4
TRANS_FF_DIM   = 64
TRANS_N_LAYERS = 2
TRANS_DROPOUT  = 0.1
TRANS_EPOCHS   = 80
TRANS_BATCH    = 32
TRANS_LR       = 0.0005

# ── GA + Bayesian Hyperparameter Optimisation (NEW) ──────────────────────────
# Phase 1 (GA):  explore a population of hyperparameter sets
# Phase 2 (Bayes): refine locally around the best individual found
GA_POPULATION    = 6    # candidate sets per generation
GA_GENERATIONS   = 3    # number of GA generations
GA_QUICK_EPOCHS  = 8    # epochs for fitness evaluation (fast proxy)
GA_MUTATION_RATE = 0.3  # fraction of genes that mutate per offspring

# ── 4-model Ensemble Weights per Regime (updated from 3→4 models) ────────────
# These are STATIC fallback weights. Learned gating overrides them when trained.
# Weights sum to 1.0 per regime: trend + seasonal + cycle + shock = 1.0
ENSEMBLE_WEIGHTS = {
    "Calm":   {"trend": 0.50, "seasonal": 0.25, "cycle": 0.15, "shock": 0.10},
    "Normal": {"trend": 0.35, "seasonal": 0.25, "cycle": 0.15, "shock": 0.25},
    "Shock":  {"trend": 0.15, "seasonal": 0.10, "cycle": 0.10, "shock": 0.65},
}

# ── Training callbacks ────────────────────────────────────────────────────────
RANDOM_SEED      = 42
EARLY_STOP_PAT   = 15   # patience for EarlyStopping
REDUCE_LR_PAT    = 7    # patience for ReduceLROnPlateau
REDUCE_LR_FACTOR = 0.5

# ── Explainability ────────────────────────────────────────────────────────────
SHAP_BG_SAMPLES = 50   # background samples for gradient importance