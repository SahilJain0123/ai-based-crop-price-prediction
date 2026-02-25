# app.py — AgriPrice AI  · Production Build
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data
from predictor import PricePredictor
from shock_detector import ShockDetector
from vmd_decomposer import VMDDecomposer

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — update CSV_PATH to point to your file
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "https://raw.githubusercontent.com/SahilJain0123/ai-based-crop-price-prediction/main/data/onion_data.csv"

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriPrice AI · Ludhiana Onion",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700;800&display=swap');

:root {
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --accent:  #f6a623;
    --blue:    #58a6ff;
    --green:   #39d353;
    --red:     #ff6b6b;
    --muted:   #8b949e;
    --text:    #e6edf3;
}

html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
}
.stApp { background: var(--bg); }

/* Hide sidebar completely */
section[data-testid="stSidebar"] { display: none !important; }
button[data-testid="collapsedControl"] { display: none !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .72rem !important;
    letter-spacing: .08em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stMetricDelta"] {
    font-size: .78rem !important;
    font-family: 'Space Mono', monospace !important;
}

/* Date inputs */
[data-testid="stDateInput"] label {
    color: var(--muted) !important;
    font-size: .75rem !important;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
}

/* Headings */
h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -.02em;
    margin: 0 !important;
}
h2, h3 { font-weight: 700 !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* Brain cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
    height: 100%;
}
.card-title {
    font-size: .9rem;
    font-weight: 700;
    margin: 0 0 6px;
}
.card-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 4px 0;
}
.card-sub {
    font-size: .78rem;
    color: var(--muted);
    margin: 2px 0;
    line-height: 1.5;
}
.pill {
    display: inline-block;
    padding: 1px 9px;
    border-radius: 20px;
    font-size: .65rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: .05em;
    margin-bottom: 6px;
}
.pill-blue  { background: #1c2d4a; color: var(--blue); }
.pill-green { background: #1a3326; color: var(--green); }
.pill-red   { background: #3d1f1f; color: var(--red); }

/* Prediction hero */
.pred-hero {
    background: linear-gradient(135deg, rgba(246,166,35,.12), rgba(22,27,34,.95));
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 22px 24px;
    text-align: center;
}
.pred-hero-price {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--accent);
    margin: 4px 0;
    line-height: 1;
}
.pred-hero-label { font-size: .82rem; color: var(--muted); margin: 4px 0; }

/* Regime banner */
.regime-banner {
    border-radius: 10px;
    padding: 14px 20px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-top: 0;
}
.regime-icon { font-size: 1.6rem; flex-shrink: 0; margin-top: 2px; }
.regime-title { font-size: .9rem; font-weight: 700; margin: 0 0 3px; }
.regime-text  { font-size: .83rem; color: var(--text); margin: 0; line-height: 1.5; }

/* Control bar */
.ctrl-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_market_data():
    return load_data(CSV_PATH)

@st.cache_resource(show_spinner=False)
def load_predictor():
    p = PricePredictor()
    p.load()
    return p

@st.cache_data(show_spinner=False)
def run_vmd(prices_key: tuple) -> pd.DataFrame:
    """Run VMD decomposition on full price series. Cached so it only runs once."""
    vd = VMDDecomposer()
    return vd.decompose(np.array(prices_key))

@st.cache_data(show_spinner=False)
def run_prediction(prices_key: tuple, last_date_str: str):
    """Cache the prediction result. Invalidates when data changes."""
    return None  # placeholder — prediction runs inline below


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading market data..."):
        df = load_market_data()
except FileNotFoundError:
    st.error(
        f"**CSV not found:** `{CSV_PATH}`\n\n"
        "Open `app.py` and update `CSV_PATH` to your file location."
    )
    st.stop()
except Exception as e:
    st.error(f"**Error loading data:** {e}")
    st.stop()

# Validate data has required columns and enough rows
if "Arrival_Date" not in df.columns or "Modal_Price" not in df.columns:
    st.error("CSV must have columns: `Arrival_Date` and `Modal_Price`")
    st.stop()
if len(df) < 50:
    st.error(f"Need at least 50 rows of data. Found {len(df)}.")
    st.stop()

# Key dates
prices_arr = df["Modal_Price"].values.astype(float)
dates_arr  = pd.to_datetime(df["Arrival_Date"])
first_date = dates_arr.iloc[0].date()
last_date  = dates_arr.iloc[-1].date()

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading AI models..."):
    predictor = load_predictor()
sd = ShockDetector()


# ─────────────────────────────────────────────────────────────────────────────
#  RUN PREDICTION (always horizon=7 to match trained gating network)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Running prediction pipeline..."):
    try:
        result = predictor.predict(df, horizon=7)
    except Exception as e:
        st.error(f"**Prediction failed:** {e}\n\nTry retraining: `python trainer.py`")
        st.stop()

regime  = result["regime"]
metrics = result.get("metrics", {})
decomp  = result["decomposition"]
forecast_prices = result["forecast_series"]     # list of 7 floats
forecast_dates  = [pd.to_datetime(d).date() for d in result["forecast_dates"]]


# ─────────────────────────────────────────────────────────────────────────────
#  RUN VMD (cached, full dataset)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Decomposing signal..."):
    try:
        modes_df = run_vmd(tuple(prices_arr.tolist()))
        modes_df.index = dates_arr
        vmd_ok = True
    except Exception:
        vmd_ok = False


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1>🌾 AgriPrice AI</h1>", unsafe_allow_html=True)
st.markdown(
    '<p style="color:var(--muted);font-size:.9rem;margin:2px 0 20px;">'
    'Variational Mode Decomposition · LSTM · CNN-LSTM · Transformer · '
    'Ensemble Gating · Ludhiana Mandi Onion</p>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
#  CONTROLS: Date range + Predict date
#  No rerun() calls — all state handled by Streamlit widget keys natively
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="ctrl-bar">', unsafe_allow_html=True)
cc1, cc2, cc3 = st.columns([1, 1, 1])

with cc1:
    view_start = st.date_input(
        "📅 History From",
        value=max(first_date, last_date - timedelta(days=90)),
        min_value=first_date,
        max_value=last_date,
        key="view_start",
        help="Start of the date range shown in charts",
    )

with cc2:
    view_end = st.date_input(
        "📅 History To",
        value=last_date,
        min_value=first_date,
        max_value=last_date,
        key="view_end",
        help="End of the date range shown in charts",
    )

with cc3:
    predict_date = st.date_input(
        "🎯 Predict Price For",
        value=last_date + timedelta(days=1),
        min_value=last_date + timedelta(days=1),
        max_value=last_date + timedelta(days=7),
        key="pred_date",
        help="Pick any date within 7 days ahead. Model forecasts up to 7 days.",
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Guard: view_start must be before view_end ──────────────────────────────
if view_start >= view_end:
    st.warning("⚠️ 'History From' must be before 'History To'. Showing last 90 days.")
    view_start = max(first_date, last_date - timedelta(days=90))
    view_end   = last_date


# ─────────────────────────────────────────────────────────────────────────────
#  DERIVE TARGET PRICE for chosen predict_date
# ─────────────────────────────────────────────────────────────────────────────
try:
    pred_idx     = forecast_dates.index(predict_date)
    target_price = forecast_prices[pred_idx]
except (ValueError, IndexError):
    # Fallback: nearest available forecast day
    target_price = forecast_prices[0]
    predict_date = forecast_dates[0]

target_price = float(np.clip(target_price, 50, 50000))


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────
regime_icon  = {"Calm": "🟢", "Normal": "🔵", "Shock": "🔴"}.get(regime, "⚪")
regime_color = {"Calm": "#39d353", "Normal": "#58a6ff", "Shock": "#ff6b6b"}.get(regime, "#f6a623")

# 30-day delta
last_actual  = prices_arr[-1]
prev_30      = prices_arr[-31] if len(prices_arr) > 30 else prices_arr[0]
delta_30_pct = f"{(last_actual - prev_30) / (prev_30 + 1e-8) * 100:+.1f}%"
pred_vs_last = f"{(target_price - last_actual) / (last_actual + 1e-8) * 100:+.1f}%"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last Actual Price",   f"₹{last_actual:,.0f}",  delta_30_pct + " vs 30 days")
m2.metric("AI Predicted Price",  f"₹{target_price:,.0f}", pred_vs_last + " vs today")
m3.metric(f"Market Regime",      f"{regime_icon} {regime}", "")
m4.metric("Model Confidence",    f"{result['confidence']}%", "")
m5.metric("RMSE / MAPE",         f"₹{metrics.get('rmse','—')}", f"{metrics.get('mape','—')}% error")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  FILTER DATA FOR VIEW WINDOW
# ─────────────────────────────────────────────────────────────────────────────
mask     = (dates_arr.dt.date >= view_start) & (dates_arr.dt.date <= view_end)
df_view  = df[mask].copy().reset_index(drop=True)

if len(df_view) < 2:
    st.warning("Selected date range has fewer than 2 data points. Please widen the range.")
    st.stop()

# Shock detection on FULL dataset (needs enough history for rolling stats)
df_reg_full = sd.detect(df.copy().reset_index(drop=True))
# Filter shock events to view window for chart
shock_mask   = (pd.to_datetime(df_reg_full["Arrival_Date"]).dt.date >= view_start) & \
               (pd.to_datetime(df_reg_full["Arrival_Date"]).dt.date <= view_end)
shock_view   = df_reg_full[shock_mask & (df_reg_full["regime"] == "Shock")]

# Forecast as DataFrame
forecast_df = pd.DataFrame({
    "date":  pd.to_datetime(result["forecast_dates"]),
    "price": result["forecast_series"],
})
band = forecast_df["price"] * 0.08
upper_band = forecast_df["price"] + band
lower_band = forecast_df["price"] - band


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 1: Price History + Forecast
#
#  Layout fix — rangeselector vs legend overlap:
#  ┌─────────────────────────────────────────────────────────┐
#  │  [1M][3M][6M][1Y][All]  ← rangeselector (always left)  │
#  │                          Legend ▶ (moved to right) →   │
#  │  ░░░░░░░░░░░░░░░ plot area ░░░░░░░░░░░░░░░░░░░░░░░░░░ │
#  └─────────────────────────────────────────────────────────┘
#  Plotly pins the rangeselector to the top-left of the x-axis and it cannot
#  be repositioned.  Setting legend x=1 / xanchor="right" moves it to the
#  opposite corner so the two never collide regardless of chart width.
#  margin.t is raised from 50 → 68 so both rows of items clear the plot edge.
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Price History & 7-Day AI Forecast")
st.caption(
    f"Showing {view_start.strftime('%d %b %Y')} → {view_end.strftime('%d %b %Y')} "
    f"({len(df_view):,} data points). Use the range selector below the chart to zoom."
)

fig = go.Figure()

# 1. Confidence band (behind everything)
fig.add_trace(go.Scatter(
    x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
    y=list(upper_band) + list(lower_band[::-1]),
    fill="toself",
    fillcolor="rgba(246,166,35,0.10)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Confidence Band",
    hoverinfo="skip",
    showlegend=True,
))

# 2. Actual price line
fig.add_trace(go.Scatter(
    x=df_view["Arrival_Date"],
    y=df_view["Modal_Price"],
    name="Actual Price",
    line=dict(color="#58a6ff", width=2),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: ₹%{y:,.0f}<extra></extra>",
))

# 3. Shock event markers (from full-data detection, filtered to view)
if len(shock_view) > 0:
    fig.add_trace(go.Scatter(
        x=shock_view["Arrival_Date"],
        y=shock_view["Modal_Price"],
        mode="markers",
        name="⚡ Shock Event",
        marker=dict(
            color="#ff6b6b", size=9, symbol="x-thin",
            line=dict(width=2.5, color="#ff6b6b")
        ),
        hovertemplate="<b>SHOCK ⚡</b><br>%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>",
    ))

# 4. Forecast line
fig.add_trace(go.Scatter(
    x=forecast_df["date"],
    y=forecast_df["price"],
    name="7-Day Forecast",
    line=dict(color="#f6a623", width=2.5, dash="dot"),
    hovertemplate="<b>Forecast</b><br>%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>",
))

# 5. Target date diamond
fig.add_trace(go.Scatter(
    x=[pd.Timestamp(predict_date)],
    y=[target_price],
    mode="markers+text",
    name=f"Prediction ({predict_date.strftime('%d %b')})",
    text=[f" ₹{target_price:,.0f}"],
    textposition="middle right",
    textfont=dict(color="#f6a623", size=13, family="Space Mono"),
    marker=dict(
        color="#f6a623", size=15, symbol="diamond",
        line=dict(color="#0d1117", width=2)
    ),
    hovertemplate=(
        f"<b>Prediction for {predict_date.strftime('%d %b %Y')}</b>"
        f"<br>₹{target_price:,.0f}<extra></extra>"
    ),
))

# 6. Vertical separator at last actual date (using shapes, not add_vline)
last_ts = pd.Timestamp(last_date)
fig.add_shape(
    type="line",
    x0=last_ts, x1=last_ts, y0=0, y1=1, yref="paper",
    line=dict(color="rgba(139,148,158,0.35)", width=1, dash="dot"),
)
fig.add_annotation(
    x=last_ts, y=0.98, yref="paper",
    text=" Forecast ▶",
    showarrow=False,
    font=dict(color="#8b949e", size=10, family="Sora"),
    xanchor="left",
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.85)",
    font=dict(color="#e6edf3", family="Sora"),
    xaxis=dict(
        gridcolor="#21262d",
        tickformat="%d %b %Y",
        rangeselector=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#e6edf3", size=11),
            buttons=[
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
        rangeslider=dict(
            visible=True,
            bgcolor="#161b22",
            thickness=0.06,
        ),
        range=[
            str(view_start - timedelta(days=3)),
            str(forecast_dates[-1] + timedelta(days=3)),
        ],
    ),
    yaxis=dict(
        gridcolor="#21262d",
        tickprefix="₹",
        title="Modal Price (₹ / quintal)",
    ),
    # ── FIX: legend moved to top-RIGHT so it never overlaps the ──────────
    # rangeselector buttons which Plotly always pins to the top-LEFT.
    # xanchor="right" + x=1  places it against the right edge of the chart.
    # yanchor="bottom" + y=1.02 keeps it just above the plot area, same row.
    # margin.t raised 50 → 68 px so both items have comfortable clearance.
    legend=dict(
        bgcolor="rgba(22,27,34,0.9)",
        bordercolor="#30363d",
        borderwidth=1,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",   # ← was "left" (same corner as rangeselector)
        x=1,               # ← was 0  (now pinned to right edge)
    ),
    margin=dict(l=10, r=10, t=68, b=10),   # ← t was 50, now 68 for clearance
    height=480,
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS 2-4: VMD Component Decomposition — 3 separate charts
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔬 Signal Decomposition — 3 Independent Components")
st.caption(
    "VMD splits the raw price series into Trend + Seasonal + Shock components. "
    "Each component is fed into a different AI model specialised for its pattern type."
)

if vmd_ok:
    # Filter modes to view window
    modes_view = modes_df[
        (modes_df.index.date >= view_start) &
        (modes_df.index.date <= view_end)
    ]

    if len(modes_view) < 2:
        st.info("Widen the date range to see decomposition charts.")
    else:
        mode_meta = [
            ("Trend",    "#58a6ff", "pill-blue",  "LSTM",
             "Long-term price direction — driven by supply/demand fundamentals"),
            ("Seasonal", "#39d353", "pill-green", "CNN-LSTM",
             "Cyclical harvest & weekly market patterns"),
            ("Shock",    "#ff6b6b", "pill-red",   "Transformer",
             "Sudden supply disruptions, weather events, policy changes"),
        ]

        for (col_name, color, pill_cls, model_name, desc) in mode_meta:
            if col_name not in modes_view.columns:
                continue

            series = modes_view[col_name]
            is_centered = col_name in ("Seasonal", "Shock")  # oscillates around 0

            # Hex → rgba fill
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            fill_color = f"rgba({r},{g},{b},0.08)"

            fm = go.Figure()
            fm.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                name=col_name,
                line=dict(color=color, width=1.6),
                fill="tozeroy",
                fillcolor=fill_color,
                hovertemplate=(
                    f"<b>{col_name}</b><br>"
                    "%{x|%d %b %Y}<br>"
                    "₹%{y:,.1f}<extra></extra>"
                ),
            ))

            # Zero reference line for oscillating components
            if is_centered:
                fm.add_hline(
                    y=0,
                    line=dict(color="rgba(139,148,158,0.25)", width=1, dash="dot"),
                )

            fm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,27,34,0.85)",
                font=dict(color="#e6edf3", family="Sora"),
                title=dict(
                    text=(
                        f"<b>{col_name} Component</b>"
                        f"  <span style='font-size:11px;color:#8b949e;'>"
                        f"→ {model_name} &nbsp;|&nbsp; {desc}"
                        f"</span>"
                    ),
                    font=dict(size=13, color="#e6edf3"),
                    x=0, pad=dict(l=0),
                ),
                xaxis=dict(
                    gridcolor="#21262d",
                    tickformat="%d %b %Y",
                    range=[str(view_start), str(view_end)],
                ),
                yaxis=dict(
                    gridcolor="#21262d",
                    tickprefix="₹",
                ),
                margin=dict(l=10, r=10, t=48, b=10),
                height=210,
                showlegend=False,
                hovermode="x unified",
            )
            st.plotly_chart(fm, use_container_width=True)
else:
    st.warning("VMD decomposition failed. Charts unavailable.")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.subheader(f"🎯 Prediction for {predict_date.strftime('%d %b %Y')}")

# Row: Prediction hero + 3 brain cards
col_hero, col_t, col_s, col_k = st.columns([1.1, 1, 1, 1])

with col_hero:
    badge_color = regime_color
    st.markdown(f"""
<div class="pred-hero">
    <div class="pred-hero-label">AI Price Prediction</div>
    <div class="pred-hero-price">₹{target_price:,.0f}</div>
    <div class="pred-hero-label" style="margin-top:8px;">
        {predict_date.strftime('%d %b %Y')}
    </div>
    <div class="pred-hero-label" style="margin-top:4px;">
        <span style="color:{badge_color};font-weight:700;">
            {regime_icon} {regime} Market
        </span>
    </div>
    <div class="pred-hero-label" style="margin-top:4px;">
        Confidence: <strong style="color:#e6edf3;">{result['confidence']}%</strong>
    </div>
</div>
""", unsafe_allow_html=True)

with col_t:
    d = decomp.get("trend", {})
    val   = d.get("value", 0)
    cpct  = d.get("contribution_pct", 0)
    label = d.get("label", "")
    sign  = "▲" if val >= 0 else "▼"
    st.markdown(f"""
<div class="card">
    <div class="pill pill-blue">LSTM · TREND</div>
    <div class="card-title">📈 Trend Component</div>
    <div class="card-value" style="color:#58a6ff;">{sign} ₹{abs(val):,.0f}</div>
    <div class="card-sub">Weight: {cpct:.0f}% of forecast</div>
    <div class="card-sub">{label}</div>
</div>
""", unsafe_allow_html=True)

with col_s:
    d = decomp.get("seasonal", {})
    val   = d.get("value", 0)
    cpct  = d.get("contribution_pct", 0)
    label = d.get("label", "")
    sign  = "▲" if val >= 0 else "▼"
    st.markdown(f"""
<div class="card">
    <div class="pill pill-green">CNN-LSTM · SEASONAL</div>
    <div class="card-title">🔄 Seasonal Component</div>
    <div class="card-value" style="color:#39d353;">{sign} ₹{abs(val):,.0f}</div>
    <div class="card-sub">Weight: {cpct:.0f}% of forecast</div>
    <div class="card-sub">{label}</div>
</div>
""", unsafe_allow_html=True)

with col_k:
    d = decomp.get("shock", {})
    val   = d.get("value", 0)
    cpct  = d.get("contribution_pct", 0)
    label = d.get("label", "")
    sign  = "▲" if val >= 0 else "▼"
    st.markdown(f"""
<div class="card">
    <div class="pill pill-red">TRANSFORMER · SHOCK</div>
    <div class="card-title">⚡ Shock Component</div>
    <div class="card-value" style="color:#ff6b6b;">{sign} ₹{abs(val):,.0f}</div>
    <div class="card-sub">Weight: {cpct:.0f}% of forecast</div>
    <div class="card-sub">{label}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row: Donut chart + Forecast table
dcol, tcol = st.columns([1, 1.5])

with dcol:
    pct_vals = [decomp.get(k, {}).get("contribution_pct", 0)
                for k in ("trend", "seasonal", "shock")]
    fig_pie = go.Figure(go.Pie(
        labels=["Trend", "Seasonal", "Shock"],
        values=pct_vals,
        hole=0.60,
        marker=dict(colors=["#58a6ff", "#39d353", "#ff6b6b"],
                    line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#e6edf3", size=12),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        direction="clockwise",
        sort=False,
    ))
    fig_pie.add_annotation(
        text=f"₹{target_price:,.0f}",
        x=0.5, y=0.5,
        font=dict(size=16, color="#f6a623", family="Space Mono"),
        showarrow=False,
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
        font=dict(color="#e6edf3"),
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    y=-0.15, x=0.5, xanchor="center"),
        title=dict(text="Model Contributions", font=dict(size=12), x=0.5),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with tcol:
    st.markdown("**7-Day Forecast**")
    rows = []
    for i, (fd, fp) in enumerate(zip(forecast_dates, forecast_prices)):
        chg = (fp - last_actual) / (last_actual + 1e-8) * 100
        is_target = (fd == predict_date)
        rows.append({
            "Date":        fd.strftime("%d %b %Y"),
            "Predicted":   f"₹{fp:,.0f}",
            "vs Today":    f"{chg:+.1f}%",
            "Selected":    "◀ Your pick" if is_target else "",
        })
    fc_df = pd.DataFrame(rows)
    st.dataframe(fc_df, use_container_width=True, hide_index=True, height=260)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  MARKET ADVICE BANNER
# ─────────────────────────────────────────────────────────────────────────────
advice_map = {
    "Calm": (
        "✅", "#1a3326", "#39d353",
        "Stable Market",
        "Trend and seasonal cycles are the dominant drivers. "
        "Prices are predictable — a good time to plan bulk procurement "
        "or negotiate forward contracts with confidence."
    ),
    "Normal": (
        "🟡", "#1c2d4a", "#58a6ff",
        "Normal Conditions",
        "Market is active but within expected seasonal bounds. "
        "Monitor upcoming harvest arrivals and weather reports. "
        "Moderate caution advised for large commitments."
    ),
    "Shock": (
        "🔴", "#3d1f1f", "#ff6b6b",
        "Shock Detected — High Volatility",
        "A significant price disruption has been detected. "
        "Transformer model is weighted heavily in this regime. "
        "Avoid large buy/sell decisions until prices stabilise over 3–5 days."
    ),
}
icon, bg, border, title, text = advice_map.get(regime, advice_map["Normal"])
st.markdown(f"""
<div class="regime-banner" style="background:{bg}22;border:1px solid {border};">
    <div class="regime-icon">{icon}</div>
    <div>
        <div class="regime-title" style="color:{border};">{title}</div>
        <div class="regime-text">{text}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<p style="color:#8b949e;font-size:.75rem;text-align:center;margin-top:8px;">'
    f'AgriPrice AI &nbsp;·&nbsp; VMD + LSTM + CNN-LSTM + Transformer + Ensemble Gating &nbsp;·&nbsp; '
    f'{len(df):,} records &nbsp;·&nbsp; '
    f'{first_date.strftime("%d %b %Y")} → {last_date.strftime("%d %b %Y")} &nbsp;·&nbsp; '
    f'RMSE ₹{metrics.get("rmse", "—")} &nbsp;·&nbsp; MAPE {metrics.get("mape", "—")}%'
    f'</p>',
    unsafe_allow_html=True,
)
